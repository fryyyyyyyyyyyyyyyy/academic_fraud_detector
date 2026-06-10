"""
MinerU API client for converting PDF files to Markdown.

Uses the Precision Extract API in batch-upload mode:
1. Request a temporary upload URL.
2. Upload the local PDF bytes.
3. Poll the batch result endpoint.
4. Download the result zip and cache full.md plus image assets.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any, Optional
from urllib.parse import unquote
import uuid
import zipfile

import requests
from dotenv import load_dotenv

from .api_client import get_session

logger = logging.getLogger(__name__)

MINERU_BASE_URL = "https://mineru.net"
MINERU_MODEL_VERSION = "vlm"
MINERU_CACHE_DIR = Path.home() / ".cache" / "academic_fraud_detector" / "mineru"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60
DEFAULT_POLL_INTERVAL_SECONDS = 2.0
DEFAULT_POLL_TIMEOUT_SECONDS = 300.0
MAX_MINERU_ZIP_ENTRIES = 2000
MAX_MINERU_ASSET_BYTES = 50 * 1024 * 1024
MAX_MINERU_TOTAL_ASSET_BYTES = 500 * 1024 * 1024
MINERU_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
}
_PENDING_STATES = {"waiting-file", "pending", "running", "converting"}
_DONE_STATES = {"done"}
_FAILED_STATES = {"failed", "error"}
_PLACEHOLDER_VALUES = {
    "your_mineru_api_key_here",
    "your_mineru_api_token_here",
    "填入你的mineru_api_key",
    "填入你的mineru_api_token",
}
_IMAGE_LINK_RE = re.compile(r"!\[([^\]]*)\]\(([^)\n]+)\)")


@dataclass
class MinerUMarkdownResult:
    """Markdown plus cached image assets extracted from a MinerU result zip."""

    markdown: str
    full_md_path: str
    raw_full_md_path: str
    cache_dir: str
    images: list[dict[str, Any]]
    zip_path: Optional[str] = None


class MinerUError(Exception):
    """Base error for MinerU extraction failures."""


class MinerUConfigError(MinerUError):
    """Raised when MinerU API key/configuration is missing or unusable."""


class MinerUTimeoutError(MinerUError):
    """Raised when MinerU polling exceeds the configured timeout."""


class MinerUResultError(MinerUError):
    """Raised when MinerU returns an unusable result payload or zip."""


def get_mineru_api_key() -> Optional[str]:
    """Read MinerU API key from .env/environment without logging the value."""
    load_dotenv()
    key = os.getenv("MINERU_API_KEY") or os.getenv("MINERU_API_TOKEN")
    if not key:
        return None

    key = key.strip()
    if not key or key.lower() in _PLACEHOLDER_VALUES:
        return None
    return key


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = float(value)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", name, value, default)
        return default
    return parsed if parsed >= 0 else default


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "*/*",
    }


def _result_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Accept": "*/*",
    }


def _response_json(response: requests.Response, context: str) -> dict[str, Any]:
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError as exc:
        raise MinerUResultError(f"MinerU {context} response is not valid JSON") from exc

    if not isinstance(payload, dict):
        raise MinerUResultError(f"MinerU {context} response must be a JSON object")

    code = payload.get("code")
    if code not in (None, 0):
        msg = payload.get("msg") or payload.get("message") or "unknown error"
        raise MinerUResultError(f"MinerU {context} failed: code={code}, msg={msg}")

    return payload


def _extract_upload_url(item: Any) -> Optional[str]:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("upload_url", "url", "file_url", "put_url"):
            value = item.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _parse_batch_upload_response(payload: dict[str, Any]) -> tuple[str, str]:
    data = payload.get("data")
    if not isinstance(data, dict):
        raise MinerUResultError("MinerU upload-url response missing data object")

    batch_id = data.get("batch_id")
    if not isinstance(batch_id, str) or not batch_id:
        raise MinerUResultError("MinerU upload-url response missing batch_id")

    file_urls = data.get("file_urls") or data.get("urls") or data.get("files")
    if not isinstance(file_urls, list) or not file_urls:
        raise MinerUResultError("MinerU upload-url response missing file upload URL")

    upload_url = _extract_upload_url(file_urls[0])
    if not upload_url:
        raise MinerUResultError("MinerU upload-url response contains no usable upload URL")

    return batch_id, upload_url


def _select_extract_result(
    payload: dict[str, Any],
    *,
    data_id: str,
    file_name: str,
) -> dict[str, Any]:
    data = payload.get("data")
    if not isinstance(data, dict):
        raise MinerUResultError("MinerU batch result response missing data object")

    results = data.get("extract_result") or data.get("extract_results") or data.get("results")
    if isinstance(results, dict):
        results = [results]
    if not isinstance(results, list) or not results:
        raise MinerUResultError("MinerU batch result response missing extract_result")

    for item in results:
        if isinstance(item, dict) and item.get("data_id") == data_id:
            return item

    for item in results:
        if isinstance(item, dict) and item.get("file_name") == file_name:
            return item

    first = results[0]
    if not isinstance(first, dict):
        raise MinerUResultError("MinerU batch result item must be an object")
    return first


def _poll_batch_result(
    session: requests.Session,
    *,
    api_key: str,
    batch_id: str,
    data_id: str,
    file_name: str,
    request_timeout_seconds: float,
    poll_interval_seconds: float,
    poll_timeout_seconds: float,
) -> str:
    url = f"{MINERU_BASE_URL}/api/v4/extract-results/batch/{batch_id}"
    started_at = time.monotonic()

    while True:
        response = session.get(
            url,
            headers=_result_headers(api_key),
            timeout=request_timeout_seconds,
        )
        payload = _response_json(response, "batch-result")
        item = _select_extract_result(payload, data_id=data_id, file_name=file_name)
        state = str(item.get("state") or "").lower()

        if state in _DONE_STATES:
            full_zip_url = item.get("full_zip_url")
            if not isinstance(full_zip_url, str) or not full_zip_url:
                raise MinerUResultError("MinerU completed but did not return full_zip_url")
            return full_zip_url

        if state in _FAILED_STATES:
            err_msg = item.get("err_msg") or item.get("error") or "unknown error"
            raise MinerUResultError(f"MinerU extraction failed: {err_msg}")

        if state and state not in _PENDING_STATES:
            raise MinerUResultError(f"MinerU returned unknown extraction state: {state}")

        if time.monotonic() - started_at >= poll_timeout_seconds:
            raise MinerUTimeoutError(
                f"MinerU extraction timed out after {poll_timeout_seconds:g} seconds"
            )

        if poll_interval_seconds > 0:
            time.sleep(poll_interval_seconds)


def _safe_slug(value: str, default: str = "paper", max_length: int = 80) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not slug:
        slug = default
    return slug[:max_length]


def _cache_output_dir(file_name: str, data_id: str, cache_dir: Optional[str | Path]) -> Path:
    root = Path(cache_dir) if cache_dir else MINERU_CACHE_DIR
    safe_stem = _safe_slug(Path(file_name or "paper.pdf").stem)
    return root / f"{safe_stem}_{data_id}"


def _normalize_zip_path(name: str) -> Optional[str]:
    if not name or "\x00" in name:
        return None

    normalized = name.replace("\\", "/")
    if normalized.startswith("/") or normalized.startswith("//"):
        return None
    if re.match(r"^[A-Za-z]:", normalized):
        return None

    parts = []
    for part in normalized.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            return None
        parts.append(part)

    if not parts:
        return None
    return "/".join(parts)


def _find_full_md_member(names: list[str]) -> str:
    normalized_pairs = [(_normalize_zip_path(name), name) for name in names]
    for normalized, original in normalized_pairs:
        if normalized == "full.md":
            return original

    for normalized, original in normalized_pairs:
        if normalized and normalized.rstrip("/").split("/")[-1].lower() == "full.md":
            return original

    raise MinerUResultError("MinerU result zip does not contain full.md")


def _decode_markdown(raw: bytes) -> str:
    try:
        markdown = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        markdown = raw.decode("utf-8", errors="replace")

    if not markdown.strip():
        raise MinerUResultError("MinerU full.md is empty")
    return markdown


def _image_size_and_format(image_bytes: bytes, suffix: str) -> tuple[Optional[int], Optional[int], str]:
    fallback_format = suffix.lstrip(".").upper() or "UNKNOWN"
    try:
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
            image_format = image.format or fallback_format
            return int(width), int(height), str(image_format).upper()
    except Exception as exc:
        logger.debug("Failed to inspect MinerU image dimensions: %s", exc)
        return None, None, fallback_format


def _asset_lookup_keys(path: str) -> list[str]:
    path_without_fragment = path.split("#", 1)[0]
    path_without_query = path_without_fragment.split("?", 1)[0]
    decoded = unquote(path_without_query).replace("\\", "/")
    while decoded.startswith("./"):
        decoded = decoded[2:]
    normalized = _normalize_zip_path(decoded)
    return [normalized] if normalized else []


def _is_external_markdown_target(target: str) -> bool:
    lowered = target.strip().lower()
    return lowered.startswith(("http://", "https://", "data:", "file:", "#"))


def _split_markdown_link_target(raw_target: str) -> tuple[str, str, bool]:
    stripped = raw_target.strip()
    if stripped.startswith("<") and ">" in stripped:
        closing = stripped.index(">")
        return stripped[1:closing], stripped[closing + 1 :], True

    if not stripped:
        return "", "", False

    parts = stripped.split(maxsplit=1)
    target = parts[0]
    suffix = f" {parts[1]}" if len(parts) > 1 else ""
    return target, suffix, False


def _rewrite_markdown_image_links(markdown: str, asset_map: dict[str, str]) -> str:
    if not asset_map:
        return markdown

    basename_to_local: dict[str, Optional[str]] = {}
    for original_path, local_path in asset_map.items():
        basename = original_path.rsplit("/", 1)[-1]
        if basename in basename_to_local and basename_to_local[basename] != local_path:
            basename_to_local[basename] = None
        else:
            basename_to_local[basename] = local_path

    def replace(match: re.Match[str]) -> str:
        alt_text = match.group(1)
        raw_target = match.group(2)
        target, suffix, was_angle_wrapped = _split_markdown_link_target(raw_target)
        if not target or _is_external_markdown_target(target):
            return match.group(0)

        local_path = None
        for key in _asset_lookup_keys(target):
            local_path = asset_map.get(key)
            if local_path:
                break
            basename_match = basename_to_local.get(key.rsplit("/", 1)[-1])
            if basename_match:
                local_path = basename_match
                break

        if not local_path:
            return match.group(0)

        if suffix or was_angle_wrapped or " " in local_path:
            rewritten_target = f"<{local_path}>{suffix}"
        else:
            rewritten_target = f"{local_path}{suffix}"
        return f"![{alt_text}]({rewritten_target})"

    return _IMAGE_LINK_RE.sub(replace, markdown)


def _cache_mineru_images(
    archive: zipfile.ZipFile,
    *,
    full_md_member: str,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    assets_dir = output_dir / "assets" / "images"
    assets_dir.mkdir(parents=True, exist_ok=True)

    full_md_normalized = _normalize_zip_path(full_md_member)
    images: list[dict[str, Any]] = []
    asset_map: dict[str, str] = {}
    total_asset_bytes = 0

    for info in archive.infolist():
        if info.is_dir():
            continue

        normalized = _normalize_zip_path(info.filename)
        if not normalized or normalized == full_md_normalized:
            continue

        suffix = Path(normalized).suffix.lower()
        if suffix not in MINERU_IMAGE_EXTENSIONS:
            continue

        if info.file_size > MAX_MINERU_ASSET_BYTES:
            logger.warning("Skipping oversized MinerU image asset: %s", normalized)
            continue

        if total_asset_bytes + info.file_size > MAX_MINERU_TOTAL_ASSET_BYTES:
            logger.warning("Skipping remaining MinerU image assets after total size limit")
            break

        try:
            image_bytes = archive.read(info)
        except Exception as exc:
            logger.warning("Failed to read MinerU image asset %s: %s", normalized, exc)
            continue

        if len(image_bytes) > MAX_MINERU_ASSET_BYTES:
            logger.warning("Skipping oversized MinerU image asset after read: %s", normalized)
            continue

        index = len(images) + 1
        filename = f"image_{index:04d}{suffix}"
        filepath = assets_dir / filename
        try:
            filepath.write_bytes(image_bytes)
        except OSError as exc:
            logger.warning("Failed to cache MinerU image asset %s: %s", normalized, exc)
            continue

        total_asset_bytes += len(image_bytes)
        width, height, image_format = _image_size_and_format(image_bytes, suffix)
        local_markdown_path = filepath.absolute().as_posix()
        image_entry = {
            "filename": filename,
            "filepath": str(filepath.absolute()),
            "format": image_format,
            "width": width,
            "height": height,
            "page_number": None,
            "xref": None,
            "size_bytes": len(image_bytes),
            "source": "mineru",
            "original_path": normalized,
            "markdown_path": local_markdown_path,
        }
        images.append(image_entry)
        asset_map[normalized] = local_markdown_path

    return images, asset_map


def _write_manifest(output_dir: Path, result: MinerUMarkdownResult) -> None:
    manifest_path = output_dir / "manifest.json"
    payload = {
        "full_md_path": result.full_md_path,
        "raw_full_md_path": result.raw_full_md_path,
        "cache_dir": result.cache_dir,
        "zip_path": result.zip_path,
        "images": result.images,
    }
    try:
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.debug("Failed to write MinerU cache manifest: %s", exc)


def _extract_full_md_and_assets_from_zip(
    zip_bytes: bytes,
    *,
    output_dir: Path,
    save_zip: bool = True,
) -> MinerUMarkdownResult:
    try:
        archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as exc:
        raise MinerUResultError("MinerU result is not a valid zip archive") from exc

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise MinerUResultError("Failed to create MinerU cache directory") from exc

    with archive:
        names = archive.namelist()
        if len(names) > MAX_MINERU_ZIP_ENTRIES:
            raise MinerUResultError("MinerU result zip contains too many entries")

        full_md_member = _find_full_md_member(names)
        raw_markdown = _decode_markdown(archive.read(full_md_member))

        zip_path: Optional[Path] = None
        if save_zip:
            zip_path = output_dir / "result.zip"
            try:
                zip_path.write_bytes(zip_bytes)
            except OSError as exc:
                raise MinerUResultError("Failed to cache MinerU result zip") from exc

        raw_full_md_path = output_dir / "full_raw.md"
        try:
            raw_full_md_path.write_text(raw_markdown, encoding="utf-8")
        except OSError as exc:
            raise MinerUResultError("Failed to cache MinerU raw Markdown") from exc

        images, asset_map = _cache_mineru_images(
            archive,
            full_md_member=full_md_member,
            output_dir=output_dir,
        )
        rewritten_markdown = _rewrite_markdown_image_links(raw_markdown, asset_map)

        full_md_path = output_dir / "full.md"
        try:
            full_md_path.write_text(rewritten_markdown, encoding="utf-8")
        except OSError as exc:
            raise MinerUResultError("Failed to cache MinerU rewritten Markdown") from exc

    result = MinerUMarkdownResult(
        markdown=rewritten_markdown,
        full_md_path=str(full_md_path.absolute()),
        raw_full_md_path=str(raw_full_md_path.absolute()),
        cache_dir=str(output_dir.absolute()),
        images=images,
        zip_path=str(zip_path.absolute()) if zip_path else None,
    )
    _write_manifest(output_dir, result)
    return result


def _extract_full_md_from_zip(zip_bytes: bytes) -> str:
    try:
        archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as exc:
        raise MinerUResultError("MinerU result is not a valid zip archive") from exc

    with archive:
        names = archive.namelist()
        full_md_member = _find_full_md_member(names)
        return _decode_markdown(archive.read(full_md_member))


def _download_full_markdown(
    session: requests.Session,
    *,
    full_zip_url: str,
    request_timeout_seconds: float,
) -> str:
    response = session.get(full_zip_url, timeout=request_timeout_seconds)
    response.raise_for_status()
    return _extract_full_md_from_zip(response.content)


def _download_full_markdown_result(
    session: requests.Session,
    *,
    full_zip_url: str,
    request_timeout_seconds: float,
    output_dir: Path,
    save_zip: bool = True,
) -> MinerUMarkdownResult:
    response = session.get(full_zip_url, timeout=request_timeout_seconds)
    response.raise_for_status()
    return _extract_full_md_and_assets_from_zip(
        response.content,
        output_dir=output_dir,
        save_zip=save_zip,
    )


def extract_pdf_markdown_with_mineru_assets(
    pdf_content: bytes,
    file_name: str,
    max_pages: Optional[int] = None,
    session: Optional[requests.Session] = None,
    poll_interval_seconds: Optional[float] = None,
    poll_timeout_seconds: Optional[float] = None,
    request_timeout_seconds: Optional[float] = None,
    cache_dir: Optional[str | Path] = None,
    save_zip: bool = True,
) -> MinerUMarkdownResult:
    """
    Convert a local PDF byte buffer to Markdown with MinerU VLM and cache assets.

    The returned Markdown is the rewritten local-cache version whose image links
    point at saved files. Raises MinerUError subclasses on MinerU/config/result
    failure so callers can fall back to local extraction.
    """
    api_key = get_mineru_api_key()
    if not api_key:
        raise MinerUConfigError("MINERU_API_KEY is not configured")

    if not pdf_content:
        raise MinerUResultError("PDF content is empty")

    request_timeout = request_timeout_seconds or _env_float(
        "MINERU_REQUEST_TIMEOUT_SECONDS", DEFAULT_REQUEST_TIMEOUT_SECONDS
    )
    poll_interval = (
        poll_interval_seconds
        if poll_interval_seconds is not None
        else _env_float("MINERU_POLL_INTERVAL_SECONDS", DEFAULT_POLL_INTERVAL_SECONDS)
    )
    poll_timeout = (
        poll_timeout_seconds
        if poll_timeout_seconds is not None
        else _env_float("MINERU_POLL_TIMEOUT_SECONDS", DEFAULT_POLL_TIMEOUT_SECONDS)
    )

    client = session or get_session(total_retries=3, backoff_factor=1.0)
    data_id = f"afd_{uuid.uuid4().hex}"
    safe_file_name = file_name or "paper.pdf"

    file_entry: dict[str, Any] = {
        "name": safe_file_name,
        "data_id": data_id,
    }
    if max_pages is not None and max_pages > 0:
        file_entry["page_ranges"] = f"1-{max_pages}"

    upload_response = client.post(
        f"{MINERU_BASE_URL}/api/v4/file-urls/batch",
        headers=_headers(api_key),
        json={
            "files": [file_entry],
            "model_version": MINERU_MODEL_VERSION,
        },
        timeout=request_timeout,
    )
    upload_payload = _response_json(upload_response, "upload-url")
    batch_id, upload_url = _parse_batch_upload_response(upload_payload)

    put_response = client.put(upload_url, data=pdf_content, timeout=request_timeout)
    put_response.raise_for_status()

    full_zip_url = _poll_batch_result(
        client,
        api_key=api_key,
        batch_id=batch_id,
        data_id=data_id,
        file_name=safe_file_name,
        request_timeout_seconds=request_timeout,
        poll_interval_seconds=poll_interval,
        poll_timeout_seconds=poll_timeout,
    )
    output_dir = _cache_output_dir(safe_file_name, data_id, cache_dir)
    result = _download_full_markdown_result(
        client,
        full_zip_url=full_zip_url,
        request_timeout_seconds=request_timeout,
        output_dir=output_dir,
        save_zip=save_zip,
    )
    logger.info(
        "MinerU PDF-to-Markdown extraction succeeded for %s (%d cached images)",
        safe_file_name,
        len(result.images),
    )
    return result


def extract_pdf_markdown_with_mineru(
    pdf_content: bytes,
    file_name: str,
    max_pages: Optional[int] = None,
    session: Optional[requests.Session] = None,
    poll_interval_seconds: Optional[float] = None,
    poll_timeout_seconds: Optional[float] = None,
    request_timeout_seconds: Optional[float] = None,
    cache_dir: Optional[str | Path] = None,
    save_zip: bool = True,
) -> str:
    """
    Convert a local PDF byte buffer to Markdown with MinerU VLM.

    Raises MinerUError subclasses on any MinerU/config/result failure. Callers are
    expected to catch those errors and fall back to local extraction.
    """
    result = extract_pdf_markdown_with_mineru_assets(
        pdf_content,
        file_name=file_name,
        max_pages=max_pages,
        session=session,
        poll_interval_seconds=poll_interval_seconds,
        poll_timeout_seconds=poll_timeout_seconds,
        request_timeout_seconds=request_timeout_seconds,
        cache_dir=cache_dir,
        save_zip=save_zip,
    )
    return result.markdown
