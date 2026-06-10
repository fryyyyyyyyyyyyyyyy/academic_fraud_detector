"""Tests for the MinerU PDF-to-Markdown client."""

import base64
import io
from pathlib import Path
import zipfile

import pytest
import requests

from academic_fraud_detector.utils.mineru_client import (
    MinerUConfigError,
    MinerUResultError,
    MinerUTimeoutError,
    extract_pdf_markdown_with_mineru,
    extract_pdf_markdown_with_mineru_assets,
)


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/"
    "l3X6V8AAAAASUVORK5CYII="
)


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, content=b"", text=""):
        self.status_code = status_code
        self._json_data = json_data
        self.content = content
        self.text = text

    def json(self):
        if self._json_data is None:
            raise ValueError("no json")
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append(("post", url, kwargs))
        return self.responses.pop(0)

    def put(self, url, **kwargs):
        self.calls.append(("put", url, kwargs))
        return self.responses.pop(0)

    def get(self, url, **kwargs):
        self.calls.append(("get", url, kwargs))
        return self.responses.pop(0)


def make_zip(entries: dict[str, str | bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in entries.items():
            if isinstance(content, str):
                content = content.encode("utf-8")
            archive.writestr(name, content)
    return buffer.getvalue()


def make_zip_with_full_md(markdown: str, name: str = "full.md") -> bytes:
    return make_zip({name: markdown})


def upload_payload():
    return {
        "code": 0,
        "msg": "ok",
        "data": {
            "batch_id": "batch-123",
            "file_urls": ["https://upload.example/paper.pdf"],
        },
    }


def done_payload(full_zip_url="https://cdn.example/result.zip"):
    return {
        "code": 0,
        "msg": "ok",
        "data": {
            "batch_id": "batch-123",
            "extract_result": [
                {
                    "file_name": "paper.pdf",
                    "state": "done",
                    "full_zip_url": full_zip_url,
                    "err_msg": "",
                }
            ],
        },
    }


def mineru_success_session(zip_bytes: bytes) -> FakeSession:
    return FakeSession([
        FakeResponse(json_data=upload_payload()),
        FakeResponse(status_code=200),
        FakeResponse(json_data=done_payload()),
        FakeResponse(content=zip_bytes),
    ])


def test_extract_pdf_markdown_success(monkeypatch, tmp_path):
    monkeypatch.setenv("MINERU_API_KEY", "test-token")
    monkeypatch.setenv("MINERU_API_TOKEN", "")

    markdown = "# 标题\n\nMinerU Markdown 内容"
    session = mineru_success_session(make_zip_with_full_md(markdown))

    result = extract_pdf_markdown_with_mineru(
        b"%PDF-1.7 fake",
        file_name="paper.pdf",
        max_pages=2,
        session=session,
        poll_interval_seconds=0,
        cache_dir=tmp_path,
    )

    assert result == markdown

    post_call = session.calls[0]
    assert post_call[0] == "post"
    assert post_call[2]["headers"]["Authorization"] == "Bearer test-token"
    assert post_call[2]["json"]["model_version"] == "vlm"
    assert post_call[2]["json"]["files"][0]["name"] == "paper.pdf"
    assert post_call[2]["json"]["files"][0]["page_ranges"] == "1-2"

    put_call = session.calls[1]
    assert put_call[0] == "put"
    assert put_call[1] == "https://upload.example/paper.pdf"
    assert put_call[2]["data"] == b"%PDF-1.7 fake"


def test_extract_pdf_markdown_assets_cache_markdown_and_images(monkeypatch, tmp_path):
    monkeypatch.setenv("MINERU_API_KEY", "test-token")
    markdown = "# Paper\n\n![Figure 1](images/fig1.png)\n\nResults showed p = 0.03."
    session = mineru_success_session(make_zip({"full.md": markdown, "images/fig1.png": PNG_BYTES}))

    result = extract_pdf_markdown_with_mineru_assets(
        b"%PDF-1.7 fake",
        file_name="paper.pdf",
        session=session,
        poll_interval_seconds=0,
        cache_dir=tmp_path,
    )

    assert Path(result.raw_full_md_path).read_text(encoding="utf-8") == markdown
    assert Path(result.full_md_path).read_text(encoding="utf-8") == result.markdown
    assert result.zip_path is not None
    assert Path(result.zip_path).exists()
    assert len(result.images) == 1

    image = result.images[0]
    assert image["source"] == "mineru"
    assert image["original_path"] == "images/fig1.png"
    assert image["format"] == "PNG"
    assert image["size_bytes"] == len(PNG_BYTES)
    assert Path(image["filepath"]).exists()
    assert image["markdown_path"] in result.markdown
    assert "images/fig1.png" not in result.markdown


def test_extract_pdf_markdown_assets_skips_unsafe_zip_member(monkeypatch, tmp_path):
    monkeypatch.setenv("MINERU_API_KEY", "test-token")
    markdown = "# Paper\n\n![](images/good.png)\n\n![](../evil.png)"
    zip_bytes = make_zip({
        "full.md": markdown,
        "images/good.png": PNG_BYTES,
        "../evil.png": PNG_BYTES,
    })
    session = mineru_success_session(zip_bytes)

    result = extract_pdf_markdown_with_mineru_assets(
        b"%PDF-1.7 fake",
        file_name="paper.pdf",
        session=session,
        poll_interval_seconds=0,
        cache_dir=tmp_path,
    )

    assert len(result.images) == 1
    assert result.images[0]["original_path"] == "images/good.png"
    assert not (tmp_path / "evil.png").exists()
    assert not (tmp_path.parent / "evil.png").exists()


def test_extract_pdf_markdown_assets_without_images(monkeypatch, tmp_path):
    monkeypatch.setenv("MINERU_API_KEY", "test-token")
    markdown = "# Paper\n\nNo images here."
    session = mineru_success_session(make_zip_with_full_md(markdown))

    result = extract_pdf_markdown_with_mineru_assets(
        b"%PDF-1.7 fake",
        file_name="paper.pdf",
        session=session,
        poll_interval_seconds=0,
        cache_dir=tmp_path,
    )

    assert result.markdown == markdown
    assert result.images == []
    assert Path(result.full_md_path).exists()
    assert Path(result.raw_full_md_path).exists()


def test_missing_api_key_raises_config_error(monkeypatch):
    monkeypatch.setenv("MINERU_API_KEY", "")
    monkeypatch.setenv("MINERU_API_TOKEN", "")
    session = FakeSession([])

    with pytest.raises(MinerUConfigError):
        extract_pdf_markdown_with_mineru(b"pdf", "paper.pdf", session=session)

    assert session.calls == []


def test_poll_until_done(monkeypatch, tmp_path):
    monkeypatch.setenv("MINERU_API_KEY", "test-token")
    monkeypatch.setattr("academic_fraud_detector.utils.mineru_client.time.sleep", lambda _: None)

    session = FakeSession([
        FakeResponse(json_data=upload_payload()),
        FakeResponse(status_code=200),
        FakeResponse(json_data={
            "code": 0,
            "data": {
                "extract_result": [{"file_name": "paper.pdf", "state": "running"}],
            },
        }),
        FakeResponse(json_data=done_payload()),
        FakeResponse(content=make_zip_with_full_md("# done")),
    ])

    result = extract_pdf_markdown_with_mineru(
        b"pdf",
        "paper.pdf",
        session=session,
        poll_interval_seconds=0,
        cache_dir=tmp_path,
    )

    assert result == "# done"
    get_calls = [call for call in session.calls if call[0] == "get"]
    assert len(get_calls) == 3  # two result polls + one zip download


def test_failed_state_raises_result_error(monkeypatch):
    monkeypatch.setenv("MINERU_API_KEY", "test-token")
    session = FakeSession([
        FakeResponse(json_data=upload_payload()),
        FakeResponse(status_code=200),
        FakeResponse(json_data={
            "code": 0,
            "data": {
                "extract_result": [
                    {"file_name": "paper.pdf", "state": "failed", "err_msg": "bad pdf"}
                ],
            },
        }),
    ])

    with pytest.raises(MinerUResultError, match="bad pdf"):
        extract_pdf_markdown_with_mineru(
            b"pdf",
            "paper.pdf",
            session=session,
            poll_interval_seconds=0,
        )


def test_missing_full_zip_url_raises_result_error(monkeypatch):
    monkeypatch.setenv("MINERU_API_KEY", "test-token")
    payload = done_payload(full_zip_url="")
    session = FakeSession([
        FakeResponse(json_data=upload_payload()),
        FakeResponse(status_code=200),
        FakeResponse(json_data=payload),
    ])

    with pytest.raises(MinerUResultError, match="full_zip_url"):
        extract_pdf_markdown_with_mineru(
            b"pdf",
            "paper.pdf",
            session=session,
            poll_interval_seconds=0,
        )


def test_zip_without_full_md_raises_result_error(monkeypatch, tmp_path):
    monkeypatch.setenv("MINERU_API_KEY", "test-token")
    session = FakeSession([
        FakeResponse(json_data=upload_payload()),
        FakeResponse(status_code=200),
        FakeResponse(json_data=done_payload()),
        FakeResponse(content=make_zip_with_full_md("not full", name="other.md")),
    ])

    with pytest.raises(MinerUResultError, match="full.md"):
        extract_pdf_markdown_with_mineru(
            b"pdf",
            "paper.pdf",
            session=session,
            poll_interval_seconds=0,
            cache_dir=tmp_path,
        )


def test_poll_timeout_raises_timeout_error(monkeypatch):
    monkeypatch.setenv("MINERU_API_KEY", "test-token")
    session = FakeSession([
        FakeResponse(json_data=upload_payload()),
        FakeResponse(status_code=200),
        FakeResponse(json_data={
            "code": 0,
            "data": {
                "extract_result": [{"file_name": "paper.pdf", "state": "running"}],
            },
        }),
    ])

    with pytest.raises(MinerUTimeoutError):
        extract_pdf_markdown_with_mineru(
            b"pdf",
            "paper.pdf",
            session=session,
            poll_interval_seconds=0,
            poll_timeout_seconds=0,
        )
