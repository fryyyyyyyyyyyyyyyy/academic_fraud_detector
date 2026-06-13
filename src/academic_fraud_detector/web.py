"""本地 WebUI 工作台入口。

该模块只使用 Python 标准库，作为现有 CLI 调查流程的薄适配层。
"""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import threading
import uuid
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from academic_fraud_detector.main import (
    _extract_risk_level_from_markdown,
    _extract_risk_score_from_markdown,
    _extract_title_from_markdown,
    run_investigation,
)
from academic_fraud_detector.utils.case_folder import discover_case_folder

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
STATIC_DIR = Path(__file__).with_name("web_static")
VALID_IDENTIFIER_TYPES = {
    "doi",
    "arxiv_id",
    "title",
    "url",
    "semantic_scholar_id",
    "local_pdf",
    "local_case",
}
REPORT_FILE_KEYS = {"md", "html", "json"}

JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.RLock()


def _now_iso() -> str:
    """返回便于前端展示的本地时间戳。"""
    return datetime.now().isoformat(timespec="seconds")


def _json_safe(value: Any) -> Any:
    """把 Path 等对象转换为 JSON 可序列化值。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    """返回可发送给浏览器的 job 视图。"""
    job_id = job["id"]
    file_paths = job.get("file_paths", {})
    file_links = {
        key: f"/api/jobs/{job_id}/files/{key}"
        for key in REPORT_FILE_KEYS
        if key in file_paths
    }
    return {
        "id": job_id,
        "status": job["status"],
        "message": job.get("message", ""),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "identifier_type": job.get("identifier_type"),
        "display_name": job.get("display_name"),
        "output_dir": job.get("output_dir"),
        "elapsed": job.get("elapsed"),
        "risk_level": job.get("risk_level"),
        "risk_score": job.get("risk_score"),
        "title": job.get("title"),
        "markdown_preview": job.get("markdown_preview"),
        "files": file_links,
        "file_paths": file_paths,
        "manifest": job.get("manifest"),
        "error": job.get("error"),
    }


def _error(message: str, *, details: Any = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"error": message}
    if details is not None:
        payload["details"] = _json_safe(details)
    return payload


def _default_output_dir(job_id: str) -> Path:
    return Path.cwd() / "reports" / "webui" / job_id


def _validate_job_payload(payload: dict[str, Any], job_id: str) -> tuple[int, dict[str, Any]]:
    identifier_type = str(payload.get("identifier_type", "")).strip()
    paper_identifier = str(payload.get("paper_identifier", "")).strip()
    output_dir_value = str(payload.get("output_dir", "")).strip()

    if identifier_type not in VALID_IDENTIFIER_TYPES:
        return HTTPStatus.BAD_REQUEST, _error("不支持的调查类型。")
    if not paper_identifier:
        return HTTPStatus.BAD_REQUEST, _error("调查对象不能为空。")

    manifest = None
    normalized_identifier = paper_identifier

    if identifier_type == "local_pdf":
        pdf_path = Path(paper_identifier).expanduser()
        if not pdf_path.exists():
            return HTTPStatus.BAD_REQUEST, _error(f"PDF 文件不存在: {pdf_path}")
        if not pdf_path.is_file():
            return HTTPStatus.BAD_REQUEST, _error(f"PDF 路径不是文件: {pdf_path}")
        if pdf_path.suffix.lower() != ".pdf":
            return HTTPStatus.BAD_REQUEST, _error(f"文件必须是 PDF 格式: {pdf_path}")
        normalized_identifier = str(pdf_path)

    if identifier_type == "local_case":
        case_dir = Path(paper_identifier).expanduser()
        manifest = discover_case_folder(str(case_dir))
        if manifest.get("errors"):
            return HTTPStatus.BAD_REQUEST, _error("案例目录不可用。", details=manifest)
        normalized_identifier = str(case_dir)

    output_dir = Path(output_dir_value).expanduser() if output_dir_value else _default_output_dir(job_id)
    return HTTPStatus.OK, {
        "identifier_type": identifier_type,
        "paper_identifier": normalized_identifier,
        "output_dir": str(output_dir),
        "manifest": manifest,
    }


def _record_job_update(job_id: str, **updates: Any) -> None:
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(updates)


def _run_job(job_id: str) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        paper_identifier = job["paper_identifier"]
        identifier_type = job["identifier_type"]
        output_dir = job["output_dir"]

    _record_job_update(
        job_id,
        status="running",
        message="调查正在运行。此过程可能需要数分钟，请保持页面打开。",
        started_at=_now_iso(),
    )

    try:
        result = run_investigation(
            paper_identifier=paper_identifier,
            identifier_type=identifier_type,
            output_dir=output_dir,
            open_browser=False,
        )
    except Exception as exc:  # noqa: BLE001 - WebUI 需要统一呈现后台任务失败
        logger.exception("WebUI job failed: %s", job_id)
        _record_job_update(
            job_id,
            status="failed",
            message="调查失败。请检查输入、文件路径与 API 配置。",
            error=str(exc) or exc.__class__.__name__,
            finished_at=_now_iso(),
        )
        return

    markdown_text = str(result.get("markdown", ""))
    file_paths = {
        key: str(path)
        for key, path in result.get("files", {}).items()
        if key in REPORT_FILE_KEYS
    }
    _record_job_update(
        job_id,
        status="succeeded",
        message="调查完成。报告文件已生成。",
        elapsed=result.get("elapsed"),
        finished_at=_now_iso(),
        markdown_preview=markdown_text[:12000],
        file_paths=file_paths,
        title=_extract_title_from_markdown(markdown_text),
        risk_level=_extract_risk_level_from_markdown(markdown_text),
        risk_score=_extract_risk_score_from_markdown(markdown_text),
    )


def create_job(payload: dict[str, Any], *, start_worker: bool = True) -> tuple[int, dict[str, Any]]:
    """创建调查任务，返回 HTTP 状态码和响应体。"""
    job_id = uuid.uuid4().hex
    status, validated = _validate_job_payload(payload, job_id)
    if status != HTTPStatus.OK:
        return status, validated

    display_name = Path(validated["paper_identifier"]).name
    if validated["identifier_type"] not in {"local_pdf", "local_case"}:
        display_name = validated["paper_identifier"]

    job = {
        "id": job_id,
        "status": "queued",
        "message": "任务已创建，等待后台调查开始。",
        "created_at": _now_iso(),
        "started_at": None,
        "finished_at": None,
        "identifier_type": validated["identifier_type"],
        "paper_identifier": validated["paper_identifier"],
        "display_name": display_name,
        "output_dir": validated["output_dir"],
        "elapsed": None,
        "risk_level": None,
        "risk_score": None,
        "title": None,
        "markdown_preview": None,
        "file_paths": {},
        "manifest": validated.get("manifest"),
        "error": None,
    }
    with JOBS_LOCK:
        JOBS[job_id] = job

    if start_worker:
        worker = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
        worker.start()

    return HTTPStatus.ACCEPTED, {"job_id": job_id, "job": _public_job(job)}


def get_job(job_id: str) -> dict[str, Any] | None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return None
        return _public_job(job.copy())


def get_report_file_path(job_id: str, file_key: str) -> Path | None:
    """返回受控报告文件路径，不接受任意路径拼接。"""
    if file_key not in REPORT_FILE_KEYS:
        return None
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return None
        file_path = job.get("file_paths", {}).get(file_key)
    if not file_path:
        return None
    return Path(file_path)


class WebUIHandler(BaseHTTPRequestHandler):
    """本地 WebUI 请求处理器。"""

    server_version = "AcademicFraudDetectorWebUI/0.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        logger.info("%s - %s", self.address_string(), format % args)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(_json_safe(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, status: int, text: str) -> None:
        body = text.encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, *, content_type: str | None = None) -> None:
        if not path.exists() or not path.is_file():
            self._send_json(HTTPStatus.NOT_FOUND, _error("文件不存在。"))
            return
        body = path.read_bytes()
        guessed_type = content_type or mimetypes.guess_type(path.name)[0]
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", guessed_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, request_path: str) -> None:
        if request_path == "/":
            static_path = STATIC_DIR / "index.html"
        else:
            relative = unquote(request_path.removeprefix("/static/"))
            static_path = (STATIC_DIR / relative).resolve()
            try:
                static_path.relative_to(STATIC_DIR.resolve())
            except ValueError:
                self._send_json(HTTPStatus.FORBIDDEN, _error("静态资源路径不可访问。"))
                return

        content_type = "text/html; charset=utf-8" if static_path.name.endswith(".html") else None
        self._send_file(static_path, content_type=content_type)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        request_path = parsed.path

        if request_path == "/" or request_path.startswith("/static/"):
            self._serve_static(request_path)
            return

        parts = [part for part in request_path.strip("/").split("/") if part]
        if len(parts) == 3 and parts[:2] == ["api", "jobs"]:
            job = get_job(parts[2])
            if job is None:
                self._send_json(HTTPStatus.NOT_FOUND, _error("任务不存在。"))
                return
            self._send_json(HTTPStatus.OK, {"job": job})
            return

        if len(parts) == 5 and parts[:2] == ["api", "jobs"] and parts[3] == "files":
            self._serve_job_file(parts[2], parts[4])
            return

        self._send_json(HTTPStatus.NOT_FOUND, _error("接口不存在。"))

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/jobs":
            self._send_json(HTTPStatus.NOT_FOUND, _error("接口不存在。"))
            return

        content_type = self.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            self._send_json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, _error("仅支持 JSON 请求。"))
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(HTTPStatus.BAD_REQUEST, _error("请求长度无效。"))
            return
        if content_length <= 0 or content_length > 64_000:
            self._send_json(HTTPStatus.BAD_REQUEST, _error("请求体为空或过大。"))
            return

        try:
            raw_body = self.rfile.read(content_length).decode("utf-8")
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            self._send_json(HTTPStatus.BAD_REQUEST, _error("JSON 格式无效。"))
            return

        if not isinstance(payload, dict):
            self._send_json(HTTPStatus.BAD_REQUEST, _error("请求体必须是 JSON 对象。"))
            return

        status, response = create_job(payload)
        self._send_json(status, response)

    def _serve_job_file(self, job_id: str, file_key: str) -> None:
        if file_key not in REPORT_FILE_KEYS:
            self._send_json(HTTPStatus.NOT_FOUND, _error("报告文件类型不存在。"))
            return

        with JOBS_LOCK:
            job_exists = job_id in JOBS
        if not job_exists:
            self._send_json(HTTPStatus.NOT_FOUND, _error("任务不存在。"))
            return

        path = get_report_file_path(job_id, file_key)
        if path is None:
            self._send_json(HTTPStatus.NOT_FOUND, _error("该报告文件尚未生成。"))
            return
        content_type = None
        if file_key == "html":
            content_type = "text/html; charset=utf-8"
        elif file_key == "md":
            content_type = "text/markdown; charset=utf-8"
        elif file_key == "json":
            content_type = "application/json; charset=utf-8"
        self._send_file(path, content_type=content_type)


def run_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """启动本地 WebUI 服务。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    server = ThreadingHTTPServer((host, port), WebUIHandler)
    url = f"http://{host}:{port}"
    print("Academic Fraud Detector WebUI")
    print(f"Local: {url}")
    print("按 Ctrl+C 停止服务。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在停止 WebUI 服务。")
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="启动 Academic Fraud Detector 本地 WebUI。")
    parser.add_argument("--host", default=DEFAULT_HOST, help="监听地址，默认 127.0.0.1。")
    parser.add_argument("--port", default=DEFAULT_PORT, type=int, help="监听端口，默认 8765。")
    args = parser.parse_args(argv)
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
