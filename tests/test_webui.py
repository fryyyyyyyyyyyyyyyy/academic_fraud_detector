"""本地 WebUI 适配层测试。"""

from http import HTTPStatus
import webbrowser

import pytest

from academic_fraud_detector import main as main_module
from academic_fraud_detector.main import save_report_files
from academic_fraud_detector import web


@pytest.fixture(autouse=True)
def clear_webui_jobs():
    with web.JOBS_LOCK:
        web.JOBS.clear()
    yield
    with web.JOBS_LOCK:
        web.JOBS.clear()


def test_save_report_files_can_skip_browser_open(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(webbrowser, "open", lambda url: calls.append(url))

    files = save_report_files("# 调查报告\n\n正文", tmp_path, "report", open_browser=False)

    assert calls == []
    assert files["md"].exists()
    assert files["html"].exists()


def test_run_investigation_passes_open_browser_to_report_writer(monkeypatch, tmp_path):
    class FakeCrew:
        def kickoff(self, inputs):
            assert inputs == {"paper_identifier": "10.1234/example", "identifier_type": "doi"}
            return "# 报告\n\n| **风险等级** | 中 |\n| **风险评分** | 42 |"

    class FakeCrewBuilder:
        def __init__(self, local_only=False):
            assert local_only is False

        def crew(self):
            return FakeCrew()

    captured = {}

    def fake_save_report_files(markdown_text, output_dir, base_name, open_browser=True):
        captured["open_browser"] = open_browser
        return {
            "md": output_dir / f"{base_name}.md",
            "html": output_dir / f"{base_name}.html",
        }

    monkeypatch.setattr(main_module, "AcademicFraudDetectionCrew", FakeCrewBuilder)
    monkeypatch.setattr(main_module, "save_report_files", fake_save_report_files)

    result = main_module.run_investigation(
        "10.1234/example",
        "doi",
        output_dir=str(tmp_path),
        open_browser=False,
    )

    assert captured["open_browser"] is False
    assert result["files"]["json"].exists()
    assert result["elapsed"] >= 0


def test_create_job_rejects_empty_identifier():
    status, response = web.create_job(
        {"identifier_type": "doi", "paper_identifier": ""},
        start_worker=False,
    )

    assert status == HTTPStatus.BAD_REQUEST
    assert "不能为空" in response["error"]


def test_create_job_rejects_invalid_local_pdf(tmp_path):
    txt_path = tmp_path / "paper.txt"
    txt_path.write_text("not a pdf", encoding="utf-8")

    status, response = web.create_job(
        {"identifier_type": "local_pdf", "paper_identifier": str(txt_path)},
        start_worker=False,
    )

    assert status == HTTPStatus.BAD_REQUEST
    assert "PDF" in response["error"]


def test_create_job_rejects_invalid_local_case(tmp_path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "paper.pdf").write_bytes(b"%PDF-1.7")

    status, response = web.create_job(
        {"identifier_type": "local_case", "paper_identifier": str(case_dir)},
        start_worker=False,
    )

    assert status == HTTPStatus.BAD_REQUEST
    assert "案例目录不可用" in response["error"]
    assert response["details"]["errors"]


def test_create_job_without_worker_records_public_job(tmp_path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "paper.pdf").write_bytes(b"%PDF-1.7")
    (case_dir / "data.xlsx").write_bytes(b"placeholder")

    status, response = web.create_job(
        {"identifier_type": "local_case", "paper_identifier": str(case_dir)},
        start_worker=False,
    )

    assert status == HTTPStatus.ACCEPTED
    job = response["job"]
    assert job["status"] == "queued"
    assert job["manifest"]["selected_pdf"].endswith("paper.pdf")
    assert job["output_dir"].endswith(job["id"])


def test_report_file_paths_are_exposed_only_by_allowed_keys(tmp_path):
    md_path = tmp_path / "report.md"
    md_path.write_text("# 报告", encoding="utf-8")
    with web.JOBS_LOCK:
        web.JOBS["job1"] = {
            "id": "job1",
            "status": "succeeded",
            "message": "done",
            "created_at": "2026-06-13T00:00:00",
            "identifier_type": "doi",
            "display_name": "10.1234/example",
            "output_dir": str(tmp_path),
            "file_paths": {"md": str(md_path)},
        }

    public_job = web.get_job("job1")

    assert public_job["files"] == {"md": "/api/jobs/job1/files/md"}
    assert web.get_report_file_path("job1", "md") == md_path
    assert web.get_report_file_path("job1", "../secret") is None
    assert web.get_report_file_path("missing", "md") is None
