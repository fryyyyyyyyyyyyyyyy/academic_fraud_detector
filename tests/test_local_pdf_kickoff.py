"""本地 PDF / 本地案例 kickoff 前的确定性预加载测试。"""

import json

from academic_fraud_detector.crew import AcademicFraudDetectionCrew


class FakeLoader:
    def __init__(self, output=None, error=None):
        self.output = output
        self.error = error
        self.calls = []

    def _run(self, file_path, **kwargs):
        self.calls.append({"file_path": file_path, **kwargs})
        if self.error:
            raise self.error
        return self.output


def make_crew_with_loader(loader):
    crew = object.__new__(AcademicFraudDetectionCrew)
    crew._local_paper_loader = loader
    return crew


def test_validate_inputs_preloads_local_pdf_without_images(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7 fake")
    payload = {
        "source": "local_pdf",
        "file_path": str(pdf_path),
        "full_text_available": True,
        "full_text": "# Title\n\nMethods text p = 0.031.",
        "full_text_length_chars": 31,
        "page_count": 1,
        "images": [],
        "panels": [],
        "tables": [{"data": [["A", "B"]]}],
        "pre_extracted_stats": {"p_values": [0.031]},
        "paper_claims": {"schema_version": "paper_claims.v1", "claims": [], "summary": {}},
        "mineru": {"used": True, "image_count": 0},
        "error": None,
        "_summary": "PDF loaded without images",
    }
    loader = FakeLoader(output=json.dumps(payload, ensure_ascii=False))
    crew = make_crew_with_loader(loader)

    result = crew.validate_inputs({
        "paper_identifier": str(pdf_path),
        "identifier_type": "local_pdf",
    })

    assert loader.calls == [
        {"file_path": str(pdf_path), "extract_images": False, "extract_tables": True}
    ]
    assert result["local_paper_payload"] == payload
    assert result["local_paper_load_status"] == "success"
    assert result["local_paper_images_json"] == "[]"
    assert result["local_paper_panels_json"] == "[]"
    assert json.loads(result["paper_claims_json"])["schema_version"] == "paper_claims.v1"
    assert result["image_forensics_precheck_json"] == "{}"
    assert result["cross_figure_precheck_json"] == "{}"
    assert "不执行图像" in result["image_forensics_precheck"]


def test_validate_inputs_local_case_injects_raw_data_precheck(monkeypatch, tmp_path):
    case_dir = tmp_path / "案例 A"
    case_dir.mkdir()
    pdf_path = case_dir / "paper.pdf"
    xlsx_path = case_dir / "Source Data Fig.1.xlsx"
    pdf_path.write_bytes(b"%PDF-1.7 fake")
    xlsx_path.write_bytes(b"fake xlsx placeholder")

    payload = {
        "source": "local_pdf",
        "file_path": str(pdf_path),
        "full_text_available": True,
        "full_text": "Paper text 12.3 ± 1.1",
        "full_text_length_chars": 22,
        "page_count": 1,
        "images": [],
        "panels": [],
        "tables": [],
        "pre_extracted_stats": {"means_and_sds": [{"mean": 12.3, "sd": 1.1}]},
        "paper_claims": {
            "schema_version": "paper_claims.v1",
            "claims": [{"claim_id": "PCL-0001", "claim_type": "reported_mean_sd"}],
            "summary": {"claim_count": 1},
        },
        "mineru": {"used": False},
        "error": None,
        "_summary": "PDF loaded without images",
    }
    loader = FakeLoader(output=json.dumps(payload, ensure_ascii=False))
    crew = make_crew_with_loader(loader)

    def fail_image_precheck(_inputs):
        raise AssertionError("image precheck should not run for local_case")

    monkeypatch.setattr(crew, "_inject_image_forensics_precheck", fail_image_precheck)
    monkeypatch.setattr(
        "academic_fraud_detector.crew.load_raw_data_files",
        lambda paths: {
            "files": [{"path": paths[0], "status": "success"}],
            "datasets": [{"dataset_id": "D1", "values": [1.1, 2.2, 3.3]}],
            "profile": {"file_count": 1, "dataset_count": 1, "numeric_value_count": 3},
        },
    )
    monkeypatch.setattr(
        "academic_fraud_detector.crew.run_raw_data_precheck",
        lambda raw_payload, paper_payload: {
            "status": "success",
            "deterministic_findings": [{"evidence_id": "E-0001"}],
            "confidence_summary": {"overall_risk": "high", "evidence_count": 1},
            "allowed_claims": [{"evidence_id": "E-0001", "claim": "test"}],
            "evidence_cross_validation": {
                "schema_version": "evidence_cross_validation.v1",
                "summary": {"validation_count": 1},
                "validations": [],
            },
        },
    )
    monkeypatch.setattr(
        "academic_fraud_detector.crew.format_raw_data_precheck_for_agent",
        lambda precheck: "raw precheck formatted",
    )

    result = crew.validate_inputs({
        "paper_identifier": str(case_dir),
        "identifier_type": "local_case",
    })

    assert loader.calls == [
        {"file_path": str(pdf_path), "extract_images": False, "extract_tables": True}
    ]
    assert json.loads(result["case_manifest_json"])["selected_pdf"] == str(pdf_path)
    assert json.loads(result["raw_data_files_json"])[0]["path"] == str(xlsx_path)
    assert json.loads(result["raw_data_profile_json"])["dataset_count"] == 1
    assert result["raw_data_precheck"] == "raw precheck formatted"
    assert json.loads(result["deterministic_evidence_json"])[0]["evidence_id"] == "E-0001"
    assert json.loads(result["confidence_summary_json"])["overall_risk"] == "high"
    assert json.loads(result["paper_claims_json"])["summary"]["claim_count"] == 1
    assert json.loads(result["evidence_cross_validation_json"])["summary"]["validation_count"] == 1
    assert result["image_forensics_precheck_json"] == "{}"
    assert result["cross_figure_precheck_json"] == "{}"


def test_validate_inputs_non_local_keeps_default_payload_fields():
    loader = FakeLoader(error=AssertionError("loader should not run"))
    crew = make_crew_with_loader(loader)

    result = crew.validate_inputs({
        "paper_identifier": "10.1234/example",
        "identifier_type": "doi",
    })

    assert loader.calls == []
    assert result["local_paper_payload"] == {}
    assert result["local_paper_payload_json"] == "{}"
    assert result["local_paper_load_status"] == "not_applicable"
    assert result["raw_data_precheck_json"] == "{}"
    assert result["deterministic_evidence_json"] == "[]"
    assert result["paper_claims_json"] == "{}"
    assert result["evidence_cross_validation_json"] == "{}"
