"""Tests for deterministic local PDF preload before Crew kickoff."""

import json

from academic_fraud_detector.crew import AcademicFraudDetectionCrew
from academic_fraud_detector.utils import cross_figure_pipeline


class FakeLoader:
    def __init__(self, output=None, error=None):
        self.output = output
        self.error = error
        self.calls = []

    def _run(self, file_path):
        self.calls.append(file_path)
        if self.error:
            raise self.error
        return self.output


class FakeForensicsTool:
    def __init__(self, name, output=None):
        self.name = name
        self.output = output or {"flagged": False, "match_count": 0}
        self.calls = []

    def _run(self, **kwargs):
        self.calls.append(kwargs)
        return json.dumps(self.output, ensure_ascii=False)


def make_crew_with_loader(loader):
    crew = object.__new__(AcademicFraudDetectionCrew)
    crew._local_paper_loader = loader
    crew._ela = FakeForensicsTool("error_level_analysis")
    crew._clone_detection = FakeForensicsTool("clone_detection")
    crew._ai_image = FakeForensicsTool("ai_image_detection")
    crew._background_consistency = FakeForensicsTool("background_consistency_check")
    crew._cross_image_duplicate = FakeForensicsTool("cross_image_duplicate_check")
    crew._feature_duplicate = FakeForensicsTool("feature_based_duplicate_check")
    return crew


def patch_precheck(monkeypatch):
    precheck = {
        "datasets": [],
        "matches": [],
        "has_critical_match": False,
        "statistical_precheck": {"total_flagged_checks": 0},
    }
    monkeypatch.setattr(
        cross_figure_pipeline,
        "run_cross_figure_pipeline",
        lambda pdf_path, images_dir=None, images=None: precheck,
    )
    monkeypatch.setattr(
        cross_figure_pipeline,
        "format_precheck_for_agent",
        lambda result: "formatted precheck",
    )
    return precheck


def test_validate_inputs_preloads_local_pdf_payload(monkeypatch, tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7 fake")
    payload = {
        "source": "local_pdf",
        "file_path": str(pdf_path),
        "full_text_available": True,
        "full_text": "# Title\n\nMethods text p = 0.031.",
        "full_text_length_chars": 31,
        "page_count": 1,
        "images": [{"filepath": "C:/cache/image.png", "source": "mineru"}],
        "panels": ["C:/cache/panel.png"],
        "tables": [{"data": [["A", "B"]]}],
        "pre_extracted_stats": {"p_values": [0.031]},
        "mineru": {"used": True, "image_count": 1},
        "error": None,
        "_summary": "PDF loaded: paper.pdf",
    }
    loader = FakeLoader(output=json.dumps(payload, ensure_ascii=False))
    precheck = patch_precheck(monkeypatch)
    crew = make_crew_with_loader(loader)

    result = crew.validate_inputs({
        "paper_identifier": str(pdf_path),
        "identifier_type": "local_pdf",
    })

    assert loader.calls == [str(pdf_path)]
    assert result["local_paper_payload"] == payload
    assert json.loads(result["local_paper_payload_json"])["full_text_available"] is True
    assert result["local_paper_load_status"] == "success"
    assert result["local_paper_load_error"] == ""
    assert result["local_paper_summary"] == "PDF loaded: paper.pdf"
    assert json.loads(result["local_paper_images_json"]) == payload["images"]
    assert json.loads(result["local_paper_panels_json"]) == payload["panels"]
    assert json.loads(result["local_paper_stats_json"])["pre_extracted_stats"] == {
        "p_values": [0.031]
    }
    assert result["local_paper_text"] == payload["full_text"]
    assert json.loads(result["image_forensics_precheck_json"])["status"] == "no_input"
    assert result["cross_figure_precheck"] == "formatted precheck"
    assert json.loads(result["cross_figure_precheck_json"]) == precheck


def test_validate_inputs_runs_deterministic_image_forensics_precheck(monkeypatch, tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7 fake")
    image_a = tmp_path / "image_a.png"
    image_b = tmp_path / "image_b.png"
    panel_a = tmp_path / "panel_a.png"
    panel_b = tmp_path / "panel_b.png"
    for image_path in [image_a, image_b, panel_a, panel_b]:
        image_path.write_bytes(b"fake image bytes")

    payload = {
        "source": "local_pdf",
        "file_path": str(pdf_path),
        "full_text_available": True,
        "full_text": "正文",
        "images": [
            {"filepath": str(image_a), "source": "mineru"},
            {"filepath": str(image_b), "source": "mineru"},
        ],
        "panels": [str(panel_a), str(panel_b)],
        "tables": [],
        "pre_extracted_stats": {},
        "mineru": {"used": True},
        "error": None,
    }
    loader = FakeLoader(output=json.dumps(payload, ensure_ascii=False))
    patch_precheck(monkeypatch)
    crew = make_crew_with_loader(loader)

    result = crew.validate_inputs({
        "paper_identifier": str(pdf_path),
        "identifier_type": "local_pdf",
    })

    for tool in [crew._ela, crew._clone_detection, crew._ai_image, crew._background_consistency]:
        assert len(tool.calls) == 2
        assert tool.calls[0]["image_path_or_url"] == str(image_a)
        assert tool.calls[1]["image_path_or_url"] == str(image_b)

    assert len(crew._cross_image_duplicate.calls) == 1
    assert json.loads(crew._cross_image_duplicate.calls[0]["image_paths"]) == [
        str(image_a),
        str(image_b),
    ]
    assert len(crew._feature_duplicate.calls) == 1
    assert json.loads(crew._feature_duplicate.calls[0]["image_paths"]) == [
        str(panel_a),
        str(panel_b),
    ]
    assert crew._feature_duplicate.calls[0]["min_inliers"] == 8
    assert crew._feature_duplicate.calls[0]["ratio_threshold"] == 0.80
    assert crew._feature_duplicate.calls[0]["sift_contrast_threshold"] == 0.02

    precheck = json.loads(result["image_forensics_precheck_json"])
    assert precheck["status"] == "success"
    assert precheck["coverage"]["single_image_tools_analyzed"] == 2
    assert precheck["coverage"]["cross_image_paths_analyzed"] == 2
    assert precheck["coverage"]["feature_panel_paths_analyzed"] == 2
    assert set(precheck["tools_attempted"]) == {
        "error_level_analysis",
        "clone_detection",
        "ai_image_detection",
        "background_consistency_check",
        "cross_image_duplicate_check",
        "feature_based_duplicate_check",
    }
    assert "图像取证工具预检" in result["image_forensics_precheck"]


def test_validate_inputs_injects_error_payload_when_loader_fails(monkeypatch, tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7 fake")
    loader = FakeLoader(error=RuntimeError("boom"))
    patch_precheck(monkeypatch)
    crew = make_crew_with_loader(loader)

    result = crew.validate_inputs({
        "paper_identifier": str(pdf_path),
        "identifier_type": "local_pdf",
    })

    assert result["local_paper_load_status"] == "error"
    assert "boom" in result["local_paper_load_error"]
    payload = json.loads(result["local_paper_payload_json"])
    assert payload["error"] == "boom"
    assert payload["images"] == []
    assert result["local_paper_images_json"] == "[]"
    assert json.loads(result["image_forensics_precheck_json"])["status"] == "no_input"
    assert result["cross_figure_precheck"] == "formatted precheck"


def test_validate_inputs_non_local_pdf_keeps_default_payload_fields():
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
    assert result["local_paper_images_json"] == "[]"
    assert result["local_paper_panels_json"] == "[]"
    assert result["local_paper_stats_json"] == "{}"
    assert result["local_paper_text"] == ""
    assert result["image_forensics_precheck"] == "未在当前模式下执行图像取证工具预检。"
    assert result["image_forensics_precheck_json"] == "{}"
    assert result["cross_figure_precheck_json"] == "{}"
