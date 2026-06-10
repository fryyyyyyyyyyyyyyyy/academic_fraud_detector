"""Regression tests for deterministic cross-figure precheck pipeline."""

from PIL import Image

from academic_fraud_detector.utils import cross_figure_pipeline as pipeline


MINIMAL_TEXT_VALUES = {
    "total_values": 0,
    "all_numbers": [],
    "means": [],
    "sds": [],
    "p_values": [],
    "mean_sd_pairs": [],
    "ordered_sequences_from_text": [],
}


def patch_pipeline_common(monkeypatch, *, panels, bar_candidates, datasets):
    captured = {}
    monkeypatch.setattr(pipeline, "_build_page_figure_map", lambda pdf_path: {1: "Figure 1"})
    monkeypatch.setattr(
        pipeline,
        "_get_all_panel_images",
        lambda pdf_path, images_dir, page_fig_map, images=None: panels,
    )
    monkeypatch.setattr(
        pipeline,
        "_identify_bar_chart_candidates",
        lambda panel_list: bar_candidates,
    )
    monkeypatch.setattr(
        pipeline,
        "_check_risk_pairs",
        lambda panel_list, candidates, page_fig_map: [],
    )
    monkeypatch.setattr(
        pipeline,
        "_extract_all_bar_chart_datasets",
        lambda candidates: datasets,
    )
    monkeypatch.setattr(
        pipeline,
        "_extract_numeric_values_from_pdf_text",
        lambda pdf_path: MINIMAL_TEXT_VALUES,
    )
    monkeypatch.setattr(
        pipeline,
        "_run_statistical_prechecks",
        lambda datasets, text_values: {"total_flagged_checks": 0, "checks_run": []},
    )

    def fake_guidance(**kwargs):
        captured.update(kwargs)
        return "guidance"

    monkeypatch.setattr(pipeline, "_generate_enhanced_guidance", fake_guidance)
    return captured


def test_pipeline_handles_panels_without_datasets(monkeypatch):
    captured = patch_pipeline_common(
        monkeypatch,
        panels=[{"filepath": "panel1.png", "figure_label": "Fig 1"}],
        bar_candidates=[],
        datasets=[],
    )

    result = pipeline.run_cross_figure_pipeline("paper.pdf")

    assert result["datasets"] == []
    assert result["matches"] == []
    assert result["has_critical_match"] is False
    assert result["guidance"] == "guidance"
    assert captured["matches"] == []


def test_pipeline_handles_single_dataset_without_comparison(monkeypatch):
    dataset = {
        "label": "Fig 1A",
        "values": [1.0, 2.0, 3.0],
        "ordered_series": [1.0, 2.0, 3.0],
        "group_labels": ["A", "B", "C"],
        "bar_count": 3,
        "confidence": "high",
    }
    captured = patch_pipeline_common(
        monkeypatch,
        panels=[{"filepath": "panel1.png", "figure_label": "Fig 1"}],
        bar_candidates=[{"filepath": "panel1.png", "figure_label": "Fig 1"}],
        datasets=[dataset],
    )

    def fail_if_called(datasets):
        raise AssertionError("Cross-figure comparison should not run for one dataset")

    monkeypatch.setattr(pipeline, "_run_cross_figure_comparison", fail_if_called)

    result = pipeline.run_cross_figure_pipeline("paper.pdf")

    assert result["datasets"] == [dataset]
    assert result["matches"] == []
    assert result["has_critical_match"] is False
    assert captured["matches"] == []


def test_pipeline_handles_no_panels(monkeypatch):
    patch_pipeline_common(
        monkeypatch,
        panels=[],
        bar_candidates=[],
        datasets=[],
    )

    result = pipeline.run_cross_figure_pipeline("paper.pdf")

    assert result["datasets"] == []
    assert result["matches"] == []
    assert "No panel images found." in result["errors"]


def test_pipeline_records_critical_match_for_two_datasets(monkeypatch):
    datasets = [
        {"label": "Fig 1A", "values": [1.0, 2.0], "ordered_series": [1.0, 2.0]},
        {"label": "Fig 2A", "values": [1.0, 2.0], "ordered_series": [1.0, 2.0]},
    ]
    patch_pipeline_common(
        monkeypatch,
        panels=[{"filepath": "panel1.png", "figure_label": "Fig 1"}],
        bar_candidates=[{"filepath": "panel1.png", "figure_label": "Fig 1"}],
        datasets=datasets,
    )
    match = {
        "dataset_a": "Fig 1A",
        "dataset_b": "Fig 2A",
        "confidence": "critical",
        "type": "exact_match",
    }
    monkeypatch.setattr(pipeline, "_run_cross_figure_comparison", lambda datasets: [match])

    result = pipeline.run_cross_figure_pipeline("paper.pdf")

    assert result["matches"] == [match]
    assert result["has_critical_match"] is True


def test_get_all_panel_images_uses_preloaded_images_without_pdf_extraction(monkeypatch, tmp_path):
    panel_path = tmp_path / "mineru_panel.png"
    Image.new("RGB", (12, 10), color="white").save(panel_path)
    captured = {}
    preloaded_images = [{"filepath": "C:/cache/image.png", "filename": "image.png", "source": "mineru"}]

    def fake_from_images(images, output_dir=None, **kwargs):
        captured["images"] = images
        captured["output_dir"] = output_dir
        return [
            {
                "filename": "image.png",
                "page_number": 1,
                "source": "mineru",
                "panels": [
                    {
                        "filepath": str(panel_path),
                        "filename": panel_path.name,
                        "panel_index": 0,
                        "pdf_page": 1,
                        "width": 12,
                        "height": 10,
                        "source": "mineru",
                    }
                ],
            }
        ]

    def fail_pdf_extraction(*args, **kwargs):
        raise AssertionError("PDF image extraction should not run when preloaded images are provided")

    monkeypatch.setattr(
        "academic_fraud_detector.utils.figure_splitter.extract_all_panels_from_images",
        fake_from_images,
    )
    monkeypatch.setattr(
        "academic_fraud_detector.utils.figure_splitter.extract_all_panels_from_pdf",
        fail_pdf_extraction,
    )

    panels = pipeline._get_all_panel_images(
        "paper.pdf",
        str(tmp_path),
        {1: "Figure 1"},
        images=preloaded_images,
    )

    assert captured["images"] == preloaded_images
    assert captured["output_dir"] == str(tmp_path)
    assert panels[0]["source"] == "mineru"
    assert panels[0]["full_label"] == "Figure 1A"


def test_get_all_panel_images_reuses_existing_preloaded_panels(monkeypatch, tmp_path):
    panels_dir = tmp_path / "panels"
    panels_dir.mkdir()
    panel_path = panels_dir / "image_0001_panel_0.png"
    duplicate_panel_path = panels_dir / "image_0001_panel_0_1.png"
    Image.new("RGB", (16, 12), color="white").save(panel_path)
    Image.new("RGB", (16, 12), color="white").save(duplicate_panel_path)
    preloaded_images = [
        {
            "filepath": str(tmp_path / "image_0001.png"),
            "filename": "image_0001.png",
            "page_number": 2,
            "source": "mineru",
        }
    ]

    def fail_from_images(*args, **kwargs):
        raise AssertionError("Existing panels should be reused instead of re-splitting images")

    monkeypatch.setattr(
        "academic_fraud_detector.utils.figure_splitter.extract_all_panels_from_images",
        fail_from_images,
    )

    panels = pipeline._get_all_panel_images(
        "paper.pdf",
        str(tmp_path),
        {2: "Figure 2"},
        images=preloaded_images,
    )

    assert len(panels) == 1
    assert panels[0]["filename"] == "image_0001_panel_0.png"
    assert panels[0]["source"] == "mineru"
    assert panels[0]["full_label"] == "Figure 2A"


def test_get_all_panel_images_creates_unique_dir_when_no_inputs(monkeypatch, tmp_path):
    unique_dir = tmp_path / "unique_cross_figure"
    captured = {}

    def fake_unique_dir(*args, **kwargs):
        unique_dir.mkdir(parents=True, exist_ok=True)
        return unique_dir

    def fake_from_pdf(pdf_path, output_dir=None, **kwargs):
        captured["pdf_path"] = pdf_path
        captured["output_dir"] = output_dir
        return []

    monkeypatch.setattr(
        "academic_fraud_detector.utils.text_extraction.create_unique_image_output_dir",
        fake_unique_dir,
    )
    monkeypatch.setattr(
        "academic_fraud_detector.utils.figure_splitter.extract_all_panels_from_pdf",
        fake_from_pdf,
    )

    panels = pipeline._get_all_panel_images("paper.pdf", None, {1: "Figure 1"})

    assert panels == []
    assert captured == {"pdf_path": "paper.pdf", "output_dir": str(unique_dir)}
