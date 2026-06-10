"""Tests for figure panel splitting helpers."""

from PIL import Image

from academic_fraud_detector.utils import figure_splitter


def test_extract_all_panels_from_images_uses_existing_image_metadata(monkeypatch, tmp_path):
    image_path = tmp_path / "mineru_image.png"
    Image.new("RGB", (20, 20), color="white").save(image_path)

    def fake_split_composite_figure(image, **kwargs):
        return [
            {
                "panel_index": 0,
                "grid_position": (0, 0),
                "bbox": (0, 0, 10, 10),
                "width": 10,
                "height": 10,
                "panel_image": Image.new("RGB", (10, 10), color="white"),
            }
        ]

    monkeypatch.setattr(figure_splitter, "split_composite_figure", fake_split_composite_figure)

    result = figure_splitter.extract_all_panels_from_images(
        [
            {
                "filename": "mineru_image.png",
                "filepath": str(image_path),
                "source": "mineru",
                "page_number": 2,
            }
        ],
        output_dir=str(tmp_path / "panels_run"),
    )

    assert len(result) == 1
    assert result[0]["source"] == "mineru"
    assert result[0]["panel_count"] == 1
    assert result[0]["panels"][0]["source"] == "mineru"
    assert result[0]["panels"][0]["pdf_page"] == 2
    assert (tmp_path / "panels_run" / "panels" / "mineru_image_panel_0.png").exists()
