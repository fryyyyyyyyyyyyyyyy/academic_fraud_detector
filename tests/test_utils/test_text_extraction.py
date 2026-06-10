"""Tests for PDF text/image extraction helpers."""

from academic_fraud_detector.utils.text_extraction import create_unique_image_output_dir


def test_create_unique_image_output_dir_uses_unique_child_directories(tmp_path):
    first = create_unique_image_output_dir(base_dir=tmp_path, source_name="paper.pdf")
    second = create_unique_image_output_dir(base_dir=tmp_path, source_name="paper.pdf")

    assert first != second
    assert first.parent == tmp_path
    assert second.parent == tmp_path
    assert first.exists()
    assert second.exists()
    assert first.name.startswith("paper_")
    assert second.name.startswith("paper_")
