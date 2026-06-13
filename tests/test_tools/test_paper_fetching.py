"""Tests for local PDF loading behavior."""

import json

from academic_fraud_detector.tools import paper_fetching
from academic_fraud_detector.tools.paper_fetching import LocalPaperLoaderTool
from academic_fraud_detector.utils.mineru_client import MinerUMarkdownResult, MinerUResultError


EXPECTED_LOCAL_PDF_KEYS = {
    "source",
    "file_path",
    "file_name",
    "file_size_bytes",
    "full_text_available",
    "full_text",
    "full_text_length_chars",
    "page_count",
    "images",
    "image_output_dir",
    "tables",
    "panels",
    "mineru",
    "pre_extracted_stats",
    "paper_claims",
    "error",
    "supplementary_files",
    "_summary",
}


def patch_panel_splitter(monkeypatch):
    monkeypatch.setattr(
        "academic_fraud_detector.utils.figure_splitter.extract_all_panels_from_pdf",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "academic_fraud_detector.utils.figure_splitter.extract_all_panels_from_images",
        lambda *args, **kwargs: [],
    )


def test_local_loader_uses_mineru_markdown_and_cached_images(monkeypatch, tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7 fake")
    markdown = "# 论文标题\n\n![Figure](C:/cache/image_0001.png)\n\nResults showed p = 0.031."
    mineru_image = {
        "filename": "image_0001.png",
        "filepath": "C:/cache/image_0001.png",
        "format": "PNG",
        "width": 100,
        "height": 80,
        "page_number": None,
        "xref": None,
        "size_bytes": 123,
        "source": "mineru",
        "original_path": "images/fig1.png",
        "markdown_path": "C:/cache/image_0001.png",
    }

    def fail_extract_pdf_images(*args, **kwargs):
        raise AssertionError("PyMuPDF image extraction should not run when MinerU succeeds")

    monkeypatch.setattr(paper_fetching, "extract_pdf_tables", lambda *args, **kwargs: [])
    monkeypatch.setattr(paper_fetching, "extract_pdf_images", fail_extract_pdf_images)
    monkeypatch.setattr(
        paper_fetching,
        "extract_pdf_markdown_with_mineru_assets",
        lambda *args, **kwargs: MinerUMarkdownResult(
            markdown=markdown,
            full_md_path="C:/cache/full.md",
            raw_full_md_path="C:/cache/full_raw.md",
            cache_dir="C:/cache",
            images=[mineru_image],
            zip_path="C:/cache/result.zip",
        ),
    )
    monkeypatch.setattr(
        LocalPaperLoaderTool,
        "_count_pdf_pages",
        staticmethod(lambda pdf_bytes, max_pages=None: 1),
    )
    patch_panel_splitter(monkeypatch)

    result = json.loads(
        LocalPaperLoaderTool()._run(
            str(pdf_path),
            max_pages=5,
            extract_images=True,
            extract_tables=True,
        )
    )

    assert EXPECTED_LOCAL_PDF_KEYS.issubset(result.keys())
    assert result["full_text_available"] is True
    assert result["full_text"] == markdown
    assert result["full_text_length_chars"] == len(markdown)
    assert result["page_count"] == 1
    assert 0.031 in result["pre_extracted_stats"]["p_values"]
    assert result["pre_extracted_stats"]["p_value_records"][0]["source"]["page"] is None
    assert result["pre_extracted_stats"]["p_value_records"][0]["source"]["page_detection"] == "missing_page_marker"
    assert result["paper_claims"]["summary"]["reported_p_value_count"] == 1
    assert result["mineru"] == {
        "used": True,
        "cache_dir": "C:/cache",
        "full_md_path": "C:/cache/full.md",
        "raw_full_md_path": "C:/cache/full_raw.md",
        "zip_path": "C:/cache/result.zip",
        "image_count": 1,
    }
    assert result["image_output_dir"] == "C:/cache"
    assert len(result["images"]) == 1
    assert result["images"][0]["source"] == "mineru"
    assert {image.get("source") for image in result["images"]} == {"mineru"}
    assert result["error"] is None


def test_local_loader_falls_back_to_pymupdf_when_mineru_fails(monkeypatch, tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7 fake")
    fallback_text = "[Page 1]\nFallback text reported p = 0.04 and p = 1e-8."
    pymupdf_image = {
        "filename": "page1_img1.png",
        "filepath": "C:/cache/page1_img1.png",
        "format": "PNG",
        "width": 200,
        "height": 160,
        "page_number": 1,
        "xref": 7,
        "size_bytes": 456,
    }

    def fail_mineru(*args, **kwargs):
        raise MinerUResultError("MinerU failed")

    monkeypatch.setattr(paper_fetching, "extract_pdf_markdown_with_mineru_assets", fail_mineru)
    monkeypatch.setattr(
        paper_fetching,
        "extract_pdf_text",
        lambda *args, **kwargs: fallback_text,
    )
    image_calls = []

    def fake_extract_pdf_images(*args, **kwargs):
        image_calls.append((args, kwargs))
        return [pymupdf_image]

    monkeypatch.setattr(paper_fetching, "extract_pdf_images", fake_extract_pdf_images)
    monkeypatch.setattr(paper_fetching, "extract_pdf_tables", lambda *args, **kwargs: [])
    patch_panel_splitter(monkeypatch)

    result = json.loads(
        LocalPaperLoaderTool()._run(
            str(pdf_path),
            max_pages=5,
            extract_images=True,
            extract_tables=True,
        )
    )

    assert EXPECTED_LOCAL_PDF_KEYS.issubset(result.keys())
    assert result["full_text_available"] is True
    assert result["full_text"] == fallback_text
    assert result["page_count"] == 1
    assert 0.04 in result["pre_extracted_stats"]["p_values"]
    assert 1e-08 in result["pre_extracted_stats"]["p_values"]
    assert 0.0 not in result["pre_extracted_stats"]["p_values"]
    assert result["pre_extracted_stats"]["p_value_records"][0]["source"]["page"] == 1
    assert result["pre_extracted_stats"]["p_value_records"][0]["source"]["page_detection"] == "page_marker"
    assert result["paper_claims"]["summary"]["reported_p_value_count"] == 2
    assert result["mineru"] == {
        "used": False,
        "fallback": "pymupdf",
        "reason": "MinerUResultError",
    }
    assert len(image_calls) == 1
    assert image_calls[0][1]["output_dir"]
    assert result["image_output_dir"] == image_calls[0][1]["output_dir"]
    assert len(result["images"]) == 1
    assert result["images"][0]["source"] == "pymupdf"
    assert result["error"] is None


def test_load_single_pdf_uses_only_mineru_images_when_mineru_succeeds(monkeypatch, tmp_path):
    pdf_path = tmp_path / "supplement.pdf"
    pdf_path.write_bytes(b"%PDF-1.7 fake")
    markdown = "[Page 1]\nSupplementary markdown."
    mineru_image = {
        "filename": "supp_image.png",
        "filepath": "C:/cache/supp_image.png",
        "format": "PNG",
        "width": 120,
        "height": 90,
        "source": "mineru",
    }

    def fail_extract_pdf_images(*args, **kwargs):
        raise AssertionError("PyMuPDF image extraction should not run when MinerU succeeds")

    monkeypatch.setattr(paper_fetching, "extract_pdf_images", fail_extract_pdf_images)
    monkeypatch.setattr(
        paper_fetching,
        "extract_pdf_markdown_with_mineru_assets",
        lambda *args, **kwargs: MinerUMarkdownResult(
            markdown=markdown,
            full_md_path="C:/cache/full.md",
            raw_full_md_path="C:/cache/full_raw.md",
            cache_dir="C:/cache",
            images=[mineru_image],
            zip_path="C:/cache/result.zip",
        ),
    )
    patch_panel_splitter(monkeypatch)

    result = LocalPaperLoaderTool()._load_single_pdf(
        str(pdf_path),
        max_pages=5,
        extract_images=True,
        extract_tables=False,
        image_min_size=100,
    )

    assert result["mineru"]["used"] is True
    assert result["image_output_dir"] == "C:/cache"
    assert result["images"] == [mineru_image]
    assert {image.get("source") for image in result["images"]} == {"mineru"}


def test_local_loader_merges_supplementary_claims_without_mutating_file_claims(
    monkeypatch, tmp_path
):
    main_pdf = tmp_path / "main.pdf"
    supp_pdf = tmp_path / "supplement.pdf"
    main_pdf.write_bytes(b"%PDF-1.7 fake main")
    supp_pdf.write_bytes(b"%PDF-1.7 fake supp")

    def fake_extract_mineru_first(self, pdf_bytes, file_name, max_pages):
        if file_name == "main.pdf":
            text = "[Page 1]\nResults\nMain value was 1.0 ± 0.1, p = 0.05."
        else:
            text = "[Page 1]\nResults\nSupplement value was 2.0 ± 0.2, n = 5."
        return text, [], {"used": False, "fallback": "pymupdf", "reason": "test"}

    monkeypatch.setattr(LocalPaperLoaderTool, "_extract_mineru_first", fake_extract_mineru_first)
    monkeypatch.setattr(paper_fetching, "extract_pdf_tables", lambda *args, **kwargs: [])
    patch_panel_splitter(monkeypatch)

    result = json.loads(
        LocalPaperLoaderTool()._run(
            str(main_pdf),
            max_pages=5,
            extract_images=False,
            extract_tables=True,
            supplementary_paths=json.dumps([str(supp_pdf)]),
        )
    )

    summary = result["paper_claims"]["summary"]
    assert summary["claim_count"] == 4
    assert summary["reported_mean_sd_count"] == 2
    assert summary["reported_p_value_count"] == 1
    assert summary["reported_n_count"] == 1
    assert result["paper_claims"]["claims"][-1]["claim_id"] == "PCL-0004"
    supp_claims = result["supplementary_files"][0]["paper_claims"]["claims"]
    assert supp_claims[0]["claim_id"] == "PCL-0001"
