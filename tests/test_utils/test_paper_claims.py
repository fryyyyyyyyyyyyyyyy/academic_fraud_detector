"""论文 claim anchor 与 source span 抽取测试。"""

from academic_fraud_detector.utils.table_extraction import (
    extract_means_and_sds_with_sources,
    extract_p_values,
    extract_p_values_with_sources,
    extract_paper_claims,
)


def test_extract_p_values_keeps_float_api_and_handles_decimal_notation():
    text = (
        "Results: p=.05, p = 0.031, P < 0.001, "
        "p = 1e-5, p = 2.3e-4. Control text p = 2 should be ignored."
    )

    assert extract_p_values(text) == [0.05, 0.031, 0.001, 1e-05, 0.00023]

    records = extract_p_values_with_sources(text, file_name="paper.pdf")
    assert [record["value"] for record in records] == [0.05, 0.031, 0.001, 1e-05, 0.00023]
    assert records[0]["operator"] == "="
    assert records[0]["source"]["file_name"] == "paper.pdf"


def test_source_span_tracks_page_section_and_raw_text():
    text = "[Page 1]\nMethods\nn = 6 per group.\n[Page 2]\nResults\nIL-6 was 12.3 ± 1.1 and p = 0.031."

    mean_records = extract_means_and_sds_with_sources(text, file_name="paper.pdf")
    assert len(mean_records) == 1
    mean_record = mean_records[0]
    source = mean_record["source"]

    assert mean_record["mean"] == 12.3
    assert mean_record["sd"] == 1.1
    assert source["page"] == 2
    assert source["section"] == "Results"
    assert text[source["char_start"]:source["char_end"]] == "12.3 ± 1.1"
    assert source["page_detection"] == "page_marker"

    claims = extract_paper_claims(text, file_name="paper.pdf", extraction_method="pymupdf_text")
    claim_types = [claim["claim_type"] for claim in claims["claims"]]
    assert claim_types == ["reported_n", "reported_mean_sd", "reported_p_value"]
    assert claims["claims"][1]["claim_id"] == "PCL-0002"
    assert claims["claims"][1]["source"]["page"] == 2


def test_mineru_markdown_without_page_marker_does_not_guess_page():
    text = "# Results\nThe measured value was 4.5 ± 0.7, p < 0.001."

    claims = extract_paper_claims(
        text,
        file_name="paper.pdf",
        extraction_method="mineru_markdown",
    )

    assert claims["summary"]["reported_mean_sd_count"] == 1
    assert claims["summary"]["reported_p_value_count"] == 1
    for claim in claims["claims"]:
        assert claim["source"]["page"] is None
        assert claim["source"]["page_detection"] == "missing_page_marker"
        assert claim["source"]["extraction_method"] == "mineru_markdown"


def test_mean_prose_parenthetical_sd_is_extracted():
    text = "[Page 3]\nResults\nMean age was 23.5 (SD = 2.1), while unrelated values were stable."

    records = extract_means_and_sds_with_sources(text)

    assert len(records) == 1
    assert records[0]["mean"] == 23.5
    assert records[0]["sd"] == 2.1
    assert records[0]["pattern_type"] == "mean_prose_sd_parenthetical"


def test_extract_paper_claims_truncates_after_document_order_sorting():
    text = "[Page 1]\nResults\np = 0.04\n" + "\n".join(
        f"late value {idx}: {idx + 10}.0 ± 1.0" for idx in range(5)
    )

    claims = extract_paper_claims(text, max_claims=3)

    assert [claim["claim_type"] for claim in claims["claims"]] == [
        "reported_p_value",
        "reported_mean_sd",
        "reported_mean_sd",
    ]
    assert claims["claims"][0]["claim_id"] == "PCL-0001"
    assert claims["warnings"]
