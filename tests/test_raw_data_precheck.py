"""原始数据确定性预检测试。"""

from academic_fraud_detector.utils.raw_data_precheck import run_raw_data_precheck


def dataset(dataset_id, values, *, suffixes=None, digits=None, label=None):
    suffixes = suffixes or {}
    digits = digits if digits is not None else [None] * len(values)
    return {
        "dataset_id": dataset_id,
        "label": label or dataset_id,
        "source": {
            "file_path": f"/tmp/{dataset_id}.xlsx",
            "file_name": f"{dataset_id}.xlsx",
            "sheet": "Sheet1",
            "orientation": "column",
            "range": "B2:B99",
            "header": label or dataset_id,
        },
        "values": values,
        "raw_values": [str(v) for v in values],
        "n": len(values),
        "last_decimal_digits": digits,
        "decimal_suffixes": {
            "1": suffixes.get("1", []),
            "2": suffixes.get("2", []),
            "3": suffixes.get("3", []),
        },
        "is_designed_sequence_candidate": False,
    }


def run_for(*datasets):
    return run_raw_data_precheck({
        "datasets": list(datasets),
        "profile": {"dataset_count": len(datasets)},
    })


def evidence_types(precheck):
    return {item["evidence_type"] for item in precheck["deterministic_findings"]}


def test_detects_last_digit_overrepresentation_and_absence_probability():
    digits = [5] * 26 + [0] * 7 + [1] * 7 + [2] * 7 + [4] * 6 + [6] * 6 + [7] * 5 + [8] * 3 + [9] * 3
    values = [idx + digit / 10 for idx, digit in enumerate(digits, start=1)]
    precheck = run_for(dataset("digits", values, digits=digits))

    findings = precheck["deterministic_findings"]
    over = [f for f in findings if f["evidence_type"] == "last_digit_overrepresentation"]
    absence = [f for f in findings if f["evidence_type"] == "last_digit_absence"]

    assert over
    digit5 = next(f for f in over if f["statistics"]["digit"] == "5")
    assert digit5["statistics"]["observed_count"] == 26
    assert digit5["statistics"]["p_value"] < 1e-7
    assert "q_value" in digit5["statistics"]

    assert absence
    digit3 = next(f for f in absence if f["statistics"]["digit"] == "3")
    assert digit3["statistics"]["p_value"] == 0.9 ** 70
    assert digit3["statistics"]["any_digit_missing_probability"] > digit3["statistics"]["p_value"]


def test_detects_approximate_arithmetic_sequence():
    values = [10.0, 11.01, 11.99, 13.0, 14.02, 15.01, 16.0, 17.0, 18.01, 19.0]
    precheck = run_for(dataset("ap", values, label="Observed response"))

    arithmetic = [
        f for f in precheck["deterministic_findings"]
        if f["evidence_type"] == "approximate_arithmetic_sequence"
    ]

    assert arithmetic
    assert arithmetic[0]["statistics"]["cv_of_diffs"] < 0.05
    assert arithmetic[0]["statistics"]["r_squared"] > 0.99


def test_detects_permutation_invariant_duplicate_values():
    left = [1.23, 3.41, 2.89, 4.57, 6.02, 5.11, 8.76, 7.33]
    right = [7.33, 1.23, 5.11, 8.76, 2.89, 6.02, 3.41, 4.57]
    precheck = run_for(dataset("left", left), dataset("right", right))

    matches = [
        f for f in precheck["deterministic_findings"]
        if f["evidence_type"] == "raw_numeric_similarity"
    ]

    assert matches
    assert matches[0]["statistics"]["match_type"] == "permutation_exact"
    assert matches[0]["severity"] == "critical"


def test_detects_decimal_suffix_reuse_across_datasets():
    suffixes = ["12", "34", "56", "78", "90", "13", "24", "35", "46", "57"]
    left_values = [10 + int(suffix) / 100 for suffix in suffixes]
    right_values = [90 + int(suffix) / 100 for suffix in reversed(suffixes)]
    precheck = run_for(
        dataset("left_suffix", left_values, suffixes={"2": suffixes}),
        dataset("right_suffix", right_values, suffixes={"2": list(reversed(suffixes))}),
    )

    assert "decimal_suffix_reuse_across_datasets" in evidence_types(precheck)
