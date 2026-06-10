"""原始数据确定性预检测试。"""

from academic_fraud_detector.utils.raw_data_precheck import run_raw_data_precheck


def dataset(
    dataset_id,
    values,
    *,
    suffixes=None,
    digits=None,
    label=None,
    raw_values=None,
    source=None,
    designed=False,
):
    suffixes = suffixes or {}
    digits = digits if digits is not None else [None] * len(values)
    source = source or {
        "file_path": f"/tmp/{dataset_id}.xlsx",
        "file_name": f"{dataset_id}.xlsx",
        "sheet": "Sheet1",
        "orientation": "column",
        "range": "B2:B99",
        "header": label or dataset_id,
    }
    return {
        "dataset_id": dataset_id,
        "label": label or dataset_id,
        "source": source,
        "values": values,
        "raw_values": raw_values or [str(v) for v in values],
        "n": len(values),
        "last_decimal_digits": digits,
        "decimal_suffixes": {
            "1": suffixes.get("1", []),
            "2": suffixes.get("2", []),
            "3": suffixes.get("3", []),
        },
        "is_designed_sequence_candidate": designed,
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


def test_zero_datasets_do_not_trigger_raw_numeric_similarity():
    precheck = run_for(
        dataset("zero_left", [0.0] * 8, raw_values=["0"] * 8),
        dataset("zero_right", [0.0] * 8, raw_values=["0"] * 8),
    )

    matches = [
        f for f in precheck["deterministic_findings"]
        if f["evidence_type"] == "raw_numeric_similarity"
    ]

    assert not matches


def test_zero_dominated_datasets_do_not_trigger_raw_numeric_similarity():
    values = [0.0] * 8 + [1.0]
    precheck = run_for(dataset("mostly_zero_left", values), dataset("mostly_zero_right", values))

    matches = [
        f for f in precheck["deterministic_findings"]
        if f["evidence_type"] == "raw_numeric_similarity"
    ]

    assert not matches


def test_informative_duplicate_with_some_zeros_still_triggers_raw_numeric_similarity():
    values = [0.0, 0.0, 0.0, 1.1, 2.2, 3.3, 4.4, 5.5]
    precheck = run_for(dataset("informative_left", values), dataset("informative_right", values))

    matches = [
        f for f in precheck["deterministic_findings"]
        if f["evidence_type"] == "raw_numeric_similarity"
    ]

    assert matches
    assert matches[0]["statistics"]["match_type"] == "same_order_exact"


def test_raw_similarity_source_location_includes_table_metadata():
    left_source = {
        "file_path": "/tmp/source.xlsx",
        "file_name": "source.xlsx",
        "sheet": "Fig1",
        "orientation": "column",
        "range": "B2:B9",
        "header": "Signal",
        "table_id": "source.xlsx::Fig1::B2:B9",
        "table_range": "B2:B9",
        "table_title": "Fig.1a",
    }
    right_source = {**left_source, "range": "C2:C9", "table_range": "C2:C9"}
    values = [1.23, 3.41, 2.89, 4.57, 6.02, 5.11, 8.76, 7.33]
    precheck = run_for(
        dataset("left_table", values, source=left_source),
        dataset("right_table", list(reversed(values)), source=right_source),
    )

    match = next(
        f for f in precheck["deterministic_findings"]
        if f["evidence_type"] == "raw_numeric_similarity"
    )

    location = match["affected_datasets"][0]["source_location"]
    assert location["table_id"] == "source.xlsx::Fig1::B2:B9"
    assert location["table_range"] == "B2:B9"
    assert location["table_title"] == "Fig.1a"


def test_detects_decimal_suffix_reuse_across_datasets():
    suffixes = ["12", "34", "56", "78", "90", "13", "24", "35", "46", "57"]
    left_values = [10 + int(suffix) / 100 for suffix in suffixes]
    right_values = [90 + int(suffix) / 100 for suffix in reversed(suffixes)]
    precheck = run_for(
        dataset("left_suffix", left_values, suffixes={"2": suffixes}),
        dataset("right_suffix", right_values, suffixes={"2": list(reversed(suffixes))}),
    )

    assert "decimal_suffix_reuse_across_datasets" in evidence_types(precheck)


def test_detects_constant_numeric_series():
    precheck = run_for(dataset("constant", [1.23] * 8, raw_values=["1.23"] * 8))

    constants = [
        item for item in precheck["deterministic_findings"]
        if item["evidence_type"] == "constant_numeric_series"
    ]

    assert constants
    assert constants[0]["statistics"]["unique_numeric_count"] == 1
    assert constants[0]["severity"] == "high"


def test_design_variable_constant_series_is_downgraded():
    regular = run_for(dataset("regular_constant", [5.0] * 8, raw_values=["5.0"] * 8))
    designed = run_for(
        dataset(
            "dose_constant",
            [5.0] * 8,
            raw_values=["5.0"] * 8,
            label="Dose",
            designed=True,
        )
    )

    regular_item = next(
        item for item in regular["deterministic_findings"]
        if item["evidence_type"] == "constant_numeric_series"
    )
    designed_item = next(
        item for item in designed["deterministic_findings"]
        if item["evidence_type"] == "constant_numeric_series"
    )

    assert designed_item["severity"] != "high"
    assert designed_item["confidence_score"] < regular_item["confidence_score"]
    assert "设计变量" in designed_item["alternative_explanations"][0]


def test_detects_positionwise_near_duplicate_with_one_changed_value():
    left = [1.11, 2.22, 3.33, 4.44, 5.55, 6.66, 7.77, 8.88]
    right = [1.11, 2.22, 3.33, 4.44, 5.55, 6.66, 7.77, 8.89]
    precheck = run_for(dataset("left_near", left), dataset("right_near", right))

    near = [
        item for item in precheck["deterministic_findings"]
        if item["evidence_type"] == "near_duplicate_numeric_series"
    ]

    assert near
    assert near[0]["statistics"]["equal_position_count"] == 7
    assert near[0]["statistics"]["different_position_count"] == 1
    assert near[0]["statistics"]["different_positions"][-1]["index"] == 7


def test_detects_positionwise_decimal_suffix_reuse_conservatively():
    suffixes = ["11", "22", "33", "44", "55", "66", "77", "88"]
    left_values = [10 + int(suffix) / 100 for suffix in suffixes]
    right_values = [90 + int(suffix) / 100 for suffix in suffixes]
    precheck = run_for(
        dataset("left_tail", left_values, suffixes={"2": suffixes}),
        dataset("right_tail", right_values, suffixes={"2": suffixes}),
    )

    near = [
        item for item in precheck["deterministic_findings"]
        if item["evidence_type"] == "near_duplicate_numeric_series"
        and item["statistics"]["match_type"] == "positionwise_decimal_suffix2_reuse"
    ]

    assert near
    assert near[0]["statistics"]["suffix2_equal_ratio"] == 1.0
    assert near[0]["severity"] != "critical"


def test_detects_repeated_fragment_within_dataset():
    precheck = run_for(dataset("fragment", [1.1, 2.2, 3.3, 9.9, 1.1, 2.2, 3.3]))

    fragments = [
        item for item in precheck["deterministic_findings"]
        if item["evidence_type"] == "repeated_numeric_fragment"
        and item["statistics"].get("within_dataset")
    ]

    assert fragments
    assert fragments[0]["statistics"]["fragment_length"] == 3
    assert fragments[0]["statistics"]["occurrence_count"] == 2


def test_detects_repeated_fragment_across_datasets():
    left_source = {
        "file_path": "/tmp/source.xlsx",
        "file_name": "source.xlsx",
        "sheet": "Fig3",
        "orientation": "column",
        "range": "B2:B9",
        "header": "Fig3",
    }
    right_source = {**left_source, "sheet": "Fig4", "range": "C2:C9", "header": "Fig4"}
    precheck = run_for(
        dataset("left_fragment", [1, 2, 3, 4, 10, 11, 12, 13], source=left_source),
        dataset("right_fragment", [8, 9, 1, 2, 3, 4, 14, 15], source=right_source),
    )

    fragments = [
        item for item in precheck["deterministic_findings"]
        if item["evidence_type"] == "repeated_numeric_fragment"
        and item["statistics"].get("cross_dataset")
    ]

    assert fragments
    assert fragments[0]["statistics"]["fragment_length"] == 4
    assert fragments[0]["statistics"]["cross_source_reuse_candidate"]


def test_small_sample_near_duplicate_is_not_high_confidence():
    precheck = run_for(
        dataset("small_left", [1.0, 2.0, 3.0, 4.0]),
        dataset("small_right", [1.0, 2.0, 3.0, 4.1]),
    )

    near = [
        item for item in precheck["deterministic_findings"]
        if item["evidence_type"] == "near_duplicate_numeric_series"
    ]

    assert not near or all(item["severity"] not in {"high", "critical"} for item in near)


def test_positionwise_pair_limit_returns_warning():
    from academic_fraud_detector.utils.raw_data_precheck import detect_near_duplicate_numeric_datasets

    datasets = [dataset(f"limited_{idx}", [idx, 2, 3, 4, 5, 6]) for idx in range(4)]
    _evidence, warnings = detect_near_duplicate_numeric_datasets(datasets, max_pairs=1)

    assert warnings
    assert "仅比较前 1 对" in warnings[0]
