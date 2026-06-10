"""原始数据确定性预检：只输出可追溯的结构化证据。"""

from __future__ import annotations

import json
import math
from collections import Counter
from itertools import combinations
from typing import Any

import numpy as np
from scipy import stats

SEVERITY_ORDER = {"informational": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
MAX_EVIDENCE_ITEMS_FOR_CONTEXT = 300
MAX_POSITIONWISE_DATASET_PAIRS = 5000
MAX_FRAGMENT_DATASET_PAIRS = 5000


def _clean_values(dataset: dict[str, Any]) -> np.ndarray:
    values = []
    for value in dataset.get("values", []):
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return np.asarray(values, dtype=np.float64)


def _is_low_information_numeric_series(values: np.ndarray) -> bool:
    """判断序列是否主要由 0 占位组成，不适合作为跨数据集重复证据。"""
    if len(values) < 5:
        return True
    zero_mask = np.isclose(values, 0.0, rtol=0.0, atol=1e-12)
    zero_count = int(np.sum(zero_mask))
    if zero_count == len(values):
        return True
    nonzero_count = len(values) - zero_count
    zero_ratio = zero_count / len(values)
    return zero_ratio >= 0.8 and nonzero_count < 5


def _source_location(dataset: dict[str, Any]) -> dict[str, Any]:
    source = dataset.get("source") or {}
    return {
        "file_path": source.get("file_path", ""),
        "file_name": source.get("file_name", ""),
        "sheet": source.get("sheet", ""),
        "range": source.get("range", ""),
        "orientation": source.get("orientation", ""),
        "header": source.get("header", ""),
        "table_id": source.get("table_id", ""),
        "table_range": source.get("table_range", ""),
        "table_title": source.get("table_title", ""),
    }


def _dataset_ref(dataset: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_id": dataset.get("dataset_id", ""),
        "label": dataset.get("label", ""),
        "source_location": _source_location(dataset),
    }


def _make_evidence(
    *,
    evidence_type: str,
    severity: str,
    confidence_score: float,
    title: str,
    affected_datasets: list[dict[str, Any]],
    statistics: dict[str, Any],
    deterministic_basis: str,
    alternative_explanations: list[str] | None = None,
    recommended_human_check: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "evidence_id": "",
        "evidence_type": evidence_type,
        "title": title,
        "source_type": "xlsx_raw_data",
        "source_location": (
            affected_datasets[0].get("source_location", {}) if affected_datasets else {}
        ),
        "affected_datasets": affected_datasets,
        "statistics": statistics,
        "severity": severity,
        "confidence_score": round(float(confidence_score), 4),
        "confidence_label": _confidence_label(confidence_score),
        "deterministic_basis": deterministic_basis,
        "alternative_explanations": alternative_explanations or [],
        "recommended_human_check": recommended_human_check or [],
    }


def _confidence_label(score: float) -> str:
    if score >= 0.95:
        return "very_high"
    if score >= 0.80:
        return "high"
    if score >= 0.60:
        return "medium"
    return "low"


def _linear_fit_r2(values: np.ndarray) -> tuple[float, float, float]:
    x = np.arange(len(values), dtype=np.float64)
    if len(values) < 2:
        return 0.0, 0.0, 0.0
    slope, intercept = np.polyfit(x, values, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((values - predicted) ** 2))
    ss_tot = float(np.sum((values - np.mean(values)) ** 2))
    r2 = 1.0 if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot
    max_residual = float(np.max(np.abs(values - predicted)))
    return float(r2), float(slope), max_residual


def _severity_from_arithmetic(n: int, cv_diff: float, r2: float, sorted_mode: bool) -> str | None:
    if n < 6:
        return None
    if sorted_mode:
        if n >= 10 and cv_diff <= 0.015 and r2 >= 0.9995:
            return "critical"
        if n >= 8 and cv_diff <= 0.03 and r2 >= 0.999:
            return "high"
        if n >= 8 and cv_diff <= 0.05 and r2 >= 0.998:
            return "medium"
        return None
    if n >= 10 and cv_diff <= 0.01 and r2 >= 0.9995:
        return "critical"
    if n >= 8 and cv_diff <= 0.03 and r2 >= 0.998:
        return "high"
    if cv_diff <= 0.05 and r2 >= 0.995:
        return "medium"
    return None


def _downgrade(severity: str) -> str:
    order = ["informational", "low", "medium", "high", "critical"]
    index = max(0, order.index(severity) - 1)
    return order[index]


def _raw_texts(dataset: dict[str, Any]) -> list[str]:
    raw_values = list(dataset.get("raw_values") or [])
    values = list(dataset.get("values") or [])
    result: list[str] = []
    for index in range(max(len(raw_values), len(values))):
        if index < len(raw_values) and raw_values[index] not in (None, ""):
            result.append(str(raw_values[index]).strip())
        elif index < len(values):
            result.append(str(values[index]).strip())
        else:
            result.append("")
    return result


def _decimal_suffixes(dataset: dict[str, Any], suffix_length: int) -> list[str | None]:
    suffixes = (dataset.get("decimal_suffixes") or {}).get(str(suffix_length), [])
    return [None if suffix is None or str(suffix) == "" else str(suffix) for suffix in suffixes]


def _apply_design_variable_downgrade(
    severity: str,
    confidence: float,
    datasets: list[dict[str, Any]],
) -> tuple[str, float, list[str]]:
    designed_count = sum(1 for dataset in datasets if dataset.get("is_designed_sequence_candidate"))
    if designed_count == 0:
        return severity, confidence, []

    severity = _downgrade(severity)
    confidence *= 0.65
    if designed_count == len(datasets) and len(datasets) > 1:
        severity = _downgrade(severity)
        confidence *= 0.80
    return severity, confidence, [
        "表头/标签显示至少一个数据集可能是剂量、时间、编号、分组或标准曲线等设计变量，已降低置信度。"
    ]


def _source_pair_context(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_source = left.get("source") or {}
    right_source = right.get("source") or {}
    same_file = (left_source.get("file_path") or left_source.get("file_name")) == (
        right_source.get("file_path") or right_source.get("file_name")
    )
    same_sheet = left_source.get("sheet") == right_source.get("sheet")
    same_range = left_source.get("range") == right_source.get("range")
    same_label = str(left.get("label", "")).strip() == str(right.get("label", "")).strip()
    return {
        "same_file": same_file,
        "same_sheet": same_sheet,
        "same_range": same_range,
        "same_label": same_label,
        "cross_source_reuse_candidate": not (same_file and same_sheet and same_range),
    }


def _pairwise_scan_stats(total_pairs: int, max_pairs: int) -> dict[str, Any]:
    scanned = min(total_pairs, max_pairs)
    return {
        "pairwise_comparisons_total": total_pairs,
        "pairwise_comparisons_scanned": scanned,
        "pairwise_comparisons_skipped": max(total_pairs - scanned, 0),
        "pairwise_comparison_limit_reached": total_pairs > max_pairs,
    }


def _positionwise_similarity(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_values = _clean_values(left)
    right_values = _clean_values(right)
    n = min(len(left_values), len(right_values))
    left_raw = _raw_texts(left)
    right_raw = _raw_texts(right)
    left_suffix2 = _decimal_suffixes(left, 2)
    right_suffix2 = _decimal_suffixes(right, 2)

    equal_count = 0
    near_equal_count = 0
    raw_equal_count = 0
    suffix2_equal_count = 0
    suffix2_valid_count = 0
    different_positions: list[dict[str, Any]] = []

    for index in range(n):
        left_value = float(left_values[index])
        right_value = float(right_values[index])
        scale = max(abs(left_value), abs(right_value), 1.0)
        abs_diff = abs(left_value - right_value)
        equal = abs_diff <= 1e-12
        near_equal = abs_diff <= max(1e-9, 1e-6 * scale)
        if equal:
            equal_count += 1
        if near_equal:
            near_equal_count += 1
        if index < len(left_raw) and index < len(right_raw) and left_raw[index] == right_raw[index]:
            raw_equal_count += 1
        if index < len(left_suffix2) and index < len(right_suffix2):
            left_suffix = left_suffix2[index]
            right_suffix = right_suffix2[index]
            if left_suffix is not None and right_suffix is not None:
                suffix2_valid_count += 1
                if left_suffix == right_suffix:
                    suffix2_equal_count += 1
        if not equal and len(different_positions) < 10:
            different_positions.append({
                "index": index,
                "left_value": round(left_value, 10),
                "right_value": round(right_value, 10),
                "left_raw": left_raw[index] if index < len(left_raw) else "",
                "right_raw": right_raw[index] if index < len(right_raw) else "",
            })

    def ratio(count: int, denominator: int) -> float:
        return round(count / denominator, 6) if denominator else 0.0

    return {
        "n_compared": int(n),
        "equal_position_count": equal_count,
        "equal_position_ratio": ratio(equal_count, n),
        "near_equal_position_count": near_equal_count,
        "near_equal_position_ratio": ratio(near_equal_count, n),
        "raw_text_equal_count": raw_equal_count,
        "raw_text_equal_ratio": ratio(raw_equal_count, n),
        "suffix2_valid_count": suffix2_valid_count,
        "suffix2_equal_count": suffix2_equal_count,
        "suffix2_equal_ratio": ratio(suffix2_equal_count, suffix2_valid_count),
        "different_position_count": int(n - equal_count),
        "different_positions": different_positions,
    }


def detect_arithmetic_sequences(datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for dataset in datasets:
        values = _clean_values(dataset)
        if len(values) < 6:
            continue
        modes = [("original_order", values), ("sorted_values", np.sort(values))]
        for order_mode, series in modes:
            diffs = np.diff(series)
            mean_abs_diff = float(np.mean(np.abs(diffs)))
            if mean_abs_diff <= 1e-12:
                continue
            cv_diff = float(np.std(diffs) / mean_abs_diff)
            r2, slope, max_residual = _linear_fit_r2(series)
            severity = _severity_from_arithmetic(
                len(series), cv_diff, r2, sorted_mode=(order_mode == "sorted_values")
            )
            if severity is None:
                continue

            designed = bool(dataset.get("is_designed_sequence_candidate"))
            confidence = {"critical": 0.96, "high": 0.88, "medium": 0.72}[severity]
            alternatives = ["若该列是剂量、时间点、编号或标准曲线浓度，近似等差可能由实验设计导致。"]
            if designed:
                severity = _downgrade(severity)
                confidence *= 0.65
                alternatives.insert(0, "表头/标签显示该序列可能是设计变量，已降低置信度。")

            evidence.append(
                _make_evidence(
                    evidence_type="approximate_arithmetic_sequence",
                    severity=severity,
                    confidence_score=confidence,
                    title="发现近似等差数列",
                    affected_datasets=[_dataset_ref(dataset)],
                    statistics={
                        "n": int(len(series)),
                        "order_mode": order_mode,
                        "cv_of_diffs": round(cv_diff, 8),
                        "r_squared": round(r2, 8),
                        "slope": round(slope, 8),
                        "max_residual": round(max_residual, 8),
                        "first_values": [round(float(v), 8) for v in series[:10]],
                    },
                    deterministic_basis=(
                        f"{dataset.get('label', '')} 在 {order_mode} 下 {len(series)} 个值的"
                        f"相邻差值 CV={cv_diff:.6g}，线性拟合 R²={r2:.6g}，"
                        "接近人工构造的等差序列。"
                    ),
                    alternative_explanations=alternatives,
                    recommended_human_check=["核对该数据列是否为实验设计变量而非实验观测值。"],
                )
            )
    return evidence


def detect_constant_numeric_datasets(datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for dataset in datasets:
        values = _clean_values(dataset)
        n = len(values)
        if n < 5:
            continue
        rounded_values = [round(float(value), 10) for value in values]
        unique_numeric = sorted(set(rounded_values))
        raw_texts = _raw_texts(dataset)[:n]
        non_empty_raw = [text for text in raw_texts if text != ""]
        unique_raw = sorted(set(non_empty_raw))
        numeric_constant = len(unique_numeric) == 1
        raw_constant = len(non_empty_raw) == n and len(unique_raw) == 1
        if not numeric_constant and not raw_constant:
            continue

        severity = "high" if n >= 8 and numeric_constant and raw_constant else "medium"
        confidence = 0.88 if severity == "high" else 0.72
        severity, confidence, design_alternatives = _apply_design_variable_downgrade(
            severity, confidence, [dataset]
        )
        constant_value = unique_numeric[0] if unique_numeric else None
        constant_raw_value = unique_raw[0] if unique_raw else None
        evidence.append(
            _make_evidence(
                evidence_type="constant_numeric_series",
                severity=severity,
                confidence_score=confidence,
                title="发现整行/整列数值完全一致",
                affected_datasets=[_dataset_ref(dataset)],
                statistics={
                    "n": int(n),
                    "unique_numeric_count": len(unique_numeric),
                    "unique_raw_text_count": len(unique_raw),
                    "constant_value": constant_value,
                    "constant_raw_value": constant_raw_value,
                    "numeric_constant": numeric_constant,
                    "raw_text_constant": raw_constant,
                    "designed_sequence_candidate": bool(dataset.get("is_designed_sequence_candidate")),
                    "first_values": [round(float(value), 10) for value in values[:10]],
                    "first_raw_values": raw_texts[:10],
                },
                deterministic_basis=(
                    f"{dataset.get('label', '')} 中 {n} 个可分析数值"
                    f"{'及显示文本' if raw_constant else ''}完全一致，"
                    "提示可能存在整列/整行复制或错误填充。"
                ),
                alternative_explanations=design_alternatives + [
                    "该列可能是固定实验条件、对照编码、批次编号或其他非观测型常量。"
                ],
                recommended_human_check=["核对该数据范围是否应为真实观测值，而非实验设计常量。"],
            )
        )
    return evidence


def detect_near_duplicate_numeric_datasets(
    datasets: list[dict[str, Any]], max_pairs: int = MAX_POSITIONWISE_DATASET_PAIRS
) -> tuple[list[dict[str, Any]], list[str]]:
    evidence: list[dict[str, Any]] = []
    warnings: list[str] = []
    comparable = [dataset for dataset in datasets if len(_clean_values(dataset)) >= 6]
    total_pairs = len(comparable) * (len(comparable) - 1) // 2
    scan_stats = _pairwise_scan_stats(total_pairs, max_pairs)
    if total_pairs > max_pairs:
        warnings.append(
            f"逐行相似性 dataset 配对数为 {total_pairs}，仅比较前 {max_pairs} 对；"
            "部分低优先级跨列/跨表近重复检测可能未覆盖。"
        )

    seen_pairs: set[tuple[str, str]] = set()
    for pair_index, (left, right) in enumerate(combinations(comparable, 2)):
        if pair_index >= max_pairs:
            break
        if _same_source_range(left, right):
            continue
        left_values = _clean_values(left)
        right_values = _clean_values(right)
        if len(left_values) != len(right_values):
            continue
        similarity = _positionwise_similarity(left, right)
        n = int(similarity["n_compared"])
        if n < 6:
            continue
        if similarity["equal_position_count"] == n:
            # 完全同序复用已由 raw_numeric_similarity 的 same_order_exact 负责。
            continue

        equal_ratio = float(similarity["equal_position_ratio"])
        near_ratio = float(similarity["near_equal_position_ratio"])
        suffix2_ratio = float(similarity["suffix2_equal_ratio"])
        diff_count = int(similarity["different_position_count"])
        suffix2_valid = int(similarity["suffix2_valid_count"])
        match_type: str | None = None
        severity: str | None = None
        confidence = 0.0

        if n >= 8 and equal_ratio >= 0.85 and diff_count <= max(2, math.floor(n * 0.15)):
            match_type = "positionwise_near_duplicate"
            severity = "high"
            confidence = 0.89
        elif n >= 8 and near_ratio >= 0.90 and diff_count <= max(3, math.floor(n * 0.20)):
            match_type = "positionwise_near_equal"
            severity = "medium"
            confidence = 0.78
        elif n >= 8 and suffix2_valid >= max(8, math.ceil(n * 0.80)) and suffix2_ratio >= 0.85:
            match_type = "positionwise_decimal_suffix2_reuse"
            severity = "medium"
            confidence = 0.74
        elif n >= 6 and equal_ratio >= 0.83 and diff_count <= 1:
            match_type = "small_sample_positionwise_near_duplicate"
            severity = "medium"
            confidence = 0.70

        if match_type is None or severity is None:
            continue

        pair_key = tuple(sorted([left.get("dataset_id", ""), right.get("dataset_id", "")]))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        severity, confidence, design_alternatives = _apply_design_variable_downgrade(
            severity, confidence, [left, right]
        )
        source_context = _source_pair_context(left, right)
        statistics = {
            "match_type": match_type,
            **similarity,
            **source_context,
            **scan_stats,
        }
        if match_type == "positionwise_decimal_suffix2_reuse":
            basis = (
                f"{left.get('label', '')} 与 {right.get('label', '')} 的小数后两位在 "
                f"{similarity['suffix2_equal_count']}/{similarity['suffix2_valid_count']} 个可比位置一致 "
                f"({suffix2_ratio:.1%})，提示可能复制小数部分后修改其他位数。"
            )
        else:
            basis = (
                f"{left.get('label', '')} 与 {right.get('label', '')} 在相同顺序下 "
                f"{similarity['equal_position_count']}/{n} 个位置数值完全相同 "
                f"({equal_ratio:.1%})，仅 {diff_count} 个位置不同。"
            )
        evidence.append(
            _make_evidence(
                evidence_type="near_duplicate_numeric_series",
                severity=severity,
                confidence_score=confidence,
                title="发现两组数值逐行高度相似",
                affected_datasets=[_dataset_ref(left), _dataset_ref(right)],
                statistics=statistics,
                deterministic_basis=basis,
                alternative_explanations=design_alternatives + [
                    "两个数据范围可能合法复用同一标准曲线、同一对照组或同一派生数据，需要结合图注和方法核对。"
                ],
                recommended_human_check=["核对两组数据是否声称来自相互独立的实验组、时间点或 figure/panel。"],
            )
        )
    return evidence, warnings


def _numeric_fragment_map(values: np.ndarray, window_size: int) -> dict[tuple[float, ...], list[int]]:
    fragments: dict[tuple[float, ...], list[int]] = {}
    if len(values) < window_size:
        return fragments
    for start in range(0, len(values) - window_size + 1):
        key = tuple(round(float(value), 8) for value in values[start:start + window_size])
        if len(set(key)) <= 1:
            continue
        fragments.setdefault(key, []).append(start)
    return fragments


def _format_fragment_values(fragment: tuple[float, ...]) -> list[float]:
    return [round(float(value), 8) for value in fragment]


def detect_repeated_numeric_fragments(
    datasets: list[dict[str, Any]], max_pairs: int = MAX_FRAGMENT_DATASET_PAIRS
) -> tuple[list[dict[str, Any]], list[str]]:
    evidence: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen: set[tuple[str, str, tuple[float, ...]]] = set()

    for dataset in datasets:
        values = _clean_values(dataset)
        n = len(values)
        if n < 6 or len(set(round(float(value), 8) for value in values)) <= 1:
            continue
        emitted_for_dataset = False
        for window_size in (5, 4, 3):
            if n < window_size * 2:
                continue
            fragments = _numeric_fragment_map(values, window_size)
            repeated = [
                (fragment, starts) for fragment, starts in fragments.items()
                if len(starts) >= 2
            ]
            if not repeated:
                continue
            repeated.sort(key=lambda item: (-len(item[1]), item[1][0]))
            fragment, starts = repeated[0]
            severity = "high" if window_size >= 4 and n >= 8 else "medium"
            confidence = 0.84 if severity == "high" else 0.72
            severity, confidence, design_alternatives = _apply_design_variable_downgrade(
                severity, confidence, [dataset]
            )
            coverage = min(1.0, len(starts) * window_size / n)
            evidence.append(
                _make_evidence(
                    evidence_type="repeated_numeric_fragment",
                    severity=severity,
                    confidence_score=confidence,
                    title="发现重复数值片段",
                    affected_datasets=[_dataset_ref(dataset)],
                    statistics={
                        "fragment_length": window_size,
                        "fragment_values": _format_fragment_values(fragment),
                        "occurrence_count": len(starts),
                        "occurrences": [{"dataset_id": dataset.get("dataset_id", ""), "start_index": start} for start in starts[:10]],
                        "within_dataset": True,
                        "cross_dataset": False,
                        "match_mode": "numeric_rounded_8dp",
                        "coverage_ratio": round(coverage, 6),
                    },
                    deterministic_basis=(
                        f"{dataset.get('label', '')} 内长度为 {window_size} 的数值片段 "
                        f"{_format_fragment_values(fragment)} 重复出现 {len(starts)} 次。"
                    ),
                    alternative_explanations=design_alternatives + [
                        "周期性实验设计、重复测量模板或派生计算可能造成短片段重复。"
                    ],
                    recommended_human_check=["核对这些重复片段是否来自独立样本或独立实验条件。"],
                )
            )
            emitted_for_dataset = True
            break
        if emitted_for_dataset:
            continue

    comparable = [dataset for dataset in datasets if len(_clean_values(dataset)) >= 6]
    total_pairs = len(comparable) * (len(comparable) - 1) // 2
    scan_stats = _pairwise_scan_stats(total_pairs, max_pairs)
    if total_pairs > max_pairs:
        warnings.append(
            f"重复片段 dataset 配对数为 {total_pairs}，仅比较前 {max_pairs} 对；"
            "部分跨表/跨列片段复用检测可能未覆盖。"
        )

    for pair_index, (left, right) in enumerate(combinations(comparable, 2)):
        if pair_index >= max_pairs:
            break
        if _same_source_range(left, right):
            continue
        left_values = _clean_values(left)
        right_values = _clean_values(right)
        min_n = min(len(left_values), len(right_values))
        if min_n < 6:
            continue
        for window_size in (5, 4, 3):
            if min_n < window_size:
                continue
            left_fragments = _numeric_fragment_map(left_values, window_size)
            right_fragments = _numeric_fragment_map(right_values, window_size)
            shared = sorted(set(left_fragments) & set(right_fragments))
            if not shared:
                continue
            fragment = shared[0]
            pair_key = tuple(sorted([left.get("dataset_id", ""), right.get("dataset_id", "")]))
            seen_key = (pair_key[0], pair_key[1], fragment)
            if seen_key in seen:
                break
            seen.add(seen_key)
            severity = "high" if window_size >= 4 and min_n >= 8 else "medium"
            confidence = 0.86 if severity == "high" else 0.74
            severity, confidence, design_alternatives = _apply_design_variable_downgrade(
                severity, confidence, [left, right]
            )
            statistics = {
                "fragment_length": window_size,
                "fragment_values": _format_fragment_values(fragment),
                "occurrence_count": len(left_fragments[fragment]) + len(right_fragments[fragment]),
                "occurrences": [
                    {"dataset_id": left.get("dataset_id", ""), "start_index": start}
                    for start in left_fragments[fragment][:5]
                ] + [
                    {"dataset_id": right.get("dataset_id", ""), "start_index": start}
                    for start in right_fragments[fragment][:5]
                ],
                "within_dataset": False,
                "cross_dataset": True,
                "match_mode": "numeric_rounded_8dp",
                "coverage_ratio": round(window_size / max(min_n, 1), 6),
                **_source_pair_context(left, right),
                **scan_stats,
            }
            evidence.append(
                _make_evidence(
                    evidence_type="repeated_numeric_fragment",
                    severity=severity,
                    confidence_score=confidence,
                    title="发现跨数据集重复数值片段",
                    affected_datasets=[_dataset_ref(left), _dataset_ref(right)],
                    statistics=statistics,
                    deterministic_basis=(
                        f"{left.get('label', '')} 与 {right.get('label', '')} 共享长度为 "
                        f"{window_size} 的数值片段 {_format_fragment_values(fragment)}，"
                        "提示可能存在局部序列复用。"
                    ),
                    alternative_explanations=design_alternatives + [
                        "两个表格可能合法引用同一对照、标准曲线或派生计算结果，需要人工核对。"
                    ],
                    recommended_human_check=["核对共享片段是否跨不同 figure、panel、实验条件或独立重复出现。"],
                )
            )
            break
    return evidence, warnings


def _rounded_counter(values: np.ndarray, decimals: int = 10) -> Counter:
    return Counter(round(float(v), decimals) for v in values)


def _multiset_containment(shorter: np.ndarray, longer: np.ndarray) -> bool:
    short_counter = _rounded_counter(shorter)
    long_counter = _rounded_counter(longer)
    return all(long_counter.get(value, 0) >= count for value, count in short_counter.items())


def _near_sorted_match(a: np.ndarray, b: np.ndarray) -> tuple[bool, float, float]:
    if len(a) != len(b) or len(a) == 0:
        return False, float("inf"), float("inf")
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    diffs = np.abs(a_sorted - b_sorted)
    scale = max(float(np.max(np.abs(a_sorted))), float(np.max(np.abs(b_sorted))), 1.0)
    max_abs = float(np.max(diffs))
    max_rel = max_abs / scale
    return bool(max_rel <= 1e-6 or max_abs <= 1e-8), max_abs, max_rel


def _same_source_range(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return _source_location(a) == _source_location(b)


def detect_numeric_similarity(
    datasets: list[dict[str, Any]], max_pairs: int = 5000
) -> tuple[list[dict[str, Any]], list[str]]:
    evidence: list[dict[str, Any]] = []
    warnings: list[str] = []
    comparable = []
    for dataset in datasets:
        values = _clean_values(dataset)
        if len(values) >= 5 and not _is_low_information_numeric_series(values):
            comparable.append(dataset)
    total_pairs = len(comparable) * (len(comparable) - 1) // 2
    if total_pairs > max_pairs:
        warnings.append(
            f"原始数据 dataset 配对数为 {total_pairs}，仅比较前 {max_pairs} 对；"
            "请缩小数据范围或提高阈值以做更完整扫描。"
        )

    seen_pairs: set[tuple[str, str, str]] = set()
    for pair_index, (left, right) in enumerate(combinations(comparable, 2)):
        if pair_index >= max_pairs:
            break
        if _same_source_range(left, right):
            continue
        a = _clean_values(left)
        b = _clean_values(right)
        if len(a) < 5 or len(b) < 5:
            continue
        pair_key = tuple(sorted([left.get("dataset_id", ""), right.get("dataset_id", "")]))

        if len(a) == len(b):
            if np.array_equal(a, b):
                match_type = "same_order_exact"
                key = (pair_key[0], pair_key[1], match_type)
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    evidence.append(
                        _similarity_evidence(left, right, match_type, "critical", 0.99, {
                            "n": int(len(a)),
                            "same_order": True,
                            "max_abs_difference": 0.0,
                        })
                    )
                continue

            max_abs = float(np.max(np.abs(a - b)))
            scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1.0)
            max_rel = max_abs / scale
            if max_rel <= 0.001 and len(a) >= 8:
                match_type = "same_order_near"
                key = (pair_key[0], pair_key[1], match_type)
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    evidence.append(
                        _similarity_evidence(left, right, match_type, "high", 0.91, {
                            "n": int(len(a)),
                            "same_order": True,
                            "max_abs_difference": round(max_abs, 10),
                            "max_relative_difference": round(max_rel, 10),
                        })
                    )
                continue

            sorted_match, sorted_max_abs, sorted_max_rel = _near_sorted_match(a, b)
            if sorted_match:
                exact_sorted = _rounded_counter(a) == _rounded_counter(b)
                match_type = "permutation_exact" if exact_sorted else "permutation_near"
                severity = "critical" if exact_sorted else "high"
                confidence = 0.98 if exact_sorted else 0.90
                key = (pair_key[0], pair_key[1], match_type)
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    evidence.append(
                        _similarity_evidence(left, right, match_type, severity, confidence, {
                            "n": int(len(a)),
                            "same_order": False,
                            "max_abs_difference_after_sort": round(sorted_max_abs, 10),
                            "max_relative_difference_after_sort": round(sorted_max_rel, 10),
                        })
                    )
                continue

            if len(a) >= 8 and len(b) >= 8:
                slope, intercept = np.polyfit(a, b, 1)
                predicted = slope * a + intercept
                ss_res = float(np.sum((b - predicted) ** 2))
                ss_tot = float(np.sum((b - np.mean(b)) ** 2))
                r2 = 1.0 if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot
                residual_scale = max(float(np.std(b)), 1.0)
                residual_ratio = float(np.sqrt(ss_res / len(a)) / residual_scale)
                if r2 >= 0.999 and residual_ratio <= 0.01:
                    match_type = "linear_transform_near"
                    key = (pair_key[0], pair_key[1], match_type)
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        evidence.append(
                            _similarity_evidence(left, right, match_type, "high", 0.88, {
                                "n": int(len(a)),
                                "r_squared": round(float(r2), 8),
                                "slope": round(float(slope), 8),
                                "intercept": round(float(intercept), 8),
                                "residual_ratio": round(residual_ratio, 8),
                            })
                        )

        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        short_ds, long_ds = (left, right) if len(a) <= len(b) else (right, left)
        if len(shorter) >= 5 and len(longer) > len(shorter) and _multiset_containment(shorter, longer):
            match_type = "permutation_subset_exact"
            key = tuple(sorted([short_ds.get("dataset_id", ""), long_ds.get("dataset_id", "")]))
            evidence_key = (key[0], key[1], match_type)
            if evidence_key not in seen_pairs:
                seen_pairs.add(evidence_key)
                severity = "critical" if len(shorter) >= 10 else "high"
                confidence = 0.96 if len(shorter) >= 10 else 0.87
                evidence.append(
                    _similarity_evidence(short_ds, long_ds, match_type, severity, confidence, {
                        "subset_n": int(len(shorter)),
                        "larger_n": int(len(longer)),
                        "containment_fraction": 1.0,
                    })
                )
    return evidence, warnings


def _similarity_evidence(
    left: dict[str, Any],
    right: dict[str, Any],
    match_type: str,
    severity: str,
    confidence: float,
    statistics: dict[str, Any],
) -> dict[str, Any]:
    basis_map = {
        "same_order_exact": "两个数据集在相同顺序下数值完全一致。",
        "same_order_near": "两个数据集在相同顺序下数值几乎一致。",
        "permutation_exact": "两个数据集包含完全相同的数值多重集合，但顺序可能不同。",
        "permutation_near": "两个数据集排序后数值几乎一致，提示可能复制后打乱顺序。",
        "permutation_subset_exact": "一个数据集的全部数值以多重集合形式出现在另一个数据集中。",
        "linear_transform_near": "两个数据集近似满足线性缩放/平移关系。",
    }
    return _make_evidence(
        evidence_type="raw_numeric_similarity",
        severity=severity,
        confidence_score=confidence,
        title="发现原始数据重复或高度相似",
        affected_datasets=[_dataset_ref(left), _dataset_ref(right)],
        statistics={"match_type": match_type, **statistics},
        deterministic_basis=(
            f"{left.get('label', '')} 与 {right.get('label', '')}: "
            f"{basis_map.get(match_type, match_type)}"
        ),
        alternative_explanations=["若两个表格本来应复用同一标准曲线或同一对照数据，需要人工核对实验设计。"],
        recommended_human_check=["核对两个数据范围是否声称来自相互独立的实验组或重复实验。"],
    )


def _valid_suffixes(dataset: dict[str, Any], suffix_length: int) -> list[str]:
    suffixes = (dataset.get("decimal_suffixes") or {}).get(str(suffix_length), [])
    return [str(s) for s in suffixes if s is not None and str(s) != ""]


def detect_decimal_suffix_patterns(
    datasets: list[dict[str, Any]], max_pairs: int = 5000
) -> tuple[list[dict[str, Any]], list[str]]:
    evidence: list[dict[str, Any]] = []
    warnings: list[str] = []

    for dataset in datasets:
        for suffix_length in (2, 3):
            suffixes = _valid_suffixes(dataset, suffix_length)
            n = len(suffixes)
            if n < 20:
                continue
            counts = Counter(suffixes)
            suffix, count = counts.most_common(1)[0]
            p_expected = 1 / (10 ** suffix_length)
            p_value = float(stats.binom.sf(count - 1, n, p_expected))
            if count >= max(4, int(n * 0.18)) and p_value <= 0.001:
                evidence.append(
                    _make_evidence(
                        evidence_type="decimal_suffix_overrepresentation",
                        severity="high" if p_value <= 1e-5 else "medium",
                        confidence_score=0.90 if p_value <= 1e-5 else 0.76,
                        title="发现小数后缀高频重复",
                        affected_datasets=[_dataset_ref(dataset)],
                        statistics={
                            "suffix_length": suffix_length,
                            "suffix": suffix,
                            "n": n,
                            "observed_count": count,
                            "observed_frequency": round(count / n, 6),
                            "expected_probability": p_expected,
                            "test": "binomial_tail_uniform_decimal_suffix",
                            "p_value": p_value,
                            "fdr_hypothesis_count": 10 ** suffix_length,
                        },
                        deterministic_basis=(
                            f"{dataset.get('label', '')} 中 {n} 个小数后缀里，后 {suffix_length} 位"
                            f" '{suffix}' 出现 {count} 次；若后缀近似均匀，"
                            f"P(X≥{count})={p_value:.3e}。"
                        ),
                        alternative_explanations=["仪器显示精度、四舍五入规则或固定步长记录可能造成后缀偏好。"],
                        recommended_human_check=["核对 Excel 显示格式和仪器输出精度。"],
                    )
                )

    comparable = [d for d in datasets if len(_valid_suffixes(d, 2)) >= 8]
    total_pairs = len(comparable) * (len(comparable) - 1) // 2
    if total_pairs > max_pairs:
        warnings.append(
            f"小数后缀 dataset 配对数为 {total_pairs}，仅比较前 {max_pairs} 对。"
        )

    for pair_index, (left, right) in enumerate(combinations(comparable, 2)):
        if pair_index >= max_pairs:
            break
        if _same_source_range(left, right):
            continue
        for suffix_length in (2, 3):
            left_suffixes = _valid_suffixes(left, suffix_length)
            right_suffixes = _valid_suffixes(right, suffix_length)
            min_n = min(len(left_suffixes), len(right_suffixes))
            if min_n < 8:
                continue
            overlap = _counter_overlap(Counter(left_suffixes), Counter(right_suffixes))
            similarity = overlap / min_n
            if similarity >= 0.85:
                evidence.append(
                    _make_evidence(
                        evidence_type="decimal_suffix_reuse_across_datasets",
                        severity="high" if min_n >= 15 else "medium",
                        confidence_score=0.90 if min_n >= 15 else 0.78,
                        title="发现跨数据集小数后缀高度复用",
                        affected_datasets=[_dataset_ref(left), _dataset_ref(right)],
                        statistics={
                            "suffix_length": suffix_length,
                            "min_n": min_n,
                            "shared_suffix_count": overlap,
                            "suffix_multiset_similarity": round(similarity, 6),
                        },
                        deterministic_basis=(
                            f"{left.get('label', '')} 与 {right.get('label', '')} 的后 "
                            f"{suffix_length} 位小数后缀多重集合重合 {overlap}/{min_n} "
                            f"({similarity:.1%})，提示可能复制小数部分后修改整数部分或打乱顺序。"
                        ),
                        alternative_explanations=["共同仪器精度、统一四舍五入规则或同一原始来源可能造成后缀重合。"],
                        recommended_human_check=["核对两组数据是否应为相互独立测量。"],
                    )
                )
    return evidence, warnings


def _counter_overlap(left: Counter, right: Counter) -> int:
    return sum(min(count, right.get(value, 0)) for value, count in left.items())


def _prob_any_digit_missing(n: int) -> float:
    total = 0.0
    for j in range(1, 11):
        total += ((-1) ** (j + 1)) * math.comb(10, j) * ((10 - j) / 10) ** n
    return min(max(float(total), 0.0), 1.0)


def detect_last_digit_anomalies(datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for dataset in datasets:
        digits = [d for d in dataset.get("last_decimal_digits", []) if d is not None]
        digits = [int(d) for d in digits if 0 <= int(d) <= 9]
        n = len(digits)
        if n < 20:
            continue
        counts = Counter(digits)
        observed = [counts.get(d, 0) for d in range(10)]
        expected = n / 10
        dominant_digit, dominant_count = max(enumerate(observed), key=lambda item: item[1])
        missing_digits = [str(digit) for digit, count in enumerate(observed) if count == 0]
        digit_summary = {
            "dominant_digit": str(dominant_digit),
            "dominant_digit_count": dominant_count,
            "dominant_digit_frequency": round(dominant_count / n, 6),
            "missing_digits": missing_digits,
        }

        for digit, count in enumerate(observed):
            if count >= max(math.ceil(expected * 2), math.ceil(expected + 5)):
                p_value = float(stats.binom.sf(count - 1, n, 0.1))
                if p_value <= 0.01:
                    evidence.append(
                        _make_evidence(
                            evidence_type="last_digit_overrepresentation",
                            severity="critical" if p_value <= 1e-6 else "high",
                            confidence_score=0.97 if p_value <= 1e-6 else 0.88,
                            title="发现小数最后一位高频异常",
                            affected_datasets=[_dataset_ref(dataset)],
                            statistics={
                                "digit": str(digit),
                                "n": n,
                                "observed_count": count,
                                "expected_count": expected,
                                "observed_frequency": round(count / n, 6),
                                "expected_frequency": 0.1,
                                "test": "binomial_tail_uniform_last_digit",
                                "p_value": p_value,
                                "fdr_hypothesis_count": 21,
                                "digit_counts": {str(d): observed[d] for d in range(10)},
                                **digit_summary,
                            },
                            deterministic_basis=(
                                f"{dataset.get('label', '')} 中 {n} 个可分析小数末位里，数字 {digit} "
                                f"出现 {count} 次（期望约 {expected:.1f} 次）。"
                                f"在均匀末位模型下 P(X≥{count}), X~Binomial({n},0.1) "
                                f"= {p_value:.3e}。"
                            ),
                            alternative_explanations=["固定记录精度或四舍五入到特定刻度可能造成末位偏好。"],
                            recommended_human_check=["核对该列是否为连续测量值，以及仪器/软件是否限制末位数字。"],
                        )
                    )

            if count == 0:
                p_zero = 0.9**n
                if p_zero <= 0.01:
                    evidence.append(
                        _make_evidence(
                            evidence_type="last_digit_absence",
                            severity="high" if p_zero <= 1e-4 else "medium",
                            confidence_score=0.86 if p_zero <= 1e-4 else 0.72,
                            title="发现某个小数最后一位完全缺失",
                            affected_datasets=[_dataset_ref(dataset)],
                            statistics={
                                "digit": str(digit),
                                "n": n,
                                "observed_count": 0,
                                "expected_count": expected,
                                "test": "binomial_zero_uniform_last_digit",
                                "p_value": p_zero,
                                "fdr_hypothesis_count": 21,
                                "any_digit_missing_probability": _prob_any_digit_missing(n),
                                "digit_counts": {str(d): observed[d] for d in range(10)},
                                **digit_summary,
                            },
                            deterministic_basis=(
                                f"{dataset.get('label', '')} 中 {n} 个可分析小数末位里，数字 {digit} "
                                f"从未出现。指定数字缺失概率为 0.9^{n} = {p_zero:.3e}；"
                                f"任一数字缺失概率约为 {_prob_any_digit_missing(n):.3e}。"
                            ),
                            alternative_explanations=["样本量有限、数据离散化或仪器输出规则可能造成某些末位缺失。"],
                            recommended_human_check=["检查该列是否适用末位均匀假设。"],
                        )
                    )

        chi2, chi_p = stats.chisquare(f_obs=observed, f_exp=[expected] * 10)
        if float(chi_p) <= 0.01:
            evidence.append(
                _make_evidence(
                    evidence_type="last_digit_distribution_chisquare",
                    severity="high" if float(chi_p) <= 1e-4 else "medium",
                    confidence_score=0.88 if float(chi_p) <= 1e-4 else 0.74,
                    title="小数最后一位整体分布异常",
                    affected_datasets=[_dataset_ref(dataset)],
                    statistics={
                        "n": n,
                        "test": "chi_square_uniform_last_digit",
                        "chi_square": round(float(chi2), 8),
                        "degrees_of_freedom": 9,
                        "p_value": float(chi_p),
                        "fdr_hypothesis_count": 21,
                        "digit_counts": {str(d): observed[d] for d in range(10)},
                        **digit_summary,
                    },
                    deterministic_basis=(
                        f"{dataset.get('label', '')} 的小数末位 0-9 整体分布偏离均匀分布，"
                        f"χ²={float(chi2):.4g}, p={float(chi_p):.3e}。"
                    ),
                    alternative_explanations=["非连续型数据或固定精度记录会破坏末位均匀假设。"],
                    recommended_human_check=["确认该数据是否为连续测量，且没有人为或仪器规定的末位限制。"],
                )
            )
    return evidence


def compare_paper_stats_to_raw_data(
    paper_payload: dict[str, Any] | None, datasets: list[dict[str, Any]]
) -> dict[str, Any]:
    """给出论文提取统计量与原始数据描述统计的候选对齐结果。"""
    paper_payload = paper_payload or {}
    extracted = (paper_payload.get("pre_extracted_stats") or {}).get("means_and_sds") or []
    dataset_stats = []
    for dataset in datasets:
        values = _clean_values(dataset)
        if len(values) < 3:
            continue
        dataset_stats.append({
            "dataset": _dataset_ref(dataset),
            "n": int(len(values)),
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        })

    alignments = []
    for item in extracted[:200]:
        try:
            reported_mean = float(item.get("mean"))
            reported_sd = float(item.get("sd"))
        except (TypeError, ValueError, AttributeError):
            continue
        candidates = []
        for ds_stat in dataset_stats:
            mean_diff = abs(ds_stat["mean"] - reported_mean)
            sd_diff = abs(ds_stat["sd"] - reported_sd)
            mean_scale = max(abs(reported_mean), abs(ds_stat["mean"]), 1.0)
            sd_scale = max(abs(reported_sd), abs(ds_stat["sd"]), 1.0)
            if mean_diff / mean_scale <= 0.02 and sd_diff / sd_scale <= 0.05:
                candidates.append({
                    "dataset": ds_stat["dataset"],
                    "n": ds_stat["n"],
                    "computed_mean": round(ds_stat["mean"], 8),
                    "computed_sd": round(ds_stat["sd"], 8),
                    "mean_relative_diff": round(mean_diff / mean_scale, 8),
                    "sd_relative_diff": round(sd_diff / sd_scale, 8),
                })
        alignments.append({
            "reported": item,
            "candidate_match_count": len(candidates),
            "candidate_matches": candidates[:5],
        })

    return {
        "reported_mean_sd_count": len(extracted),
        "raw_dataset_stats_count": len(dataset_stats),
        "alignments_sample": alignments[:50],
        "note": (
            "该对齐结果仅用于辅助定位论文报告统计量与 XLSX 数据集的可能对应关系；"
            "没有明确标签映射时，不把未匹配项直接作为造假证据。"
        ),
    }


def _apply_fdr(evidence: list[dict[str, Any]]) -> None:
    indexed: list[tuple[int, float, int]] = []
    for idx, item in enumerate(evidence):
        stats_payload = item.get("statistics", {})
        p_value = stats_payload.get("p_value")
        if isinstance(p_value, (int, float)) and math.isfinite(float(p_value)):
            family_size = int(stats_payload.get("fdr_hypothesis_count", 1) or 1)
            indexed.append((idx, float(p_value), max(1, family_size)))
    if not indexed:
        return

    indexed.sort(key=lambda pair: pair[1])
    # Use the declared tested-hypothesis family sizes rather than only the number
    # of emitted findings. This is intentionally conservative for scans such as
    # 10 last-digit overrepresentation tests + 10 absence tests + chi-square.
    m = max(len(indexed), sum(family_size for _, _, family_size in indexed))
    q_values = [1.0] * len(indexed)
    running_min = 1.0
    for rank_from_end, (idx, p_value, _family_size) in enumerate(reversed(indexed), start=1):
        rank = len(indexed) - rank_from_end + 1
        q_value = min(running_min, p_value * m / rank)
        running_min = q_value
        q_values[rank - 1] = q_value

    for (idx, _, _family_size), q_value in zip(indexed, q_values):
        evidence[idx]["statistics"]["q_value"] = min(float(q_value), 1.0)
        evidence[idx]["statistics"]["fdr_total_hypothesis_count"] = m
        _calibrate_probability_evidence(evidence[idx])


def _calibrate_probability_evidence(item: dict[str, Any]) -> None:
    q_value = item.get("statistics", {}).get("q_value")
    if not isinstance(q_value, (int, float)):
        return
    q = float(q_value)
    if q <= 1e-6:
        severity, confidence = "critical", 0.97
    elif q <= 1e-4:
        severity, confidence = "high", 0.92
    elif q <= 1e-3:
        severity, confidence = "high", 0.88
    elif q <= 1e-2:
        severity, confidence = "medium", 0.76
    elif q <= 0.05:
        severity, confidence = "low", 0.62
    else:
        item["statistics"]["fdr_significant"] = False
        if item.get("evidence_type", "").startswith(("last_digit", "decimal_suffix")):
            item["severity"] = "low"
            item["confidence_score"] = min(float(item.get("confidence_score", 0.0)), 0.55)
            item["confidence_label"] = _confidence_label(float(item["confidence_score"]))
        return
    item["statistics"]["fdr_significant"] = True
    if SEVERITY_ORDER[severity] > SEVERITY_ORDER.get(item.get("severity", "low"), 1):
        item["severity"] = severity
    item["confidence_score"] = max(float(item.get("confidence_score", 0.0)), confidence)
    item["confidence_label"] = _confidence_label(float(item["confidence_score"]))


def _assign_ids(evidence: list[dict[str, Any]]) -> None:
    evidence.sort(
        key=lambda item: (
            -SEVERITY_ORDER.get(item.get("severity", "low"), 1),
            -float(item.get("confidence_score", 0.0)),
            item.get("evidence_type", ""),
        )
    )
    for index, item in enumerate(evidence, start=1):
        item["evidence_id"] = f"E-{index:04d}"


def _confidence_summary(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(item.get("severity", "low") for item in evidence)
    max_conf = max((float(item.get("confidence_score", 0.0)) for item in evidence), default=0.0)
    highest = max((SEVERITY_ORDER.get(item.get("severity", "low"), 1) for item in evidence), default=0)
    if highest >= SEVERITY_ORDER["critical"]:
        overall = "serious"
    elif highest >= SEVERITY_ORDER["high"]:
        overall = "high"
    elif highest >= SEVERITY_ORDER["medium"]:
        overall = "medium"
    elif evidence:
        overall = "low"
    else:
        overall = "none_detected"
    return {
        "overall_risk": overall,
        "confidence_score": round(max_conf, 4),
        "confidence_label": _confidence_label(max_conf) if evidence else "low",
        "evidence_count": len(evidence),
        "critical_count": counts.get("critical", 0),
        "high_count": counts.get("high", 0),
        "medium_count": counts.get("medium", 0),
        "low_count": counts.get("low", 0) + counts.get("informational", 0),
    }


def run_raw_data_precheck(
    raw_data_payload: dict[str, Any], paper_payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    """运行本地 XLSX 原始数据的确定性预检。"""
    datasets = raw_data_payload.get("datasets", []) if isinstance(raw_data_payload, dict) else []
    warnings: list[str] = []
    evidence: list[dict[str, Any]] = []

    evidence.extend(detect_constant_numeric_datasets(datasets))
    evidence.extend(detect_arithmetic_sequences(datasets))
    near_evidence, near_warnings = detect_near_duplicate_numeric_datasets(datasets)
    evidence.extend(near_evidence)
    warnings.extend(near_warnings)
    fragment_evidence, fragment_warnings = detect_repeated_numeric_fragments(datasets)
    evidence.extend(fragment_evidence)
    warnings.extend(fragment_warnings)
    numeric_evidence, numeric_warnings = detect_numeric_similarity(datasets)
    evidence.extend(numeric_evidence)
    warnings.extend(numeric_warnings)
    suffix_evidence, suffix_warnings = detect_decimal_suffix_patterns(datasets)
    evidence.extend(suffix_evidence)
    warnings.extend(suffix_warnings)
    evidence.extend(detect_last_digit_anomalies(datasets))

    _apply_fdr(evidence)
    _assign_ids(evidence)

    confidence = _confidence_summary(evidence)
    evidence_for_context = evidence
    if len(evidence) > MAX_EVIDENCE_ITEMS_FOR_CONTEXT:
        omitted = len(evidence) - MAX_EVIDENCE_ITEMS_FOR_CONTEXT
        warnings.append(
            f"确定性证据共 {len(evidence)} 条；为控制报告上下文，仅注入严重程度最高的 "
            f"{MAX_EVIDENCE_ITEMS_FOR_CONTEXT} 条，省略 {omitted} 条。"
        )
        evidence_for_context = evidence[:MAX_EVIDENCE_ITEMS_FOR_CONTEXT]
    allowed_claims = [
        {
            "evidence_id": item["evidence_id"],
            "claim": item["deterministic_basis"],
            "severity": item["severity"],
            "confidence_score": item["confidence_score"],
        }
        for item in evidence_for_context
    ]

    return {
        "status": "success",
        "raw_data_profile": raw_data_payload.get("profile", {}) if isinstance(raw_data_payload, dict) else {},
        "deterministic_findings": evidence_for_context,
        "confidence_summary": confidence,
        "allowed_claims": allowed_claims,
        "paper_raw_data_alignment": compare_paper_stats_to_raw_data(paper_payload, datasets),
        "warnings": warnings,
        "limitations": [
            "本次预检不执行任何图像取证、图像 OCR 或图像相似度分析。",
            "小数末位检验假设连续测量值的小数最后一位近似均匀；若仪器或记录规则限制末位，该假设可能不成立。",
            "Excel 存储值与显示格式可能不一致；预检尽量使用 number_format 重建显示小数位。",
            "没有明确论文表格与 XLSX 数据列映射时，论文-原始数据对齐只作为候选定位，不单独构成造假证据。",
        ],
    }


def format_raw_data_precheck_for_agent(precheck: dict[str, Any]) -> str:
    """将 raw-data 预检结果压缩成人类可读的任务上下文。"""
    summary = precheck.get("confidence_summary", {})
    findings = precheck.get("deterministic_findings", [])
    lines = [
        "### 原始数据确定性预检（系统已执行）",
        "",
        f"- 总体风险：{summary.get('overall_risk', 'unknown')}",
        f"- 最高置信度：{summary.get('confidence_score', 0)}",
        f"- 证据总数：{summary.get('evidence_count', 0)} "
        f"（critical={summary.get('critical_count', 0)}, high={summary.get('high_count', 0)}, "
        f"medium={summary.get('medium_count', 0)}）",
        "",
    ]
    if not findings:
        lines.append("未发现达到阈值的确定性原始数据造假证据。")
    else:
        lines.append("#### 主要证据")
        for item in findings[:20]:
            stats_text = json.dumps(item.get("statistics", {}), ensure_ascii=False, default=str)
            if len(stats_text) > 500:
                stats_text = stats_text[:500] + "..."
            lines.extend([
                f"- **{item.get('evidence_id')} | {item.get('severity')} | "
                f"confidence={item.get('confidence_score')}**",
                f"  - 类型：{item.get('evidence_type')}",
                f"  - 依据：{item.get('deterministic_basis')}",
                f"  - 统计量：{stats_text}",
            ])
    if precheck.get("warnings"):
        lines.append("")
        lines.append("#### 覆盖范围警告")
        for warning in precheck.get("warnings", []):
            lines.append(f"- {warning}")
    return "\n".join(lines)
