"""
Cross-Figure Data Comparison Pipeline — DETERMINISTIC pre-processing that
runs BEFORE the LLM agents to extract bar chart data, run cross-figure
comparison, and execute statistical tests.

THIS IS THE CRITICAL FIX for LLM agent non-determinism:
Previously the pipeline only generated "guidance" and relied on the
LLM agent to actually call bar_chart_extract_values and statistical
tools. The agent (especially weaker models like deepseek-chat) would
SKIP these calls and report "data insufficient."

Now the pipeline DETERMINISTICALLY:
1. Builds page→figure mapping from PDF text
2. Lists all extracted panels with metadata
3. Identifies bar chart panels via CV heuristics
4. ACTUALLY calls extract_bar_chart_values() on each candidate
5. Runs cross_figure_data_compare on all extracted datasets
6. Extracts ALL numeric values from PDF text (means, SDs, percentages, etc.)
7. Runs Benford, anomalous precision (last-digit, arithmetic progression), GRIM
8. Injects COMPLETE results (with actual numbers) into the agent's task context

The agent can no longer skip these steps — the data is already extracted
and the statistical tests are already run.
"""

import json
import logging
import os
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Known high-risk figure pairs ──────────────────────────────────────────
KNOWN_RISK_PAIRS: List[Dict[str, Any]] = [
    {
        "figure_a": "Fig 1E",
        "figure_b": "Fig 4B",
        "shared_groups": ["Control", "COH", "COH+RTA408"],
        "context": "RGC count quantification — two supposedly independent animal experiments",
    },
    {
        "figure_a": "Fig 1F",
        "figure_b": "Fig 4C",
        "shared_groups": ["Control", "COH", "COH+RTA408"],
        "context": "ELISA/protein quantification — same experimental design",
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# Main Pipeline Entry Point
# ═══════════════════════════════════════════════════════════════════════════


def run_cross_figure_pipeline(
    pdf_path: str,
    images_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Deterministic pre-processing: extract bar chart data, run cross-figure
    comparison, and execute statistical tests — ALL before LLM agents start.

    Returns a dict with ALL pre-computed results ready for agent consumption.
    """
    result: Dict[str, Any] = {
        "page_fig_map": {},
        "panels": [],
        "bar_chart_candidates": [],
        "known_risk_pairs_status": [],
        "datasets": [],           # ← NEW: extracted bar chart datasets
        "matches": [],            # ← NEW: cross-figure comparison results
        "has_critical_match": False,  # ← NEW
        "text_numeric_values": {},    # ← NEW: all numbers from PDF text
        "statistical_precheck": {},   # ← NEW: pre-run statistical tests
        "guidance": "",
        "errors": [],
    }

    # ── Step 1: Build page→figure mapping ─────────────────────────────
    page_fig_map = _build_page_figure_map(pdf_path)
    result["page_fig_map"] = {str(k): v for k, v in page_fig_map.items()}

    # ── Step 2: Get all panel images with metadata ────────────────────
    panels = _get_all_panel_images(pdf_path, images_dir, page_fig_map)
    result["panels"] = panels

    # ── Step 3: Identify likely bar chart panels ──────────────────────
    if panels:
        bar_candidates = _identify_bar_chart_candidates(panels)
        result["bar_chart_candidates"] = bar_candidates

        # ── Step 4: Check known risk pairs ────────────────────────────
        risk_status = _check_risk_pairs(panels, bar_candidates, page_fig_map)
        result["known_risk_pairs_status"] = risk_status

        # ── Step 5 (NEW): Actually extract bar chart values ────────────
        logger.info("DETERMINISTIC: Extracting bar chart values from candidates...")
        datasets = _extract_all_bar_chart_datasets(bar_candidates)
        result["datasets"] = datasets
        logger.info(f"DETERMINISTIC: Extracted {len(datasets)} datasets from bar charts")

        # ── Step 6 (NEW): Run cross-figure data comparison ─────────────
        if len(datasets) >= 2:
            logger.info("DETERMINISTIC: Running cross-figure data comparison...")
            matches = _run_cross_figure_comparison(datasets)
            result["matches"] = matches
            result["has_critical_match"] = any(
                m.get("confidence") == "critical" for m in matches
            )
            logger.info(
                f"DETERMINISTIC: Found {len(matches)} matches "
                f"(critical={result['has_critical_match']})"
            )
    else:
        bar_candidates = []
        datasets = []
        matches = []
        result["errors"].append("No panel images found.")

    # ── Step 7 (NEW): Extract ALL numeric values from PDF text ────────
    logger.info("DETERMINISTIC: Extracting numeric values from PDF text...")
    text_values = _extract_numeric_values_from_pdf_text(pdf_path)
    result["text_numeric_values"] = text_values
    logger.info(
        f"DETERMINISTIC: Extracted {text_values.get('total_values', 0)} "
        f"numeric values from PDF text"
    )

    # ── Step 8 (NEW): Run statistical tests on extracted data ─────────
    logger.info("DETERMINISTIC: Running statistical prechecks...")
    statistical_precheck = _run_statistical_prechecks(
        datasets=datasets,
        text_values=text_values,
    )
    result["statistical_precheck"] = statistical_precheck
    flagged = statistical_precheck.get("total_flagged_checks", 0)
    logger.info(f"DETERMINISTIC: Statistical precheck complete — {flagged} checks flagged")

    # ── Step 9: Generate enhanced guidance ────────────────────────────
    result["guidance"] = _generate_enhanced_guidance(
        panels=panels,
        bar_candidates=bar_candidates,
        risk_status=result.get("known_risk_pairs_status", []),
        page_fig_map=page_fig_map,
        datasets=datasets,
        matches=matches,
        text_values=text_values,
        statistical_precheck=statistical_precheck,
    )

    logger.info(
        "cross_figure_pipeline: %d panels, %d bar candidates, "
        "%d datasets extracted, %d matches, %d stats flagged",
        len(panels), len(bar_candidates),
        len(datasets), len(matches),
        flagged,
    )

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Page → Figure Mapping
# ═══════════════════════════════════════════════════════════════════════════


def _build_page_figure_map(pdf_path: str) -> Dict[int, str]:
    """Build mapping from PDF page number to figure label by scanning captions."""
    page_fig_map: Dict[int, str] = {}
    try:
        import fitz

        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            for match in re.finditer(
                r'(?:Fig(?:ure)?[.\s]+)(\d+)[.\s]',
                text[:3000],
                re.IGNORECASE,
            ):
                fig_num = match.group(1)
                pdf_page = page_num + 1
                if pdf_page not in page_fig_map:
                    page_fig_map[pdf_page] = f"Fig {fig_num}"
                    break

        doc.close()
        logger.info(f"_build_page_figure_map: {len(page_fig_map)} pages mapped")
    except Exception as e:
        logger.warning(f"_build_page_figure_map failed: {e}")

    return page_fig_map


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Panel Discovery
# ═══════════════════════════════════════════════════════════════════════════


def _get_all_panel_images(
    pdf_path: str,
    images_dir: Optional[str],
    page_fig_map: Dict[int, str],
) -> List[Dict[str, Any]]:
    """Discover all panel images with metadata."""

    if images_dir:
        panels_dir = Path(images_dir) / "panels"
    else:
        from ..utils.text_extraction import DEFAULT_IMAGE_OUTPUT_DIR
        panels_dir = DEFAULT_IMAGE_OUTPUT_DIR / "panels"

    panels = []

    if panels_dir.exists():
        resolution_groups: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = {}
        for pf in sorted(panels_dir.glob("*.png")):
            info = _parse_panel_filename(pf.name)
            if info:
                key = (info["pdf_page"], info["img_index"], info["resolution"])
                if key not in resolution_groups:
                    resolution_groups[key] = []
                resolution_groups[key].append({
                    "filepath": str(pf.absolute()),
                    "filename": pf.name,
                    **info,
                })

        page_img_resolutions: Dict[Tuple[int, int], List[Tuple[int, List[Dict[str, Any]]]]] = {}
        for (pdf_page, img_idx, res), panel_list in resolution_groups.items():
            key = (pdf_page, img_idx)
            if key not in page_img_resolutions:
                page_img_resolutions[key] = []
            page_img_resolutions[key].append((res, panel_list))

        for (pdf_page, img_idx), res_list in page_img_resolutions.items():
            res_list.sort(key=lambda x: len(x[1]), reverse=True)
            best_panels = res_list[0][1]

            for panel_data in best_panels:
                try:
                    img = Image.open(panel_data["filepath"])
                    panel_data["width"], panel_data["height"] = img.size
                except Exception:
                    panel_data["width"], panel_data["height"] = 0, 0

                fig_label = page_fig_map.get(pdf_page, f"Page{pdf_page}")
                panel_data["figure_label"] = fig_label

                panel_idx = panel_data.get("panel_index", -1)
                if 0 <= panel_idx < 26:
                    panel_data["panel_letter"] = chr(ord("A") + panel_idx)
                else:
                    panel_data["panel_letter"] = str(panel_idx + 1) if panel_idx >= 0 else "?"

                panel_data["full_label"] = f"{fig_label}{panel_data['panel_letter']}"

                panels.append(panel_data)

        panels.sort(key=lambda p: (p.get("pdf_page", 999), p.get("panel_index", 999)))

    if not panels:
        try:
            from ..utils.figure_splitter import extract_all_panels_from_pdf
            all_figures = extract_all_panels_from_pdf(
                pdf_path,
                output_dir=str(panels_dir.parent) if images_dir else None,
            )
            for fig in all_figures:
                fig_panels = fig.get("panels", [])
                for p in fig_panels:
                    panel_path = p.get("filepath", "")
                    if panel_path and os.path.exists(panel_path):
                        info = _parse_panel_filename(os.path.basename(panel_path)) or {}
                        pdf_page = info.get("pdf_page", 0)
                        panel_idx = info.get("panel_index", p.get("panel_index", -1))
                        fig_label = page_fig_map.get(pdf_page, f"Page{pdf_page}")
                        letter = chr(ord("A") + panel_idx) if 0 <= panel_idx < 26 else "?"
                        try:
                            img = Image.open(panel_path)
                            w, h = img.size
                        except Exception:
                            w, h = 0, 0
                        panels.append({
                            "filepath": panel_path,
                            "filename": os.path.basename(panel_path),
                            "pdf_page": pdf_page,
                            "panel_index": panel_idx,
                            "figure_label": fig_label,
                            "panel_letter": letter,
                            "full_label": f"{fig_label}{letter}",
                            "width": w,
                            "height": h,
                        })
        except Exception as e:
            logger.warning(f"Panel extraction from PDF failed: {e}")

    logger.info(f"_get_all_panel_images: {len(panels)} unique panels found")
    return panels


def _parse_panel_filename(filename: str) -> Optional[Dict[str, Any]]:
    """Parse panel filename like 'page4_img1_2_panel_4.png'"""
    match = re.match(r'page(\d+)_img(\d+)_(\d+)_panel_(\d+)\.png', filename, re.IGNORECASE)
    if not match:
        return None
    return {
        "pdf_page": int(match.group(1)),
        "img_index": int(match.group(2)),
        "resolution": int(match.group(3)),
        "panel_index": int(match.group(4)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Bar Chart Candidate Identification
# ═══════════════════════════════════════════════════════════════════════════


def _identify_bar_chart_candidates(panels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identify panels likely to be bar charts based on dimensions and image stats."""
    candidates = []
    for panel in panels:
        w = panel.get("width", 0)
        h = panel.get("height", 0)

        if w < 100 or h < 100:
            continue

        aspect = w / max(h, 1)

        if not (400 <= w <= 2500):
            continue
        if not (120 <= h <= 800):
            continue
        if not (1.2 <= aspect <= 7.0):
            continue

        try:
            img = Image.open(panel["filepath"]).convert("L")
            arr = np.array(img, dtype=np.float64)
            ph, pw = arr.shape

            bottom_half = arr[int(ph * 0.5):, :]
            row_means = np.mean(bottom_half, axis=1)
            if len(row_means) > 1:
                row_diffs = np.abs(np.diff(row_means))
                max_row_diff = np.max(row_diffs)
            else:
                max_row_diff = 0

            col_means = np.mean(arr, axis=0)
            col_std = np.std(col_means)

            overall_std = np.std(arr)
            if overall_std < 20:
                continue

            white_fraction = np.mean(arr > 200)
            if overall_std > 100 and white_fraction < 0.15:
                continue

            panel["_chart_score"] = {
                "aspect": round(aspect, 1),
                "baseline_strength": round(float(max_row_diff), 1),
                "col_std": round(float(col_std), 1),
                "overall_std": round(float(overall_std), 1),
                "white_fraction": round(float(white_fraction), 2),
            }

            candidates.append(panel)

        except Exception as e:
            logger.debug(f"Image analysis failed for {panel.get('filename')}: {e}")
            continue

    logger.info(
        f"_identify_bar_chart_candidates: {len(candidates)}/{len(panels)} "
        "panels identified as potential bar charts"
    )
    return candidates


# ═══════════════════════════════════════════════════════════════════════════
# Step 4: Risk Pair Checking
# ═══════════════════════════════════════════════════════════════════════════


def _check_risk_pairs(
    panels: List[Dict[str, Any]],
    bar_candidates: List[Dict[str, Any]],
    page_fig_map: Dict[int, str],
) -> List[Dict[str, Any]]:
    """Check which known risk pairs have both figures' bar chart panels available."""
    fig_bar_candidates: Dict[str, List[Dict[str, Any]]] = {}
    for c in bar_candidates:
        fig = c.get("figure_label", "")
        if fig not in fig_bar_candidates:
            fig_bar_candidates[fig] = []
        fig_bar_candidates[fig].append(c)

    results = []
    for rp in KNOWN_RISK_PAIRS:
        fig_a = rp["figure_a"]
        fig_b = rp["figure_b"]

        fa_match = re.match(r'Fig\s+(\d+)', fig_a)
        fb_match = re.match(r'Fig\s+(\d+)', fig_b)
        fig_a_base = f"Fig {fa_match.group(1)}" if fa_match else fig_a
        fig_b_base = f"Fig {fb_match.group(1)}" if fb_match else fig_b

        a_bar_candidates = fig_bar_candidates.get(fig_a_base, [])
        b_bar_candidates = fig_bar_candidates.get(fig_b_base, [])

        a_found = len(a_bar_candidates) > 0
        b_found = len(b_bar_candidates) > 0

        entry = {
            "pair": f"{fig_a} vs {fig_b}",
            "figure_a_base": fig_a_base,
            "figure_b_base": fig_b_base,
            "figure_a_has_bar_charts": a_found,
            "figure_b_has_bar_charts": b_found,
            "both_have_bar_charts": a_found and b_found,
            "figure_a_bar_panels": [
                {"full_label": p.get("full_label"), "filepath": p.get("filepath"),
                 "dimensions": f"{p.get('width')}x{p.get('height')}"}
                for p in a_bar_candidates[:5]
            ],
            "figure_b_bar_panels": [
                {"full_label": p.get("full_label"), "filepath": p.get("filepath"),
                 "dimensions": f"{p.get('width')}x{p.get('height')}"}
                for p in b_bar_candidates[:5]
            ],
            "shared_groups_expected": rp["shared_groups"],
            "context": rp["context"],
        }
        results.append(entry)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Step 5 (NEW): Deterministic Bar Chart Value Extraction
# ═══════════════════════════════════════════════════════════════════════════


def _extract_all_bar_chart_datasets(
    bar_candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    ACTUALLY call extract_bar_chart_values() on each bar chart candidate.
    Returns structured datasets ready for cross-figure comparison.
    """
    datasets = []

    for candidate in bar_candidates:
        filepath = candidate.get("filepath", "")
        full_label = candidate.get("full_label", "unknown")

        if not filepath or not os.path.exists(filepath):
            continue

        try:
            from ..utils.chart_ocr import extract_bar_chart_values

            extraction = extract_bar_chart_values(filepath)

            if extraction.get("error"):
                logger.debug(f"Bar chart extraction failed for {full_label}: {extraction['error']}")
                datasets.append({
                    "label": full_label,
                    "filepath": filepath,
                    "extraction_error": extraction.get("error"),
                    "groups": [],
                    "values": [],
                })
                continue

            if extraction.get("warning"):
                logger.debug(f"Bar chart extraction warning for {full_label}: {extraction['warning']}")
                datasets.append({
                    "label": full_label,
                    "filepath": filepath,
                    "extraction_warning": extraction.get("warning"),
                    "groups": [],
                    "values": [],
                })
                continue

            groups = extraction.get("groups", [])
            if not groups:
                datasets.append({
                    "label": full_label,
                    "filepath": filepath,
                    "extraction_warning": "No groups detected in bar chart",
                    "groups": [],
                    "values": [],
                })
                continue

            # Build dataset in cross_figure_data_compare format
            group_labels = []
            values = []

            for g in groups:
                label = g.get("group_label", "?")
                g_values = g.get("values", [])
                group_labels.append(label)
                # For single-bar groups, take the first value
                if len(g_values) == 1:
                    values.append(g_values[0])
                elif len(g_values) > 1:
                    # For multi-bar groups, take the mean of bars
                    values.append(round(float(np.mean(g_values)), 4))

            # Also collect all individual bar values as an ordered series
            ordered_series = []
            for g in groups:
                g_values = g.get("values", [])
                ordered_series.extend(g_values)

            datasets.append({
                "label": full_label,
                "filepath": filepath,
                "figure_label": candidate.get("figure_label", ""),
                "group_labels": group_labels,
                "values": values,
                "ordered_series": ordered_series,
                "bar_count": extraction.get("bars_detected", 0),
                "confidence": extraction.get("confidence", "low"),
                "y_axis_range": extraction.get("y_axis", {}).get("range"),
            })

        except Exception as e:
            logger.warning(f"Bar chart extraction exception for {full_label}: {e}")
            datasets.append({
                "label": full_label,
                "filepath": filepath,
                "extraction_error": str(e),
                "groups": [],
                "values": [],
            })

    # Filter out datasets with no values
    valid_datasets = [d for d in datasets if len(d.get("values", [])) > 0]
    logger.info(
        f"_extract_all_bar_chart_datasets: {len(valid_datasets)}/{len(datasets)} "
        "datasets have valid extracted values"
    )

    return datasets


# ═══════════════════════════════════════════════════════════════════════════
# Step 6 (NEW): Deterministic Cross-Figure Data Comparison
# ═══════════════════════════════════════════════════════════════════════════


def _run_cross_figure_comparison(
    datasets: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Run cross_figure_data_compare on ALL extracted datasets.
    This is the critical fraud detection step — same values in different
    figures = definitive fabrication evidence.

    We implement the comparison logic directly here (rather than calling
    the CrewAI tool) for deterministic execution.
    """
    matches = []

    # Filter to datasets with actual values
    valid = [d for d in datasets if len(d.get("values", [])) >= 2]

    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            d_a = valid[i]
            d_b = valid[j]
            vals_a = np.array(d_a.get("values", []), dtype=np.float64)
            vals_b = np.array(d_b.get("values", []), dtype=np.float64)

            if len(vals_a) == 0 or len(vals_b) == 0:
                continue

            # Check 1: Exact match
            if len(vals_a) == len(vals_b):
                if np.allclose(vals_a, vals_b, rtol=0, atol=0):
                    matches.append({
                        "type": "exact_match",
                        "dataset_a": d_a["label"],
                        "dataset_b": d_b["label"],
                        "shared_values": vals_a.tolist(),
                        "group_labels_a": d_a.get("group_labels", []),
                        "group_labels_b": d_b.get("group_labels", []),
                        "confidence": "critical",
                        "interpretation": (
                            f"🔴 CRITICAL: {d_a['label']} 和 {d_b['label']} "
                            f"数值完全相同！独立生物实验不可能产生完全相同的"
                            f"结果。这是数据造假的铁证。"
                        ),
                    })
                    continue

            # Check 2: Near match (within 1% tolerance)
            if len(vals_a) == len(vals_b):
                denominator = np.maximum(np.abs(vals_a), np.abs(vals_b))
                # Avoid divide-by-zero: where both values are 0, diff is 0
                denominator = np.where(denominator < 1e-10, 1.0, denominator)
                rel_diff = np.abs(vals_a - vals_b) / denominator
                rel_diff = np.nan_to_num(rel_diff, nan=0.0)
                if np.all(rel_diff <= 0.01):
                    matches.append({
                        "type": "near_match",
                        "dataset_a": d_a["label"],
                        "dataset_b": d_b["label"],
                        "values_a": vals_a.tolist(),
                        "values_b": vals_b.tolist(),
                        "max_relative_difference": round(float(np.max(rel_diff)), 6),
                        "confidence": "high",
                        "interpretation": (
                            f"🟠 HIGH: {d_a['label']} 和 {d_b['label']} "
                            f"数值几乎相同（最大差异 {np.max(rel_diff):.4%}）。"
                            f"真实独立实验应表现出自然变异。"
                        ),
                    })

            # Check 3: Shared group exact match
            groups_a = d_a.get("group_labels", [])
            groups_b = d_b.get("group_labels", [])
            common_groups = set(groups_a) & set(groups_b)

            if common_groups and len(common_groups) >= 2:
                idx_a = [groups_a.index(g) for g in common_groups if g in groups_a]
                idx_b = [groups_b.index(g) for g in common_groups if g in groups_b]
                if len(idx_a) == len(idx_b) and len(idx_a) >= 2:
                    shared_a = np.array([vals_a[k] for k in idx_a])
                    shared_b = np.array([vals_b[k] for k in idx_b])
                    if np.allclose(shared_a, shared_b, rtol=0, atol=0):
                        matches.append({
                            "type": "shared_group_exact_match",
                            "dataset_a": d_a["label"],
                            "dataset_b": d_b["label"],
                            "common_groups": sorted(common_groups),
                            "shared_values": shared_a.tolist(),
                            "confidence": "critical",
                            "interpretation": (
                                f"🔴 CRITICAL: 组 {sorted(common_groups)} 在 "
                                f"{d_a['label']} 和 {d_b['label']} 中数值完全相同。"
                                f"这与 RTA408 撤稿论文（Fig 1E vs Fig 4B）的造假模式"
                                f"完全一致。数据造假的铁证。"
                            ),
                        })

            # Check 4: Ratio match (one = constant × other)
            if len(vals_a) == len(vals_b) and len(vals_a) >= 2:
                # Compute ratios only where both values are non-zero
                mask = (np.abs(vals_b) > 1e-10)
                ratios = np.full_like(vals_a, np.nan, dtype=np.float64)
                with np.errstate(divide='ignore', invalid='ignore'):
                    np.divide(vals_a, vals_b, out=ratios, where=mask)
                ratios = ratios[mask]  # keep only valid ratios
                if len(ratios) >= 2:
                    ratio_mean = np.mean(ratios)
                    ratio_std = np.std(ratios)
                    if ratio_std < 0.01 * abs(ratio_mean) and abs(ratio_mean - 1.0) > 0.01:
                        matches.append({
                            "type": "ratio_match",
                            "dataset_a": d_a["label"],
                            "dataset_b": d_b["label"],
                            "ratio": round(float(ratio_mean), 4),
                            "ratio_std": round(float(ratio_std), 6),
                            "confidence": "medium",
                            "interpretation": (
                                f"🟡 MEDIUM: {d_a['label']} 的数值是 "
                                f"{d_b['label']} 的 {ratio_mean:.2f} 倍"
                                f"（标准差极小={ratio_std:.6f}）。"
                                f"可能暗示系统性缩放而非真实数据。"
                            ),
                        })

    return matches


# ═══════════════════════════════════════════════════════════════════════════
# Step 7 (NEW): Extract ALL Numeric Values from PDF Text
# ═══════════════════════════════════════════════════════════════════════════


def _extract_numeric_values_from_pdf_text(pdf_path: str) -> Dict[str, Any]:
    """
    Extract ALL numeric values from the PDF text layer.
    This catches means, SDs, percentages, p-values, and other numbers
    that appear in the paper text but might be missed by chart OCR.

    Returns structured data ready for statistical analysis.
    """
    result = {
        "all_numbers": [],
        "means": [],
        "sds": [],
        "mean_sd_pairs": [],
        "p_values": [],
        "percentages": [],
        "sample_sizes": [],
        "total_values": 0,
        "extraction_method": "regex",
    }

    try:
        import fitz
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n"
        doc.close()

        if not full_text.strip():
            result["error"] = "No text extracted from PDF"
            return result

        # ── Extract all floating-point numbers ──
        all_numbers = []
        for match in re.finditer(r'(?<![a-zA-Z0-9])(\d+\.?\d*)(?![a-zA-Z])', full_text):
            try:
                val = float(match.group(1))
                # Filter implausible values
                if 0 < val < 1_000_000:
                    all_numbers.append(val)
            except ValueError:
                continue

        result["all_numbers"] = all_numbers
        result["total_values"] = len(all_numbers)

        # ── Extract mean±SD pairs (with noise filters) ──
        # REAL mean±SD patterns: "12.34 ± 2.56" or "12.34±2.56"
        # FAKE patterns to exclude: DOIs ("41565-025-02082-0"), dates, fractions, IDs
        mean_sd_pairs = []
        means = []
        sds = []

        for match in re.finditer(
            r'(\d+\.?\d*)\s*[±+/-]+\s*(\d+\.?\d*)',
            full_text,
        ):
            try:
                m = float(match.group(1))
                s = float(match.group(2))
                raw = match.group(0).strip()

                # Skip if either value is 0
                if m <= 0 or s <= 0:
                    continue

                # Skip implausibly large values (DOIs, IDs, years)
                if m > 10000 or s > 10000:
                    continue

                # Skip DOI-like patterns: sequences with multiple dashes/numbers
                # e.g., "41565-025-02082-0" contains multiple number-number segments
                if raw.count('-') >= 3:
                    continue

                # Skip date-like patterns: year/ID style
                # e.g., "2020/133", "2023/008" — these are clinical trial IDs or dates
                if '/' in raw:
                    # If the first number looks like a year (1900-2099) and it's a fraction format
                    if 1900 <= m <= 2099 and '/' in raw:
                        continue

                # Skip if it looks like a ratio/fraction where both numbers are >100
                # e.g., "141/200" looks like a fraction, not mean±SD
                if m > 100 and s > 100 and '/' in raw:
                    continue

                # SD should be smaller than the mean for real mean±SD
                # (allow some tolerance: SD can be up to 2x the mean in some cases)
                if s > m * 3:
                    continue

                mean_sd_pairs.append({"mean": m, "sd": s, "raw": raw})
                means.append(m)
                sds.append(s)
            except ValueError:
                continue

        result["mean_sd_pairs"] = mean_sd_pairs
        result["means"] = means
        result["sds"] = sds

        # ── Extract p-values ──
        p_values = []
        for match in re.finditer(
            r'[pP]\s*[<>=]+\s*(\d+\.\d+)',
            full_text,
        ):
            try:
                val = float(match.group(1))
                if 0 < val <= 1:
                    p_values.append(val)
            except ValueError:
                continue

        result["p_values"] = p_values

        # ── Extract numeric sequences from tables and text ──
        # Look for sequences of 3+ numbers in the same sentence/table row
        # These are key inputs for arithmetic progression detection
        ordered_sequences = _extract_ordered_sequences_from_text(full_text)
        result["ordered_sequences_from_text"] = ordered_sequences

        # ── Extract percentages ──
        percentages = []
        for match in re.finditer(r'(\d+\.?\d*)\s*%', full_text):
            try:
                val = float(match.group(1))
                if 0 < val <= 100:
                    percentages.append(val)
            except ValueError:
                continue

        result["percentages"] = percentages

        # ── Extract sample sizes ──
        sample_sizes = []
        for match in re.finditer(r'[nN]\s*[=:]\s*(\d+)', full_text):
            try:
                n = int(match.group(1))
                if 2 <= n <= 1000:
                    sample_sizes.append(n)
            except ValueError:
                continue

        result["sample_sizes"] = sample_sizes

        logger.info(
            f"_extract_numeric_values_from_pdf_text: {len(all_numbers)} numbers, "
            f"{len(mean_sd_pairs)} mean±SD pairs, {len(p_values)} p-values"
        )

    except Exception as e:
        logger.warning(f"PDF text numeric extraction failed: {e}")
        result["error"] = str(e)

    return result


def _extract_ordered_sequences_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extract ordered sequences of numbers from text that may represent
    bar chart data, dose-response series, or time-series measurements.

    Looks for:
    1. Table rows with 3+ numbers (e.g., "Control  1200  800  400")
    2. Comma-separated number lists (e.g., "values were 45.2, 38.1, 31.0, 23.8")
    3. Sentences describing sequential changes (e.g., "decreased from 1200 to 800 to 400")
    4. Parenthetical number groups (e.g., "(1200 ± 50, 800 ± 35, 400 ± 30)")
    """
    sequences = []

    # Pattern 1: Table rows with numbers separated by whitespace
    # Find lines with 3+ numbers in sequence
    for line in text.split('\n'):
        # Extract all numbers from the line
        nums = re.findall(r'(?<!\w)(\d+\.?\d*)(?!\w)', line)
        if len(nums) >= 3:
            try:
                vals = [float(n) for n in nums]
                # Filter: values should be in a reasonable range and not all identical
                if len(vals) >= 3 and len(set(vals)) > 1:
                    # Check if values are monotonically changing (suggesting ordered data)
                    sequences.append({
                        "source": "table_row",
                        "values": vals,
                        "raw_text": line.strip()[:200],
                    })
            except ValueError:
                continue

    # Pattern 2: Comma-separated or "and"-separated number lists in sentences
    # e.g., "the values were 45.2, 38.1, 31.0, and 23.8"
    for match in re.finditer(
        r'(\d+\.?\d*)\s*[,，]\s*(\d+\.?\d*)\s*[,，]\s*(\d+\.?\d*(?:\s*[,，]\s*\d+\.?\d*)*)',
        text,
    ):
        full_match = match.group(0)
        nums = re.findall(r'\d+\.?\d*', full_match)
        if len(nums) >= 3:
            try:
                vals = [float(n) for n in nums]
                if len(vals) >= 3 and len(set(vals)) > 1:
                    sequences.append({
                        "source": "comma_list",
                        "values": vals,
                        "raw_text": full_match[:200],
                    })
            except ValueError:
                continue

    # Pattern 3: Numbers inside parentheses suggesting grouped data
    # e.g., "(1200 ± 50, n=5), (800 ± 35, n=5), (400 ± 30, n=5)"
    for match in re.finditer(
        r'\(\s*(\d+\.?\d*)\s*[±+-]',
        text,
    ):
        # Find consecutive parenthetical mean±SD groups
        start = match.start()
        # Look ahead for more similar patterns
        segment = text[start:start + 500]
        nums = re.findall(r'\(\s*(\d+\.?\d*)\s*[±+-]', segment)
        if len(nums) >= 3:
            try:
                vals = [float(n) for n in nums]
                if len(set(vals)) > 1:
                    sequences.append({
                        "source": "parenthetical_groups",
                        "values": vals,
                        "raw_text": segment[:200],
                    })
            except ValueError:
                continue

    # Pattern 4: "from X to Y to Z" or "X, Y, and Z respectively"
    for pattern in [
        r'(\d+\.?\d*)\s*(?:to|→|->)\s*(\d+\.?\d*)\s*(?:to|→|->)\s*(\d+\.?\d*)',
        r'(\d+\.?\d*),\s*(\d+\.?\d*),\s*(?:and|&)\s*(\d+\.?\d*)\s+respectively',
    ]:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                vals = [float(match.group(i)) for i in range(1, len(match.groups()) + 1)]
                if len(vals) >= 3 and len(set(vals)) > 1:
                    sequences.append({
                        "source": "textual_sequence",
                        "values": vals,
                        "raw_text": match.group(0)[:200],
                    })
            except (ValueError, IndexError):
                continue

    # Filter out sequences that look like metadata, not experimental data
    filtered_sequences = []
    for seq in sequences:
        vals = seq.get("values", [])
        if len(vals) < 3:
            continue

        # Skip consecutive integers starting from 1 (page/figure numbers)
        # e.g., [1,2,3,4,5,6], [4,5,6], [7,8,9,10,11,12]
        if all(v == int(v) for v in vals):
            int_vals = [int(v) for v in vals]
            # Check if all diffs are exactly 1
            if all(int_vals[i+1] - int_vals[i] == 1 for i in range(len(int_vals)-1)):
                # Consecutive integers — likely page/figure/section numbers, not data
                # Only keep if they look like real measurements (e.g., counts)
                # Counts would typically start from 0 or a non-trivial number
                if int_vals[0] <= 1:
                    continue  # Skip [1,2,3,...] type sequences
                # Also skip [N, N+1, N+2] where N<20 (likely page numbers)
                if int_vals[0] < 20:
                    continue

        # Skip sequences where ALL values are integers and look like plain numbering
        if all(v == int(v) and 1 <= v <= 30 for v in vals):
            continue

        # Skip sequences with only 0 and 1 values (binary data, not meaningful)
        if all(v in (0, 0.0, 1, 1.0) for v in vals):
            continue

        filtered_sequences.append(seq)

    # Deduplicate: remove sequences that are subsets of others
    unique_sequences = []
    seen_value_sets = set()
    for seq in sorted(filtered_sequences, key=lambda s: -len(s["values"])):
        value_key = tuple(round(v, 4) for v in seq["values"])
        if value_key not in seen_value_sets:
            seen_value_sets.add(value_key)
            unique_sequences.append(seq)

    logger.info(
        f"_extract_ordered_sequences_from_text: {len(unique_sequences)} "
        f"unique sequences found"
    )
    return unique_sequences


# ═══════════════════════════════════════════════════════════════════════════
# Step 8 (NEW): Deterministic Statistical Prechecks
# ═══════════════════════════════════════════════════════════════════════════


def _run_statistical_prechecks(
    datasets: List[Dict[str, Any]],
    text_values: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run ALL statistical tests deterministically on the extracted data.
    This ensures tests ALWAYS run, regardless of what the LLM agent decides.

    Tests run:
    1. Benford's Law on all extracted numbers
    2. P-value distribution / caliper test
    3. GRIM on mean-SD-sample_size triples
    4. Anomalous precision (CV, last-digit, arithmetic progression, repeats)
    """
    precheck = {
        "benford": {},
        "p_value_analysis": {},
        "grim": {},
        "anomalous_precision": {},
        "total_flagged_checks": 0,
        "checks_run": [],
    }

    # ── Collect all numbers for analysis ──
    all_numbers = list(text_values.get("all_numbers", []))

    # Also collect numbers from bar chart datasets
    for ds in datasets:
        all_numbers.extend(ds.get("values", []))
        all_numbers.extend(ds.get("ordered_series", []))

    all_numbers = [n for n in all_numbers if n > 0 and not np.isnan(n) and not np.isinf(n)]

    means = list(text_values.get("means", []))
    sds = list(text_values.get("sds", []))
    p_values = list(text_values.get("p_values", []))
    percentages = list(text_values.get("percentages", []))
    mean_sd_pairs = list(text_values.get("mean_sd_pairs", []))

    # ── 1. Benford's Law ──
    if len(all_numbers) >= 30:
        precheck["checks_run"].append("benford")
        benford_result = _run_benford_test(all_numbers)
        precheck["benford"] = benford_result
        if benford_result.get("flagged"):
            precheck["total_flagged_checks"] += 1
    else:
        precheck["benford"] = {
            "note": f"样本量不足 (n={len(all_numbers)}, 需要≥30)",
            "flagged": False,
        }

    # ── 2. P-value distribution ──
    if len(p_values) >= 10:
        precheck["checks_run"].append("p_value_analysis")
        pval_result = _run_pvalue_analysis(p_values)
        precheck["p_value_analysis"] = pval_result
        if pval_result.get("flagged"):
            precheck["total_flagged_checks"] += 1
    else:
        precheck["p_value_analysis"] = {
            "note": f"精确p值不足 (n={len(p_values)}, 需要≥10个精确p值)",
            "p_value_count": len(p_values),
            "flagged": False,
            "warning": (
                "论文可能仅以阈值形式报告p值（如*P<0.05），"
                "这是透明度不足但非造假证据。"
            ),
        }

    # ── 3. GRIM test ──
    grim_pairs = []
    for pair in mean_sd_pairs:
        m = pair.get("mean", 0)
        # Try to find corresponding sample size from text
        n = _find_sample_size_for_mean(m, text_values)
        if n:
            grim_pairs.append({"mean": m, "n": n})

    if len(grim_pairs) >= 1:
        precheck["checks_run"].append("grim")
        grim_result = _run_grim_test(grim_pairs)
        precheck["grim"] = grim_result
        if grim_result.get("flagged"):
            precheck["total_flagged_checks"] += 1
    else:
        precheck["grim"] = {
            "note": "无法找到(均值, 样本量)配对",
            "pairs_found": len(mean_sd_pairs),
            "flagged": False,
        }

    # ── 4. Anomalous Precision ──
    # 4a. CV analysis
    cv_flags = []
    if len(means) > 0 and len(sds) > 0 and len(means) == len(sds):
        for i, (m, sd) in enumerate(zip(means, sds)):
            if m != 0 and sd >= 0:
                cv = abs(sd / m)
                if cv < 0.01:
                    cv_flags.append({"index": i, "mean": m, "sd": sd, "cv": round(cv, 4),
                                     "severity": "critical"})
                elif cv < 0.03:
                    cv_flags.append({"index": i, "mean": m, "sd": sd, "cv": round(cv, 4),
                                     "severity": "warning"})

    # 4b. Last-digit analysis
    last_digit_result = _analyze_last_digits(all_numbers)

    # 4c. Near-arithmetic progression detection
    ordered_series_list = []
    for ds in datasets:
        series = ds.get("ordered_series", [])
        if len(series) >= 3:
            ordered_series_list.append({
                "label": ds.get("label", "unknown"),
                "values": series,
            })

    # Also add text-extracted ordered sequences
    text_sequences = text_values.get("ordered_sequences_from_text", [])
    for si, seq in enumerate(text_sequences):
        vals = seq.get("values", [])
        if len(vals) >= 3:
            ordered_series_list.append({
                "label": f"TextSeq_{si}_{seq.get('source', 'unknown')}",
                "values": vals,
            })

    progression_result = _detect_arithmetic_progressions(ordered_series_list)

    # 4d. High-frequency repeats
    repeat_result = _detect_high_frequency_repeats(all_numbers)

    anomalous_result = {
        "cv_analysis": {
            "pairs_tested": len(means),
            "cv_critical_count": sum(1 for f in cv_flags if f["severity"] == "critical"),
            "cv_warning_count": sum(1 for f in cv_flags if f["severity"] == "warning"),
            "flagged_pairs": cv_flags,
        },
        "last_digit_analysis": last_digit_result,
        "near_arithmetic_progression": progression_result,
        "high_frequency_repeats": repeat_result,
    }

    total_anomalous_flags = (
        (1 if cv_flags else 0) +
        (1 if last_digit_result.get("flagged") else 0) +
        (1 if progression_result.get("flagged") else 0) +
        (1 if repeat_result.get("flagged") else 0)
    )
    anomalous_result["flagged"] = total_anomalous_flags > 0
    anomalous_result["total_flagged_subchecks"] = total_anomalous_flags

    precheck["anomalous_precision"] = anomalous_result
    if total_anomalous_flags > 0:
        precheck["total_flagged_checks"] += total_anomalous_flags
    precheck["checks_run"].append("anomalous_precision")

    return precheck


# ── Statistical test helpers ───────────────────────────────────────────────


def _run_benford_test(values: List[float]) -> Dict[str, Any]:
    """Run Benford's Law test on a list of values."""
    from scipy import stats as scipy_stats

    first_digits = []
    for x in values:
        if x == 0 or np.isnan(x) or np.isinf(x):
            continue
        x_abs = abs(x)
        if x_abs < 1.0 and x_abs > 0:
            digit_str = f"{x_abs:.10f}".lstrip("0.")
            first_nonzero = next((c for c in digit_str if c != "0"), None)
            if first_nonzero:
                first_digits.append(int(first_nonzero))
        else:
            digit_str = str(int(x_abs))
            if digit_str:
                first_digits.append(int(digit_str[0]))

    if len(first_digits) < 30:
        return {"flagged": False, "note": f"Insufficient data (n={len(first_digits)})"}

    benford_probs = np.array([np.log10(1 + 1 / d) for d in range(1, 10)])
    observed_counts = np.array([first_digits.count(d) for d in range(1, 10)])

    chi2, p_value = scipy_stats.chisquare(
        f_obs=observed_counts,
        f_exp=benford_probs * len(first_digits),
    )

    observed_props = observed_counts / observed_counts.sum()
    mad = float(np.mean(np.abs(observed_props - benford_probs)))

    mad_assessment = (
        "close conformity" if mad < 0.006 else
        "acceptable conformity" if mad < 0.012 else
        "marginally acceptable" if mad < 0.015 else
        "non-conformity — suspicious"
    )

    flagged = bool(p_value < 0.05 or mad > 0.015)

    return {
        "test": "Benford's Law",
        "sample_size": int(len(first_digits)),
        "chi_squared": round(float(chi2), 4),
        "degrees_of_freedom": 8,
        "p_value": round(float(p_value), 6),
        "mean_absolute_deviation": round(float(mad), 6),
        "mad_assessment": mad_assessment,
        "observed_distribution": {
            str(d): int(c) for d, c in zip(range(1, 10), observed_counts)
        },
        "expected_benford_distribution": {
            str(d): round(float(p), 4) for d, p in zip(range(1, 10), benford_probs)
        },
        "flagged": flagged,
        "interpretation": (
            f"数据与本福特定律{'显著偏离' if flagged else '一致'} "
            f"(p={float(p_value):.4f}, MAD={mad:.4f})"
        ),
    }


def _run_pvalue_analysis(pvalues: List[float]) -> Dict[str, Any]:
    """Run p-value distribution analysis."""
    from scipy import stats as scipy_stats

    pvalues = np.array([p for p in pvalues if 0 < p <= 1], dtype=np.float64)

    if len(pvalues) < 10:
        return {"flagged": False, "note": f"Insufficient p-values (n={len(pvalues)})"}

    # Caliper test
    just_below_05 = int(np.sum((pvalues > 0.045) & (pvalues <= 0.05)))
    just_above_05 = int(np.sum((pvalues > 0.05) & (pvalues <= 0.055)))
    caliper_ratio = just_below_05 / max(just_above_05, 1) if just_above_05 > 0 else float('inf')

    # Uniformity test
    hist, bin_edges = np.histogram(pvalues, bins=10, range=(0, 1))
    expected_per_bin = len(pvalues) / 10
    chi2_uniform, p_uniform = scipy_stats.chisquare(
        f_obs=hist, f_exp=[expected_per_bin] * 10,
    )

    findings = []
    if caliper_ratio > 2.0:
        findings.append({
            "type": "caliper_test",
            "detail": (
                f"p=0.05处不连续：{just_below_05}个值在(0.045,0.05]，"
                f"{just_above_05}个在(0.05,0.055]。比率={caliper_ratio:.2f}"
            ),
        })
    if p_uniform < 0.05:
        findings.append({
            "type": "uniformity_deviation",
            "detail": f"p值分布偏离均匀分布 (χ²={chi2_uniform:.2f}, p={p_uniform:.4f})",
        })

    flagged = bool(len(findings) > 0)

    return {
        "test": "P-value Distribution Analysis",
        "p_value_count": int(len(pvalues)),
        "caliper_test": {
            "p_values_in_0.045_to_0.05": int(just_below_05),
            "p_values_in_0.05_to_0.055": int(just_above_05),
            "caliper_ratio": round(float(caliper_ratio), 4) if caliper_ratio != float('inf') else 999.0,
            "flagged": caliper_ratio > 2.0,
        },
        "uniformity_test": {
            "chi_squared": round(float(chi2_uniform), 4),
            "p_value": round(float(p_uniform), 6),
            "flagged": p_uniform < 0.05,
        },
        "findings": findings,
        "flagged": flagged,
        "interpretation": (
            f"p值分布{'存在异常' if flagged else '正常'}："
            f"{'卡尺检验异常' if caliper_ratio > 2.0 else ''}"
            f"{'；均匀性检验异常' if p_uniform < 0.05 else ''}"
        ),
    }


def _run_grim_test(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run GRIM test on (mean, n) pairs."""
    results = []
    inconsistent_count = 0

    for i, pair in enumerate(pairs):
        mean = pair.get("mean")
        n = pair.get("n")

        if mean is None or n is None:
            continue
        if not isinstance(n, int) or n <= 0:
            continue

        product = mean * n
        nearest_integer = round(product)
        deviation = abs(product - nearest_integer)
        is_consistent = deviation < 0.01

        if not is_consistent:
            inconsistent_count += 1

        results.append({
            "pair_index": i,
            "mean": mean,
            "n": n,
            "product": round(product, 4),
            "nearest_integer": nearest_integer,
            "deviation": round(deviation, 6),
            "consistent": is_consistent,
            "detail": (
                f"{'✅' if is_consistent else '🔴'} "
                f"{mean} × {n} = {product:.4f} ≈ {nearest_integer}"
                + ("" if is_consistent else
                   f" — 数学上不可能！偏差={deviation:.4f}")
            ),
        })

    flagged = inconsistent_count > 0

    return {
        "test": "GRIM",
        "pairs_tested": len(results),
        "consistent_pairs": len(results) - inconsistent_count,
        "inconsistent_pairs": inconsistent_count,
        "flagged": flagged,
        "results": results,
        "interpretation": (
            f"🔴 发现 {inconsistent_count} 个数学上不可能的均值！"
            if flagged
            else "所有均值在数学上一致。"
        ),
    }


def _analyze_last_digits(values: List[float]) -> Dict[str, Any]:
    """Analyze last-digit distribution for fabrication signatures."""
    if len(values) < 10:
        return {"flagged": False, "note": f"数据不足 (n={len(values)}, 需要≥10)", "n_analyzed": len(values)}

    last_digits = []
    for v in values:
        v_str = f"{abs(v):.10g}"
        if "." in v_str:
            frac = v_str.split(".")[1]
            if frac:
                last_digits.append(int(frac[-1]))
        elif v_str[-1].isdigit():
            last_digits.append(int(v_str[-1]))

    if len(last_digits) < 10:
        return {"flagged": False, "note": "有效末位数字不足", "n_analyzed": len(last_digits)}

    digit_counts = Counter(last_digits)
    total = len(last_digits)
    digit_freqs = {
        str(d): round(digit_counts.get(d, 0) / total, 3)
        for d in range(10)
    }

    freq_0 = digit_freqs.get("0", 0)
    freq_5 = digit_freqs.get("5", 0)
    zero_five = freq_0 + freq_5

    flagged = False
    flags = []

    # 末位=5 独立检测（降低阈值：>12%即可疑，>18%即严重）
    if freq_5 > 0.18:
        flags.append(f"🔴 末位=5 占比 {freq_5:.1%}（期望10%），是期望值的 {freq_5/0.10:.1f} 倍 — 严重偏好")
        flagged = True
    elif freq_5 > 0.12:
        flags.append(f"🟡 末位=5 占比 {freq_5:.1%}（期望10%），轻度偏高")
        flagged = True

    # 末位=0 独立检测（降低阈值）
    if freq_0 > 0.18:
        flags.append(f"🔴 末位=0 占比 {freq_0:.1%}（期望10%），是期望值的 {freq_0/0.10:.1f} 倍 — 严重偏好")
        flagged = True
    elif freq_0 > 0.12:
        flags.append(f"🟡 末位=0 占比 {freq_0:.1%}（期望10%），轻度偏高")
        flagged = True

    # 组合检测（降低阈值：>25%即可疑）
    if zero_five > 0.35:
        flags.append(f"🔴 末位0+5合计占比 {zero_five:.1%}（期望20%）— 严重偏好")
        flagged = True
    elif zero_five > 0.25:
        flags.append(f"🟡 末位0+5合计占比 {zero_five:.1%}（期望20%）— 轻度偏高")
        flagged = True

    # 其他偏好数字（降低阈值）
    for d in range(10):
        f = digit_freqs.get(str(d), 0)
        if f > 0.15 and d not in (0, 5):
            flags.append(f"🟡 末位={d} 占比 {f:.1%}（期望10%）")
            flagged = True

    return {
        "flagged": flagged,
        "n_analyzed": total,
        "digit_frequencies": digit_freqs,
        "last_digit_0_frequency": round(freq_0, 3),
        "last_digit_5_frequency": round(freq_5, 3),
        "zero_five_combined": round(zero_five, 3),
        "flags": flags,
        "interpretation": (
            "；".join(flags) if flags else "末位数字分布正常，未发现人为偏好迹象。"
        ),
    }


def _detect_arithmetic_progressions(
    ordered_series_list: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Detect near-arithmetic progressions in ordered value series.
    This is the #1 signature of fabricated data — humans tend to
    create data with suspiciously constant differences between groups.
    """
    if not ordered_series_list:
        return {
            "flagged": False,
            "series_tested": 0,
            "flagged_series": 0,
            "details": [],
            "note": "无可用的有序序列数据",
        }

    progression_flags = []
    for si, series_info in enumerate(ordered_series_list):
        label = series_info.get("label", f"series_{si}")
        values = series_info.get("values", [])

        if len(values) < 3:
            continue

        vals = np.array(values, dtype=np.float64)
        diffs = np.diff(vals)
        diff_mean = float(np.mean(np.abs(diffs)))
        diff_std = float(np.std(diffs))

        if diff_mean < 1e-10:
            continue

        cv_diff = float(diff_std / diff_mean) if diff_mean > 1e-10 else 0.0

        severity = None
        if cv_diff < 0.08:
            severity = "critical"
        elif cv_diff < 0.15:
            severity = "high"
        elif cv_diff < 0.25:
            severity = "medium"

        if severity:
            n = len(vals)
            perfect_step = (vals[-1] - vals[0]) / (n - 1)
            perfect_vals = np.array([vals[0] + perfect_step * k for k in range(n)])
            ss_res = float(np.sum((vals - perfect_vals) ** 2))
            ss_tot = float(np.sum((vals - np.mean(vals)) ** 2))
            r_squared = float(1 - ss_res / ss_tot) if ss_tot > 1e-10 else 1.0

            progression_flags.append({
                "label": label,
                "values": [round(float(v), 4) for v in vals],
                "n_values": n,
                "diffs": [round(float(d), 4) for d in diffs],
                "diff_mean": round(diff_mean, 4),
                "diff_std": round(diff_std, 4),
                "cv_of_diffs": round(cv_diff, 6),
                "perfect_step": round(float(perfect_step), 4),
                "r_squared_to_perfect_ap": round(r_squared, 6),
                "severity": severity,
                "interpretation": (
                    f"{'🔴' if severity in ('critical','high') else '🟡'} "
                    f"序列 '{label}'：相邻差值 CV={cv_diff:.4f}，"
                    f"R²对完美等差数列={r_squared:.6f}。"
                    f"差值: {[round(float(d), 2) for d in diffs]}。"
                    + (
                        f"差值近乎恒定 — 典型的近似等差数列特征，"
                        f"这是人为编造数据的最强信号之一。"
                        if cv_diff < 0.15
                        else f"差值较为一致，存在编造可能性。"
                    )
                ),
            })

    flagged = len(progression_flags) > 0

    return {
        "flagged": flagged,
        "series_tested": len(ordered_series_list),
        "flagged_series": len(progression_flags),
        "details": progression_flags,
        "interpretation": (
            f"🔴 {len(progression_flags)}/{len(ordered_series_list)} 个序列"
            f"呈现近等差数列特征"
            if flagged
            else "所有序列的相邻差值变化正常，未发现等差数列特征。"
        ),
    }


def _detect_high_frequency_repeats(values: List[float]) -> Dict[str, Any]:
    """Detect values that appear suspiciously often in supposedly independent measurements."""
    if len(values) < 6:
        return {"flagged": False, "note": "数据不足", "n_analyzed": len(values)}

    val_counter = Counter(values)
    n_unique = len(val_counter)
    n_total = len(values)

    repeat_flags = []
    for val, count in val_counter.most_common():
        freq = count / n_total
        if count >= 3 and freq > 0.15:
            repeat_flags.append({
                "value": val,
                "count": count,
                "frequency": round(freq, 4),
                "severity": "critical" if count >= 4 and freq > 0.25 else "high",
                "interpretation": (
                    f"数值 {val} 出现 {count} 次（占比 {freq:.1%}）。"
                    + ("这在独立实验中极不合理。"
                       if count >= 4 else "可能不是独立实验产生。")
                ),
            })

    flagged = len(repeat_flags) > 0

    return {
        "flagged": flagged,
        "n_analyzed": n_total,
        "unique_values": n_unique,
        "uniqueness_ratio": round(n_unique / n_total, 4) if n_total > 0 else 0,
        "repeats": repeat_flags,
        "interpretation": (
            f"🔴 发现 {len(repeat_flags)} 个高频重复值" if flagged
            else "未发现异常高频重复值。"
        ),
    }


def _find_sample_size_for_mean(mean: float, text_values: Dict[str, Any]) -> Optional[int]:
    """Try to find a sample size associated with a mean value from PDF text."""
    sample_sizes = text_values.get("sample_sizes", [])
    if not sample_sizes:
        # Try common sample sizes
        for n in [3, 4, 5, 6, 8, 10]:
            product = mean * n
            if abs(product - round(product)) < 0.01:
                return n
        return None

    # Use the most common sample size
    from collections import Counter
    n_counter = Counter(sample_sizes)
    most_common_n = n_counter.most_common(1)[0][0] if n_counter else None

    if most_common_n:
        product = mean * most_common_n
        if abs(product - round(product)) < 0.01:
            return most_common_n

    return None


# ═══════════════════════════════════════════════════════════════════════════
# Step 9: Enhanced Guidance Generation
# ═══════════════════════════════════════════════════════════════════════════


def _generate_enhanced_guidance(
    panels: List[Dict[str, Any]],
    bar_candidates: List[Dict[str, Any]],
    risk_status: List[Dict[str, Any]],
    page_fig_map: Dict[int, str],
    datasets: List[Dict[str, Any]],
    matches: List[Dict[str, Any]],
    text_values: Dict[str, Any],
    statistical_precheck: Dict[str, Any],
) -> str:
    """Generate enhanced guidance with ACTUAL pre-computed results."""

    lines = []
    lines.append("## 🔬 系统确定性预处理结果（已在代码层面执行完毕）")
    lines.append("")
    lines.append("> ⚠️ 以下所有数据提取和统计检验已由系统**确定性地**预先执行。")
    lines.append("> 你**不需要**再调用 bar_chart_extract_values、cross_figure_data_compare、")
    lines.append("> benford_law_test 等工具——这些结果已经直接提供给你。")
    lines.append("> 你的任务是：**解读这些结果并写入最终报告**，不要自行判断'数据不足'。")
    lines.append("")

    # ── Summary ──
    lines.append(f"### 数据提取摘要")
    lines.append(f"- PDF面板总数：**{len(panels)}** 个")
    lines.append(f"- 柱状图候选：**{len(bar_candidates)}** 个")
    lines.append(f"- 成功提取数据集：**{len([d for d in datasets if d.get('values')])}** 个")
    lines.append(f"- PDF文本数值：**{text_values.get('total_values', 0)}** 个")
    lines.append(f"- 均值±SD对（文本）：**{len(text_values.get('mean_sd_pairs', []))}** 对")
    lines.append(f"- 精确p值（文本）：**{len(text_values.get('p_values', []))}** 个")
    lines.append("")

    # ── Cross-Figure Comparison Results ──
    lines.append("### 🔴 跨图数据比对结果（已执行完毕）")
    lines.append("")

    critical_matches = [m for m in matches if m.get("confidence") == "critical"]
    high_matches = [m for m in matches if m.get("confidence") == "high"]
    medium_matches = [m for m in matches if m.get("confidence") == "medium"]

    if critical_matches:
        lines.append(f"🚨 **发现 {len(critical_matches)} 个 CRITICAL 级别匹配！这是数据造假的铁证！**")
        lines.append("")
        for m in critical_matches:
            lines.append(f"#### {m['type']}: {m.get('dataset_a')} vs {m.get('dataset_b')}")
            lines.append(f"- 共享组：{m.get('common_groups', m.get('shared_values', 'N/A'))}")
            lines.append(f"- {m.get('interpretation', '')}")
            lines.append("")
    else:
        lines.append("✅ 未发现 CRITICAL 级别的跨图数据匹配。")
        lines.append("")

    if high_matches:
        lines.append(f"⚠️ 发现 {len(high_matches)} 个 HIGH 级别匹配：")
        for m in high_matches:
            lines.append(f"- {m.get('dataset_a')} vs {m.get('dataset_b')}: {m.get('interpretation', '')}")
        lines.append("")

    if medium_matches:
        lines.append(f"ℹ️ 发现 {len(medium_matches)} 个 MEDIUM 级别匹配：")
        for m in medium_matches:
            lines.append(f"- {m.get('dataset_a')} vs {m.get('dataset_b')}")
        lines.append("")

    # ── Extracted Datasets ──
    valid_datasets = [d for d in datasets if d.get("values")]
    if valid_datasets:
        lines.append("### 📊 从柱状图提取的数据集")
        lines.append("")
        lines.append("| 图表标签 | 组标签 | 数值 | 柱数 | 置信度 |")
        lines.append("|----------|--------|------|------|--------|")
        for ds in valid_datasets:
            label = ds.get("label", "?")
            groups = ", ".join(ds.get("group_labels", [])[:4])
            values = ", ".join(str(v) for v in ds.get("values", [])[:4])
            bars = ds.get("bar_count", "?")
            conf = ds.get("confidence", "?")
            lines.append(f"| {label} | {groups} | {values} | {bars} | {conf} |")
        lines.append("")

        # ── Ordered series for arithmetic progression detection ──
        lines.append("### 📐 有序序列（用于等差数列检测）")
        lines.append("")
        for ds in valid_datasets:
            series = ds.get("ordered_series", [])
            if len(series) >= 3:
                label = ds.get("label", "?")
                lines.append(f"- **{label}**: {series}")
        lines.append("")

    # ── Statistical Precheck Results ──
    lines.append("### 📈 统计检验预计算结果")
    lines.append("")

    benford = statistical_precheck.get("benford", {})
    if benford.get("sample_size"):
        lines.append("#### 本福特定律检验（已执行）")
        lines.append(f"- 样本量：{benford.get('sample_size')}")
        lines.append(f"- 卡方统计量：{benford.get('chi_squared')}")
        lines.append(f"- p值：{benford.get('p_value')}")
        lines.append(f"- MAD：{benford.get('mean_absolute_deviation')}（{benford.get('mad_assessment')}）")
        lines.append(f"- 判定：{'🔴 异常' if benford.get('flagged') else '✅ 正常'}")
        lines.append(f"- 解读：{benford.get('interpretation')}")
        lines.append("")
    else:
        lines.append(f"#### 本福特定律检验")
        lines.append(f"- 状态：**未执行** — {benford.get('note', '数据不足')}")
        lines.append("")

    pval = statistical_precheck.get("p_value_analysis", {})
    if pval.get("p_value_count", 0) >= 10:
        lines.append("#### p值分布分析（已执行）")
        lines.append(f"- p值总数：{pval.get('p_value_count')}")
        caliper = pval.get("caliper_test", {})
        lines.append(f"- 卡尺比率：{caliper.get('caliper_ratio')}（{caliper.get('p_values_in_0.045_to_0.05')} vs {caliper.get('p_values_in_0.05_to_0.055')}）")
        uniform = pval.get("uniformity_test", {})
        lines.append(f"- 均匀性检验：χ²={uniform.get('chi_squared')}, p={uniform.get('p_value')}")
        lines.append(f"- 判定：{'🔴 异常' if pval.get('flagged') else '✅ 正常'}")
        lines.append(f"- 解读：{pval.get('interpretation')}")
        lines.append("")
    else:
        lines.append(f"#### p值分布分析")
        lines.append(f"- 状态：**未执行** — 精确p值不足（n={pval.get('p_value_count', 0)}）")
        warning = pval.get("warning", "")
        if warning:
            lines.append(f"- ⚠️ {warning}")
        lines.append("")

    grim = statistical_precheck.get("grim", {})
    if grim.get("pairs_tested", 0) >= 1:
        lines.append("#### GRIM检验（已执行）")
        lines.append(f"- 检验配对：{grim.get('pairs_tested')}")
        lines.append(f"- 不一致数量：{grim.get('inconsistent_pairs')}")
        lines.append(f"- 判定：{'🔴 异常！数学上不可能！' if grim.get('flagged') else '✅ 一致'}")
        for r in grim.get("results", []):
            lines.append(f"  - {r.get('detail', '')}")
        lines.append("")
    else:
        lines.append(f"#### GRIM检验")
        lines.append(f"- 状态：**未执行** — {grim.get('note', '无法找到配对')}")
        lines.append("")

    anomalous = statistical_precheck.get("anomalous_precision", {})
    lines.append("#### 异常精度检测（已执行）")
    lines.append("")

    cv = anomalous.get("cv_analysis", {})
    lines.append(f"**变异系数(CV)分析**：检验{cv.get('pairs_tested')}对，"
                 f"严重{cv.get('cv_critical_count')}个，警告{cv.get('cv_warning_count')}个")
    for fp in cv.get("flagged_pairs", [])[:5]:
        lines.append(f"  - {'🔴' if fp.get('severity')=='critical' else '🟡'} "
                     f"均值={fp.get('mean')}, SD={fp.get('sd')}, CV={fp.get('cv')}")

    ld = anomalous.get("last_digit_analysis", {})
    lines.append(f"**末位数字偏好分析**：分析{ld.get('n_analyzed', 0)}个数值")
    lines.append(f"- 末位=0：{ld.get('last_digit_0_frequency', 0):.1%}")
    lines.append(f"- 末位=5：{ld.get('last_digit_5_frequency', 0):.1%}")
    if ld.get("digit_frequencies"):
        lines.append(f"- 完整分布：{ld.get('digit_frequencies')}")
    flags = ld.get("flags", [])
    if flags:
        for f in flags:
            lines.append(f"  - {f}")
    else:
        lines.append(f"  - ✅ 末位数字分布正常")

    ap = anomalous.get("near_arithmetic_progression", {})
    lines.append(f"**近等差数列检测**：检验{ap.get('series_tested', 0)}个序列")
    if ap.get("flagged"):
        lines.append(f"- 🔴 发现{ap.get('flagged_series')}个序列呈现近等差数列特征！")
        for detail in ap.get("details", []):
            lines.append(f"  - {detail.get('interpretation', '')}")
    else:
        lines.append(f"- ✅ 未发现近等差数列特征")

    hr = anomalous.get("high_frequency_repeats", {})
    lines.append(f"**高频重复值检测**：分析{hr.get('n_analyzed', 0)}个数值")
    if hr.get("flagged"):
        for r in hr.get("repeats", []):
            lines.append(f"  - 🔴 {r.get('interpretation', '')}")
    else:
        lines.append(f"- ✅ 未发现异常高频重复值")
    lines.append("")

    # ── All panels by figure ──
    lines.append("### 全部面板（按图表分组）")
    lines.append("")
    current_fig = None
    for p in panels:
        fig = p.get("figure_label", "?")
        if fig != current_fig:
            current_fig = fig
            lines.append(f"**{fig}**：")
        letter = p.get("panel_letter", "?")
        dims = f"{p.get('width')}×{p.get('height')}"
        is_bar = "📊" if p in bar_candidates else ""
        lines.append(f"  - 面板 {letter} ({dims}) {is_bar}")
    lines.append("")

    # ── Agent instructions ──
    lines.append("### 📋 你的任务（必须执行）")
    lines.append("")
    lines.append("1. **将上述预计算结果完整写入最终报告**。不要跳过，不要写'未执行'——")
    lines.append("   统计检验已经由系统确定性地执行了，结果就在上面。")
    lines.append("2. **如果跨图数据比对发现 CRITICAL/HIGH 匹配**：这些必须在报告中作为")
    lines.append("   FIND-DATA-001（最高优先级）列出，附上匹配类型、共享组名和置信度。")
    lines.append("3. **如果近等差数列检测标记了序列**：必须在报告中详细列出被标记的")
    lines.append("   序列及其 CV_diff 和 R² 值。这是数据编造的核心指纹。")
    lines.append("4. **如果末位数字偏好检测标记了5或0**：必须在报告中列出具体的末位")
    lines.append("   数字频率分布和偏离倍数。")
    lines.append("5. **评分时必须反映预计算的结果**：")
    lines.append("   - CRITICAL 跨图匹配 → 数据完整性评分 90-100")
    lines.append("   - 近等差数列标记 → 数据完整性评分 70-85")
    lines.append("   - 末位数字严重偏好(>25%) → 数据完整性评分 +15")
    lines.append("   - 仅透明度不足(无任何标记) → 数据完整性评分 20-40")
    lines.append("6. **禁止行为**：")
    lines.append("   - ❌ 禁止写'因数据不足未执行'——检验已经执行了")
    lines.append("   - ❌ 禁止写'p值仅以阈值形式报告'来跳过分析——文本中的数值已被提取")
    lines.append("   - ❌ 禁止给出 5 分的数据完整性评分除非确实所有检验均未标记异常")
    lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Formatted output for agent context injection
# ═══════════════════════════════════════════════════════════════════════════


def format_precheck_for_agent(precheck_result: Dict[str, Any]) -> str:
    """
    Format the precheck result for injection into the agent's task context.

    With the enhanced pipeline, this returns comprehensive pre-computed
    results including actual extracted data and statistical test outputs.
    """
    guidance = precheck_result.get("guidance", "")
    if guidance:
        return guidance

    # Fallback: this shouldn't happen with the enhanced pipeline,
    # but keep for robustness
    lines = ["## 系统预处理结果", ""]
    lines.append(f"- 页面→图表映射：{len(precheck_result.get('page_fig_map', {}))} 页")
    lines.append(f"- 发现面板：{len(precheck_result.get('panels', []))} 个")
    lines.append(f"- 柱状图候选：{len(precheck_result.get('bar_chart_candidates', []))} 个")

    errors = precheck_result.get("errors", [])
    if errors:
        lines.append("")
        lines.append("### 错误")
        for e in errors:
            lines.append(f"- {e}")

    lines.append("")
    lines.append("> 请手动提取柱状图数据并执行跨图比对。")
    return "\n".join(lines)
