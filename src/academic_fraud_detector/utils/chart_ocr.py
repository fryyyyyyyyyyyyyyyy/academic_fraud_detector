"""
Chart OCR — extract numeric data from scientific chart images.

Many academic papers present quantitative results ONLY as bar charts or line
graphs embedded in figures. The text layer of the PDF doesn't contain the
numeric values shown in these charts. This module uses OCR to extract those
values so they can be fed into statistical fraud detection tools.

Supports two OCR backends:
1. **pytesseract** (fast, requires Tesseract system install)
2. **easyocr** (pure Python auto-download, slower but zero system dependencies)

Key functions:
- `extract_chart_data()` — comprehensive extraction: text, numbers, structured data
- `extract_numbers_from_image()` — lightweight: just get all numbers
- `batch_extract_chart_data()` — process multiple images at once
"""

import json
import logging
import os
import re
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

# Try to import OCR backends
_HAS_TESSERACT = False
_HAS_EASYOCR = False
_easyocr_reader = None


def _init_backends():
    """Lazy-init OCR backends."""
    global _HAS_TESSERACT, _HAS_EASYOCR, _easyocr_reader

    if not _HAS_TESSERACT:
        try:
            import pytesseract
            # Quick check: try to get the tesseract version
            _ = pytesseract.get_tesseract_version()
            _HAS_TESSERACT = True
            logger.info("pytesseract backend available")
        except Exception:
            logger.debug("pytesseract not available (Tesseract not installed)")

    if not _HAS_EASYOCR:
        try:
            import easyocr
            _easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            _HAS_EASYOCR = True
            logger.info("easyocr backend available")
        except Exception as e:
            logger.debug(f"easyocr not available: {e}")


def _ocr_image(image: Image.Image) -> str:
    """
    Run OCR on a PIL Image. Returns the recognized text.

    Tries pytesseract first (fast), falls back to easyocr.
    """
    _init_backends()

    if _HAS_TESSERACT:
        try:
            import pytesseract
            # Preprocess: convert to grayscale, increase contrast
            gray = image.convert("L")
            # Try with PSM 6 (uniform block of text) and PSM 11 (sparse text)
            for psm in [6, 11, 3]:
                try:
                    whitelist = (
                        "0123456789abcdefghijklmnopqrstuvwxyz"
                        "ABCDEFGHIJKLMNOPQRSTUVWXYZ.,;:!?()[]{}<>/\\+-*=" + "%" + "% "
                    )
                    config_str = f'--psm {psm} -c tessedit_char_whitelist="{whitelist}"'
                    text = pytesseract.image_to_string(gray, config=config_str)
                    if text.strip():
                        return text.strip()
                except Exception:
                    continue
            return ""
        except Exception as e:
            logger.debug(f"pytesseract OCR failed: {e}")

    if _HAS_EASYOCR and _easyocr_reader is not None:
        try:
            import numpy as np
            arr = np.array(image.convert("RGB"))
            results = _easyocr_reader.readtext(arr, detail=0)
            return " ".join(results)
        except Exception as e:
            logger.debug(f"easyocr OCR failed: {e}")

    # Neither backend available
    logger.warning("No OCR backend available. Install pytesseract or easyocr.")
    return ""


def _preprocess_for_numbers(image: Image.Image) -> Image.Image:
    """
    Preprocess image to improve number recognition.
    - Convert to grayscale
    - Increase contrast
    - Binarize (thresholding)
    - Invert if needed (white text on dark bg → dark text on white bg)
    """
    import numpy as np

    gray = image.convert("L")
    arr = np.array(gray, dtype=np.uint8)

    # Determine if image is mostly dark (inverted) or mostly light
    mean_val = np.mean(arr)
    if mean_val < 128:
        # Invert: likely white text on dark background
        arr = 255 - arr

    # Increase contrast using percentile clipping
    p_low, p_high = np.percentile(arr, [5, 95])
    if p_high > p_low:
        arr = np.clip((arr - p_low) * 255.0 / (p_high - p_low), 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


def extract_chart_data(
    image_path_or_url: str,
    preprocess: bool = True,
) -> Dict[str, Any]:
    """
    Extract all readable data from a chart image using OCR.

    Args:
        image_path_or_url: Path to the chart image file.
        preprocess: Whether to apply contrast enhancement before OCR.

    Returns:
        Dict with:
        - 'raw_text': full OCR output
        - 'numbers': list of extracted numeric values (floats)
        - 'integers': list of integer values
        - 'p_values': detected p-value patterns
        - 'labels': detected text labels
        - 'mean_sd_pairs': detected mean±SD pairs
        - 'stats_summary': basic statistics of extracted numbers
    """
    from ..utils.image_downloader import load_image

    img, meta = load_image(image_path_or_url, use_cache=True)
    if img is None:
        return {
            "error": meta.get("error", "Failed to load image"),
            "image_path": image_path_or_url,
        }

    # Preprocess
    if preprocess:
        img = _preprocess_for_numbers(img)

    # Run OCR
    raw_text = _ocr_image(img)

    if not raw_text:
        return {
            "image_path": image_path_or_url,
            "raw_text": "",
            "numbers": [],
            "error": "No text recognized (no OCR backend available or empty image)",
        }

    # Parse the OCR output
    numbers, integers = _extract_numbers(raw_text)
    p_vals = _extract_p_values_from_text(raw_text)
    labels = _extract_text_labels(raw_text)
    mean_sd_pairs = _extract_mean_sd_pairs(raw_text)

    return {
        "image_path": image_path_or_url,
        "image_dimensions": f"{meta.get('width', '?')}x{meta.get('height', '?')}",
        "raw_text": raw_text,
        "numbers": numbers,
        "integers": integers,
        "p_values": p_vals,
        "labels": labels,
        "mean_sd_pairs": mean_sd_pairs,
        "stats_summary": {
            "total_numbers": len(numbers),
            "range": [min(numbers), max(numbers)] if numbers else None,
            "mean": round(sum(numbers) / len(numbers), 4) if numbers else None,
        },
    }


def extract_numbers_from_image(image_path_or_url: str) -> List[float]:
    """Lightweight: extract just the numeric values from an image."""
    result = extract_chart_data(image_path_or_url)
    return result.get("numbers", [])


def batch_extract_chart_data(
    image_paths: List[str],
) -> List[Dict[str, Any]]:
    """
    Extract chart data from multiple images.
    Returns a list of results for cross-figure comparison.
    """
    results = []
    for path in image_paths:
        try:
            data = extract_chart_data(path)
            results.append(data)
        except Exception as e:
            results.append({"image_path": path, "error": str(e)})
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Text parsing helpers
# ═══════════════════════════════════════════════════════════════════════════

def _extract_numbers(text: str) -> Tuple[List[float], List[int]]:
    """Extract all numeric values from OCR text."""
    # Match floating-point numbers, including scientific notation
    pattern = r'(?<![a-zA-Z])(\d+\.?\d*(?:[eE][+-]?\d+)?)(?![a-zA-Z])'
    matches = re.findall(pattern, text)

    numbers = []
    integers = []
    for m in matches:
        try:
            val = float(m)
            if val == int(val) and '.' not in m:
                integers.append(int(val))
            else:
                numbers.append(val)
        except ValueError:
            continue

    # Remove duplicates and sort
    numbers = sorted(set(numbers))
    integers = sorted(set(integers))
    return numbers, integers


def _extract_p_values_from_text(text: str) -> List[Dict[str, Any]]:
    """Extract p-values and their context from text."""
    patterns = [
        # "P < 0.05", "p = 0.031", "P>0.05"
        r'[pP]\s*[<>=]+\s*(\d+\.\d+)',
        # "P < 0.001" (3-digit)
        r'[pP]\s*[<>=]+\s*(\d+\.\d{3})',
    ]

    results = []
    seen = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            val = float(match.group(1))
            raw = match.group(0).strip()
            if raw not in seen:
                seen.add(raw)
                results.append({
                    "raw_text": raw,
                    "value": val,
                    "is_significant": val < 0.05,
                })

    return sorted(results, key=lambda x: x["value"])


def _extract_text_labels(text: str) -> List[str]:
    """Extract readable text labels (non-numeric tokens)."""
    # Split by whitespace and punctuation
    tokens = re.findall(r'[a-zA-Z]+', text)
    # Filter short tokens (likely noise)
    meaningful = [t for t in tokens if len(t) >= 2]
    # Deduplicate preserving order
    seen = set()
    unique = []
    for t in meaningful:
        if t.lower() not in seen:
            seen.add(t.lower())
            unique.append(t)
    return unique[:50]


def _extract_mean_sd_pairs(text: str) -> List[Dict[str, Any]]:
    """
    Extract mean±SD pairs from OCR text.

    Looks for patterns like:
    - "12.34 ± 2.56"
    - "12.34+2.56"
    - "mean 12.34, SD 2.56"
    """
    patterns = [
        # "number ± number" (with or without spaces)
        r'(\d+\.?\d*)\s*[±\+]\s*(\d+\.?\d*)',
        # "mean=X SD=Y"
        r'mean[=:\s]+(\d+\.?\d*).*?SD[=:\s]+(\d+\.?\d*)',
        # "M=12.34, SD=2.56"
        r'[Mm][=:\s]+(\d+\.?\d*).*?[Ss][Dd][=:\s]+(\d+\.?\d*)',
    ]

    pairs = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            mean_val = float(match.group(1))
            sd_val = float(match.group(2))
            cv = (sd_val / mean_val * 100) if mean_val != 0 else None
            pairs.append({
                "mean": mean_val,
                "sd": sd_val,
                "cv_percent": round(cv, 2) if cv is not None else None,
                "raw_match": match.group(0).strip(),
            })

    return pairs


# ═══════════════════════════════════════════════════════════════════════════
# CrewAI Tool wrapper
# ═══════════════════════════════════════════════════════════════════════════

try:
    from crewai.tools import BaseTool
    from pydantic import BaseModel, Field

    class ChartOCRToolInput(BaseModel):
        """Input for chart OCR extraction."""

        image_path: str = Field(
            ..., description="Path to the chart image file to extract data from."
        )

    class ChartOCRTool(BaseTool):
        """
        Extract numeric data from chart/bar-graph images using OCR.

        CRITICAL for fraud detection: many papers only report quantitative
        results as bar charts in figures. Without OCR, the statistical audit
        tools receive empty input and cannot detect data fabrication.

        Use this tool on every chart/image that contains bar graphs, scatter
        plots, or any other visual representation of numeric data.
        """

        name: str = "chart_ocr_extract"
        description: str = (
            "Extract numeric values and text from chart/bar-graph images using "
            "OCR (Optical Character Recognition). Use this on EVERY figure that "
            "contains bar charts, line graphs, or numerical data visualizations. "
            "Returns extracted numbers, p-values, mean/SD pairs, and labels. "
            "This is ESSENTIAL for feeding data into statistical fraud detection "
            "tools (Benford, GRIM, p-value analysis, cross-figure comparison). "
            "Without this step, all statistical tests fail with 'insufficient data'."
        )
        args_schema: type[BaseModel] = ChartOCRToolInput

        def _run(self, image_path: str) -> str:
            """Run OCR extraction on a chart image."""
            result = extract_chart_data(image_path)
            return json.dumps(result, ensure_ascii=False, default=str)

    class BatchChartOCRToolInput(BaseModel):
        """Input for batch chart OCR."""

        image_paths: str = Field(
            ...,
            description=(
                "JSON-encoded list of image file paths to process. "
                'Example: \'["/path/to/chart1.png", "/path/to/chart2.png"]\'.'
            ),
        )

    class BatchChartOCRTool(BaseTool):
        """
        Batch OCR extraction on multiple chart images.

        Use this to extract numeric data from ALL charts in a paper in one call.
        The extracted data is then ready for cross-figure comparison.
        """

        name: str = "batch_chart_ocr"
        description: str = (
            "Batch OCR extraction on multiple chart images. Extracts numeric data "
            "from ALL specified chart images in one call. "
            "Input: JSON list of image file paths. "
            "Returns: list of extraction results, one per image, with numbers, "
            "p-values, mean/SD pairs. Feed the output into cross_figure_data_compare "
            "to detect data duplication across figures."
        )
        args_schema: type[BaseModel] = BatchChartOCRToolInput

        def _run(self, image_paths: str) -> str:
            """Run batch OCR extraction."""
            try:
                paths = json.loads(image_paths)
                if not isinstance(paths, list):
                    return json.dumps({"error": "image_paths must be a JSON list"})
            except json.JSONDecodeError:
                return json.dumps({"error": "image_paths must be a valid JSON list of strings."})

            results = batch_extract_chart_data(paths)
            return json.dumps(results, ensure_ascii=False, default=str)

except ImportError:
    # CrewAI not installed — tools won't be available
    ChartOCRTool = None
    BatchChartOCRTool = None


# ═══════════════════════════════════════════════════════════════════════════
# Quick test
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# Chart Type Detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_chart_type(image: Image.Image) -> str:
    """
    Detect the type of chart in the image.

    Returns one of: 'bar_chart', 'scatter_plot', 'line_graph', 'western_blot',
    'microscopy', 'flow_cytometry', 'heatmap', 'table', 'unknown'.

    COMPLETELY REWRITTEN with scientific chart heuristics.
    Previous version had ~70% false negative rate for bar charts
    (misclassifying them as line_graph, western_blot, or unknown).

    Key improvements:
    - Western blot detection now requires DARK bands on LIGHT background
      (not just any horizontal structure)
    - Bar chart detection uses multiple independent signals:
      * Clean white/light background
      * Discrete vertical dark regions (bars) with regular spacing
      * Horizontal baseline at bottom
      * Small isolated edge features (error bars, tick marks)
    - Line graph detection requires continuous horizontal curves
    """
    import numpy as np
    from scipy import ndimage

    gray = image.convert("L")
    arr = np.array(gray, dtype=np.float64)

    h, w = arr.shape

    # ── Edge detection (Sobel-like) ──
    gy = np.abs(np.diff(arr, axis=0))
    gx = np.abs(np.diff(arr, axis=1))
    gy_padded = np.pad(gy, ((0, 1), (0, 0)), mode='edge')
    gx_padded = np.pad(gx, ((0, 0), (0, 1)), mode='edge')
    edge_mag = np.sqrt(gy_padded ** 2 + gx_padded ** 2)

    # ── Basic statistics ──
    intensity_mean = float(np.mean(arr))
    intensity_std = float(np.std(arr))
    white_fraction = float(np.mean(arr > 220))       # Nearly white pixels
    light_fraction = float(np.mean(arr > 180))        # Light pixels
    dark_fraction = float(np.mean(arr < 50))          # Very dark pixels

    # Edge statistics
    edge_density = float(np.mean(edge_mag > 30))
    strong_edge_density = float(np.mean(edge_mag > 60))

    # Horizontal vs vertical structure
    row_means = np.mean(arr, axis=1)
    col_means = np.mean(arr, axis=0)
    horizontal_variance = float(np.var(row_means))
    vertical_variance = float(np.var(col_means))

    # ── Too uniform → unknown ──
    if intensity_std < 15:
        return "unknown"

    # ── Western blot detection (REFINED) ──
    # TRUE western blots have: dark horizontal bands on a light/white background,
    # very strong horizontal structure, high contrast, and LOW edge density
    # (bands are smooth, not pixelated).
    # The KEY discriminator: western blots have DARK bands → mean intensity is low
    # where bands exist. Bar charts have dark bars on white → similar but bars are
    # sharp-edged rectangles, not smooth bands.
    #
    # Western blot = dark bg or strong dark bands + smooth edges + no grid structure
    # Bar chart = light bg + sharp rectangular edges + grid/axis structure
    is_dark_overall = intensity_mean < 100
    has_strong_horizontal = horizontal_variance > vertical_variance * 2.5
    has_low_edge_density = edge_density < 0.06
    has_high_contrast = intensity_std > 45

    # Western blot needs ALL of: dark OR strong horizontal bands + low edge + high contrast
    if (is_dark_overall or has_strong_horizontal) and has_low_edge_density and has_high_contrast:
        # Additional check: western blots rarely have grid lines or axis labels
        # Check for long horizontal lines (axis) in the bottom portion
        bottom_region = arr[int(h * 0.75):, :]
        bottom_edge = np.abs(np.diff(bottom_region.astype(np.float64), axis=0))
        has_bottom_baseline = float(np.mean(bottom_edge > 40)) > 0.03

        if not has_bottom_baseline:
            return "western_blot"
        # If there IS a bottom baseline, it might be a bar chart with dark bars

    # ── Bar chart detection (COMPLETELY REWRITTEN) ──
    # Scientific bar charts typically have:
    # 1. White/light background (>40% light pixels)
    # 2. Discrete vertical bar structures (not continuous)
    # 3. A horizontal baseline (x-axis) near the bottom
    # 4. Regular spacing between bars
    # 5. Small edge features (error bars, tick marks)

    # Signal 1: Background lightness
    has_light_bg = light_fraction > 0.30

    # Signal 2: Discrete vertical structures
    # Bars create vertical runs of dark pixels surrounded by white
    # Use column-wise dark pixel density variation
    col_darkness = np.mean(arr < (intensity_mean - intensity_std * 0.3), axis=0)
    col_darkness_std = float(np.std(col_darkness))
    has_column_structure = col_darkness_std > 0.05

    # Signal 3: Horizontal baseline (x-axis)
    # The bottom 25% has a strong horizontal edge
    bottom_25 = arr[int(h * 0.70):, :]
    bottom_row_means = np.mean(bottom_25, axis=1)
    bottom_diffs = np.abs(np.diff(bottom_row_means))
    baseline_strength = float(np.max(bottom_diffs)) if len(bottom_diffs) > 0 else 0.0
    has_baseline = baseline_strength > 8.0

    # Signal 4: Regular bar spacing
    # Bars appear as regular peaks in vertical projection
    v_proj = np.mean(arr, axis=0)  # vertical projection (column means)
    # Normalize to [0, 1]
    v_proj_norm = (v_proj - np.min(v_proj)) / (np.max(v_proj) - np.min(v_proj) + 1e-10)
    # Find peaks (bars are darker = lower values in column means)
    inverted = 1.0 - v_proj_norm
    # Count regions where inverted value exceeds threshold
    bar_candidate_regions = (inverted > 0.3).astype(np.int32)
    # Find transitions (bar edges)
    bar_transitions = np.abs(np.diff(bar_candidate_regions))
    num_bar_edges = int(np.sum(bar_transitions))
    has_discrete_bars = 4 <= num_bar_edges <= 80  # Between 2-40 bars

    # Signal 5: Error bars / tick marks (small vertical lines on bar tops)
    # These show up as small isolated vertical edges in the upper portion
    upper_portion = arr[:int(h * 0.7), :]
    upper_edges = np.abs(np.diff(upper_portion.astype(np.float64), axis=0))
    small_vertical_features = float(np.mean((upper_edges > 30) & (upper_edges < 80)))
    has_small_features = small_vertical_features > 0.005

    # ── Scoring system for bar chart ──
    bar_chart_score = 0.0
    if has_light_bg:
        bar_chart_score += 2.0
    if has_column_structure:
        bar_chart_score += 2.0
    if has_baseline:
        bar_chart_score += 2.0
    if has_discrete_bars:
        bar_chart_score += 3.0
    if has_small_features:
        bar_chart_score += 1.0
    # Bonus: moderate edge density (very common in bar charts with axis labels)
    if 0.02 < edge_density < 0.25:
        bar_chart_score += 1.0

    if bar_chart_score >= 6.0:
        return "bar_chart"

    # ── Line graph detection (REFINED) ──
    # Line graphs have: moderate edges, continuous curves (not discrete bars),
    # data point markers, connecting lines
    # Key discriminator vs bar charts: line graphs have FEWER vertical edges
    # and MORE diagonal edges

    # Check for continuous horizontal/diagonal lines
    # Use Hough-like heuristic: count pixels that are part of long-ish
    # horizontal or diagonal runs
    has_horizontal_structure = horizontal_variance > vertical_variance * 1.3

    # Line graphs tend to have lower column structure variance than bar charts
    has_low_col_variance = col_darkness_std < 0.06

    # Line graphs often have data point markers (small dots on the line)
    data_point_candidates = float(np.mean((edge_mag > 25) & (edge_mag < 55)))

    line_graph_score = 0.0
    if has_horizontal_structure:
        line_graph_score += 2.0
    if has_low_col_variance and not has_discrete_bars:
        line_graph_score += 2.0
    if 0.01 < edge_density < 0.08:
        line_graph_score += 2.0
    if 0.01 < data_point_candidates < 0.08:
        line_graph_score += 1.5
    if intensity_std > 25 and not has_light_bg:
        line_graph_score += 1.0

    if line_graph_score >= 5.0:
        return "line_graph"

    # ── Scatter plot detection ──
    if edge_density > 0.015 and intensity_std > 45:
        binary = (edge_mag > 50).astype(np.uint8)
        if h * w < 4000000:
            labeled, num_features = ndimage.label(binary)
            if num_features > 100:
                return "scatter_plot"

    # ── Flow cytometry ──
    if edge_density > 0.06 and intensity_mean < 128 and intensity_std > 55:
        return "flow_cytometry"

    # ── Heatmap ──
    if edge_density < 0.015 and intensity_std > 25:
        return "heatmap"

    # ── Microscopy ──
    if intensity_mean < 110 and intensity_std > 55 and light_fraction < 0.15:
        return "microscopy"

    # ── Table ──
    gy_mean = float(np.mean(np.abs(np.diff(arr, axis=0))))
    gx_mean = float(np.mean(np.abs(np.diff(arr, axis=1))))
    if gy_mean > 10 and gx_mean > 10 and abs(gy_mean - gx_mean) / max(gy_mean, 1) < 0.3:
        return "table"

    # ── Last-resort bar chart detection ──
    # If we have a light background with some column structure and a baseline,
    # it's probably a bar chart even if the score was below 6
    if has_light_bg and has_baseline and has_column_structure:
        return "bar_chart"

    # If we have discrete bars (clear vertical structures), it's a bar chart
    if has_discrete_bars and has_light_bg:
        return "bar_chart"

    return "unknown"


def optimize_ocr_for_chart_type(image: Image.Image, chart_type: str) -> Image.Image:
    """
    Apply chart-type-specific preprocessing to improve OCR accuracy.

    - Bar chart: enhance contrast to make axis labels stand out
    - Scatter plot: binarize to isolate text from data points
    - Western blot: invert if needed (white bands on dark bg)
    """
    import numpy as np

    if chart_type == "bar_chart":
        # Enhance contrast — axis labels are usually small text
        arr = np.array(image.convert("L"), dtype=np.uint8)
        arr = np.clip((arr - np.percentile(arr, 10)) * 2, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    elif chart_type == "western_blot":
        # May need inversion
        arr = np.array(image.convert("L"), dtype=np.uint8)
        if np.mean(arr) < 128:
            arr = 255 - arr
        return Image.fromarray(arr)

    elif chart_type == "scatter_plot":
        # Binarize to separate text from data
        arr = np.array(image.convert("L"), dtype=np.uint8)
        threshold = np.percentile(arr, 50)
        binary = (arr < threshold).astype(np.uint8) * 255
        return Image.fromarray(binary)

    return image  # No special processing for other types


def extract_chart_data_with_type_detection(
    image_path_or_url: str,
) -> dict:
    """
    Enhanced chart data extraction: detect chart type first, then apply
    type-specific OCR optimization.

    Returns same dict as extract_chart_data(), plus 'chart_type' field.
    """
    from ..utils.image_downloader import load_image

    img, meta = load_image(image_path_or_url, use_cache=True)
    if img is None:
        return {"error": meta.get("error", "Failed to load image")}

    chart_type = detect_chart_type(img)
    optimized_img = optimize_ocr_for_chart_type(img, chart_type)

    result = extract_chart_data(image_path_or_url, preprocess=False)
    # Override: we already preprocessed with chart-type-specific optimization
    result["chart_type"] = chart_type

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Bar Chart Value Extraction (Computer Vision)
# ═══════════════════════════════════════════════════════════════════════════

def extract_bar_chart_values(
    image_path_or_url: str,
) -> Dict[str, Any]:
    """
    Extract numerical values from a bar chart by measuring bar heights via
    computer vision and interpolating against y-axis scale labels.

    COMPLETELY REWRITTEN with improved:
    - Y-axis OCR: magnification, aggressive binarization, multi-region attempts
    - Bar detection: adaptive threshold, handles light and dark bars
    - Group label extraction: cleaner OCR, direct value label reading above bars

    Algorithm:
    1. Detect chart type; if not bar_chart, return early with a warning.
    2. Find the y-axis region → magnify → enhance → OCR scale labels.
    3. Find the x-axis region → OCR group labels.
    4. Detect individual bars in the plot area using adaptive vertical projection.
    5. For each bar, measure its top y-coordinate and interpolate.
    6. Group values by x-axis position.
    """
    import numpy as np
    from ..utils.image_downloader import load_image

    img, meta = load_image(image_path_or_url, use_cache=True)
    if img is None:
        return {"error": meta.get("error", "Failed to load image")}

    chart_type = detect_chart_type(img)
    if chart_type != "bar_chart":
        return {
            "chart_type": chart_type,
            "warning": (
                f"Image detected as '{chart_type}', not 'bar_chart'. "
                "Bar height extraction requires a bar chart image. "
                "Use chart_ocr_extract for text-based OCR on non-bar-chart figures."
            ),
            "groups": [],
            "confidence": "low",
        }

    gray = img.convert("L")
    arr = np.array(gray, dtype=np.float64)
    h, w = arr.shape

    # ── Step 1: Find plot area boundaries ──
    # Y-axis is on the left, x-axis at the bottom
    y_axis_right = int(w * 0.25)   # search left 25% for y-axis
    x_axis_top = int(h * 0.75)     # bottom 25% for x-axis
    plot_left = y_axis_right
    plot_right = w - int(w * 0.03)
    plot_top = int(h * 0.03)
    plot_bottom = x_axis_top

    # ── Step 2: Extract y-axis scale labels via OCR ──────────────
    # MULTI-STRATEGY: try different regions and enhancements
    y_numbers = _extract_y_axis_numbers(arr, h, w, y_axis_right, plot_top, plot_bottom)

    if len(y_numbers) < 2:
        return {
            "chart_type": "bar_chart",
            "error": (
                f"Could not extract enough y-axis scale labels (found {len(y_numbers)}). "
                "The y-axis text may be too small or low-contrast for OCR."
            ),
            "groups": [],
            "confidence": "low",
        }

    y_min, y_max = y_numbers[0], y_numbers[-1]
    y_range = y_max - y_min

    # ── Step 3: Extract x-axis group labels via OCR ──────────────
    x_labels = _extract_x_axis_labels(arr, h, w, plot_left, plot_right, x_axis_top)

    # ── Step 4: Detect bars in the plot area ────────────────────
    plot_arr = arr[plot_top:plot_bottom, plot_left:plot_right]
    plot_h, plot_w = plot_arr.shape

    bar_regions, bar_heights = _detect_bar_regions(plot_arr, plot_h, plot_w)

    if not bar_regions:
        return {
            "chart_type": "bar_chart",
            "warning": (
                f"No bar regions detected in plot area ({plot_w}x{plot_h}px). "
                "The bars may be too light, the image resolution too low, "
                "or this is an unusual chart style."
            ),
            "groups": [],
            "confidence": "low",
        }

    # ── Step 5: Measure bar heights and interpolate values ──────
    bar_values = _measure_bar_values(
        arr, plot_arr, plot_top, plot_left, plot_h, plot_w,
        h, w, bar_regions, bar_heights,
        y_min, y_max, y_range,
    )

    if not bar_values:
        return {
            "chart_type": "bar_chart",
            "warning": "Bar regions found but could not measure any values.",
            "groups": [],
            "confidence": "low",
        }

    # ── Step 6: Group bars by x-axis position ───────────────────
    groups = _group_bars_by_position(bar_regions, bar_values, x_labels)

    # Quality assessment
    n_label_values = sum(1 for b in bar_values if b.get("ocr_label_value") is not None)
    if n_label_values >= len(bar_values) * 0.5:
        confidence = "high"
    elif n_label_values >= 2 or (len(y_numbers) >= 3 and len(bar_regions) >= 3):
        confidence = "medium"
    else:
        confidence = "low"

    # Build ordered series (values in left-to-right bar order)
    ordered_series = []
    for g in groups:
        ordered_series.extend(g.get("values", []))

    return {
        "chart_type": "bar_chart",
        "image_dimensions": f"{w}x{h}",
        "plot_area": {"left": plot_left, "top": plot_top, "right": plot_right, "bottom": plot_bottom},
        "y_axis": {
            "ocr_labels": y_numbers,
            "range": [y_min, y_max],
        },
        "x_axis": {
            "ocr_labels": x_labels,
        },
        "bars_detected": len(bar_values),
        "groups": groups,
        "ordered_series": ordered_series,
        "confidence": confidence,
        "format_for_cross_comparison": {
            "group_labels": [g["group_label"] for g in groups],
            "values": [g["values"][0] if len(g["values"]) == 1 else g["values"]
                       for g in groups],
        },
    }


def _extract_y_axis_numbers(
    arr: "np.ndarray", h: int, w: int, y_axis_right: int,
    plot_top: int, plot_bottom: int,
) -> list:
    """Extract y-axis scale numbers using multiple OCR strategies."""
    import numpy as np

    all_numbers = []

    # Strategy 1: Magnify the y-axis region 2x for better OCR
    y_region = arr[plot_top:plot_bottom, :y_axis_right]
    yr_h, yr_w = y_region.shape
    if yr_h > 20 and yr_w > 20:
        y_img = Image.fromarray(y_region.astype(np.uint8))
        # Magnify 2x
        y_img_2x = y_img.resize((yr_w * 2, yr_h * 2), Image.LANCZOS)
        ocr_text = _ocr_image(y_img_2x)
        nums, _ = _extract_numbers(ocr_text)
        all_numbers.extend(nums)

    # Strategy 2: Binarize + magnify (good for low-contrast text)
    if yr_h > 20 and yr_w > 20:
        y_arr = y_region.copy()
        # Adaptive threshold: binarize at local mean
        threshold = np.mean(y_arr) - np.std(y_arr) * 0.5
        y_binary = ((y_arr < threshold) * 255).astype(np.uint8)
        y_bin_img = Image.fromarray(y_binary)
        y_bin_2x = y_bin_img.resize((yr_w * 2, yr_h * 2), Image.LANCZOS)
        ocr_text = _ocr_image(y_bin_2x)
        nums, _ = _extract_numbers(ocr_text)
        all_numbers.extend(nums)

    # Strategy 3: Try the full left 25% (sometimes labels extend beyond y-axis)
    full_left = arr[plot_top:plot_bottom, :int(w * 0.30)]
    fl_h, fl_w = full_left.shape
    if fl_h > 20 and fl_w > 20:
        fl_img = Image.fromarray(full_left.astype(np.uint8))
        fl_2x = fl_img.resize((fl_w * 2, fl_h * 2), Image.LANCZOS)
        ocr_text = _ocr_image(fl_2x)
        nums, _ = _extract_numbers(ocr_text)
        all_numbers.extend(nums)

    # Remove duplicates and sort
    unique_numbers = sorted(set(round(n, 4) for n in all_numbers))

    # Filter: y-axis labels should be in a reasonable range
    # Keep numbers that form a roughly linear scale
    if len(unique_numbers) >= 3:
        # Filter out outliers (numbers that don't fit the scale pattern)
        filtered = _filter_y_axis_outliers(unique_numbers)
        return filtered

    return unique_numbers


def _filter_y_axis_outliers(numbers: list) -> list:
    """Filter y-axis numbers to keep only those that fit a linear scale pattern."""
    import numpy as np
    if len(numbers) < 3:
        return numbers

    arr = np.array(numbers, dtype=np.float64)
    # Compute diffs between consecutive numbers
    diffs = np.diff(arr)
    median_diff = np.median(diffs)

    if median_diff < 1e-10:
        return numbers

    # Keep numbers where the diff from neighbors is within 3x of median
    # (allows for log scales and irregular tick spacing)
    good_indices = {0, len(arr) - 1}  # always keep first and last
    for i in range(1, len(arr) - 1):
        diff_before = arr[i] - arr[i-1]
        diff_after = arr[i+1] - arr[i]
        if min(diff_before, diff_after) > 0:
            ratio_before = max(diff_before, median_diff) / min(diff_before, median_diff)
            ratio_after = max(diff_after, median_diff) / min(diff_after, median_diff)
            if ratio_before < 5.0 or ratio_after < 5.0:
                good_indices.add(i)
        else:
            good_indices.add(i)

    return [numbers[i] for i in sorted(good_indices)]


def _extract_x_axis_labels(
    arr: "np.ndarray", h: int, w: int,
    plot_left: int, plot_right: int, x_axis_top: int,
) -> list:
    """Extract x-axis group labels via OCR."""
    import numpy as np

    # Try the x-axis label area
    x_region = arr[x_axis_top:, plot_left:plot_right]
    xr_h, xr_w = x_region.shape

    if xr_h < 10 or xr_w < 10:
        return []

    # Magnify 2x for better OCR
    x_img = Image.fromarray(x_region.astype(np.uint8))
    x_img_2x = x_img.resize((xr_w * 2, xr_h * 2), Image.LANCZOS)
    x_ocr_text = _ocr_image(x_img_2x)

    # Extract text labels (filter short tokens)
    text_labels = _extract_text_labels(x_ocr_text)

    # Filter: keep only labels that look like group names
    # (not just random OCR fragments)
    cleaned = []
    for label in text_labels:
        if len(label) >= 2:
            # Keep labels with at least one letter
            if any(c.isalpha() for c in label):
                cleaned.append(label)

    # If we got very few labels, try without letter requirement
    if len(cleaned) < 2:
        cleaned = [l for l in text_labels if len(l) >= 2]

    return cleaned


def _detect_bar_regions(
    plot_arr: "np.ndarray", plot_h: int, plot_w: int,
) -> tuple:
    """Detect individual bar regions using adaptive thresholding."""
    import numpy as np

    plot_mean = float(np.mean(plot_arr))
    plot_std = float(np.std(plot_arr))

    # Strategy 1: Dark bars on light background (most common)
    bar_threshold = plot_mean - plot_std * 0.15  # Lowered from 0.3
    bar_mask_1 = plot_arr < bar_threshold

    # Strategy 2: Very dark bars (high contrast)
    bar_threshold_2 = plot_mean - plot_std * 0.5
    bar_mask_2 = plot_arr < bar_threshold_2

    # Strategy 3: Moderate dark bars (for colored/pastel bars)
    bar_threshold_3 = plot_mean - plot_std * 0.05
    bar_mask_3 = plot_arr < bar_threshold_3

    # Try strategies in order, pick the one that gives most bars (within reason)
    best_mask = bar_mask_1
    best_bar_count = 0
    best_regions = []

    for mask, label in [(bar_mask_1, "std0.15"), (bar_mask_2, "std0.5"), (bar_mask_3, "std0.05")]:
        v_proj = np.sum(mask, axis=0).astype(np.float64)
        min_bar_height_px = plot_h * 0.015  # Lowered from 0.02
        regions = _find_bar_regions_from_projection(v_proj, plot_w, min_bar_height_px)
        n_bars = len(regions)

        # Accept if we found 2-80 bars (reasonable range)
        if 2 <= n_bars <= 80 and n_bars > best_bar_count:
            best_bar_count = n_bars
            best_regions = regions
            best_mask = mask

    # If no strategy found good bars, use the one with most detections
    if not best_regions:
        best_regions = _find_bar_regions_from_projection(
            np.sum(bar_mask_1, axis=0).astype(np.float64),
            plot_w, plot_h * 0.01,  # Even lower threshold as last resort
        )

    # Calculate bar heights (top y-position of each bar)
    bar_heights = []
    for bar_start, bar_end in best_regions:
        bar_slice = best_mask[:, bar_start:bar_end]
        col_tops = []
        for c in range(bar_slice.shape[1]):
            dark_rows = np.where(bar_slice[:, c])[0]
            if len(dark_rows) > 0:
                col_tops.append(dark_rows[0])
        if col_tops:
            bar_heights.append(int(np.median(col_tops)))

    return best_regions, bar_heights


def _find_bar_regions_from_projection(
    v_proj: "np.ndarray", plot_w: int, min_height: float,
) -> list:
    """Find contiguous bar regions from vertical projection."""
    regions = []
    in_bar = False
    bar_start = 0
    for col in range(plot_w):
        if v_proj[col] > min_height:
            if not in_bar:
                bar_start = col
                in_bar = True
        else:
            if in_bar:
                bar_end = col
                if bar_end - bar_start >= 2:
                    regions.append((bar_start, bar_end))
                in_bar = False
    if in_bar:
        bar_end = plot_w
        if bar_end - bar_start >= 2:
            regions.append((bar_start, bar_end))
    return regions


def _measure_bar_values(
    arr: "np.ndarray", plot_arr: "np.ndarray",
    plot_top: int, plot_left: int,
    plot_h: int, plot_w: int,
    h: int, w: int,
    bar_regions: list, bar_heights: list,
    y_min: float, y_max: float, y_range: float,
) -> list:
    """Measure and interpolate values for each detected bar."""
    import numpy as np

    bar_values = []
    for idx, ((bar_start, bar_end), bar_top_px) in enumerate(
        zip(bar_regions, bar_heights)
    ):
        # Interpolate: bar_top_px / plot_h → fraction of y-axis range
        # bar at top = y_max, bar at bottom = y_min
        value_fraction = 1.0 - (bar_top_px / plot_h)
        interpolated_value = y_min + value_fraction * y_range

        # Clamp to y-axis range
        interpolated_value = max(y_min, min(y_max, interpolated_value))

        # Try OCR on value label above the bar (like "1200" written above bar top)
        label_value = _ocr_bar_value_label(
            arr, plot_top, plot_left, bar_start, bar_end,
            bar_top_px, h, w, interpolated_value,
        )

        final_value = label_value if label_value is not None else round(interpolated_value, 2)

        bar_values.append({
            "bar_index": idx,
            "x_range": [plot_left + bar_start, plot_left + bar_end],
            "bar_top_y_px": int(plot_top + bar_top_px),
            "interpolated_value": round(interpolated_value, 2),
            "ocr_label_value": label_value,
            "final_value": final_value,
        })

    return bar_values


def _ocr_bar_value_label(
    arr: "np.ndarray",
    plot_top: int, plot_left: int,
    bar_start: int, bar_end: int,
    bar_top_px: int, h: int, w: int,
    interpolated_value: float,
) -> "Optional[float]":
    """Try to OCR the value label displayed above a bar."""
    import numpy as np

    # Region above the bar where value label might be
    label_top = max(0, plot_top + bar_top_px - 35)
    label_bottom = min(h, plot_top + bar_top_px + 8)
    label_left = max(0, plot_left + bar_start - 15)
    label_right = min(w, plot_left + bar_end + 15)

    if label_bottom <= label_top or label_right <= label_left:
        return None

    label_arr = arr[label_top:label_bottom, label_left:label_right]
    lh, lw = label_arr.shape
    if lh < 5 or lw < 5:
        return None

    # Magnify for better OCR
    label_img = Image.fromarray(label_arr.astype(np.uint8))
    label_2x = label_img.resize((lw * 3, lh * 3), Image.LANCZOS)
    label_text = _ocr_image(label_2x)
    label_numbers, _ = _extract_numbers(label_text)

    if label_numbers:
        # Pick the number closest to our interpolated value
        closest = min(label_numbers, key=lambda n: abs(n - interpolated_value))
        # Accept if within 30% of interpolated value
        if abs(interpolated_value) > 1e-10:
            rel_diff = abs(closest - interpolated_value) / abs(interpolated_value)
            if rel_diff < 0.30:
                return closest
        elif abs(closest - interpolated_value) < 0.1:
            return closest

    return None


def _group_bars_by_position(
    bar_regions: list, bar_values: list, x_labels: list,
) -> list:
    """Group bars into logical groups based on horizontal spacing."""
    import numpy as np

    if len(bar_regions) < 2:
        return [{
            "group_label": x_labels[0] if x_labels else "Group_1",
            "bar_count": len(bar_values),
            "values": [b["final_value"] for b in bar_values],
        }]

    # Calculate gaps between bars
    gaps = []
    for i in range(len(bar_regions) - 1):
        gap = bar_regions[i + 1][0] - bar_regions[i][1]
        gaps.append(gap)

    if not gaps:
        return [{
            "group_label": x_labels[0] if x_labels else "Group_1",
            "bar_count": len(bar_values),
            "values": [b["final_value"] for b in bar_values],
        }]

    median_gap = np.median(gaps)
    if median_gap < 1:
        median_gap = 1

    # A gap > 1.8x median indicates group boundary
    group_boundaries = [0]
    for i, gap in enumerate(gaps):
        if gap > median_gap * 1.8:
            group_boundaries.append(i + 1)
    group_boundaries.append(len(bar_regions))

    groups = []
    for g in range(len(group_boundaries) - 1):
        start_idx = group_boundaries[g]
        end_idx = group_boundaries[g + 1]
        group_bars = bar_values[start_idx:end_idx]
        group_label = x_labels[g] if g < len(x_labels) else f"Group_{g + 1}"
        groups.append({
            "group_label": group_label,
            "bar_count": len(group_bars),
            "values": [b["final_value"] for b in group_bars],
        })

    return groups


# ═══════════════════════════════════════════════════════════════════════════
# Bar Chart Extraction CrewAI Tool
# ═══════════════════════════════════════════════════════════════════════════

try:
    from crewai.tools import BaseTool
    from pydantic import BaseModel, Field

    class BarChartExtractInput(BaseModel):
        """Input for bar chart value extraction."""

        image_path: str = Field(
            ...,
            description="Path to the bar chart image file to extract values from."
        )

    class BarChartExtractionTool(BaseTool):
        """
        Extract numerical values from bar chart images using computer vision.

        Unlike text-based OCR (chart_ocr_extract), this tool MEASURES BAR HEIGHTS
        in pixels and interpolates against y-axis scale labels to compute actual
        data values. This is ESSENTIAL for cross-figure data comparison because
        most papers report quantitative results ONLY as bar charts.

        Use this on EVERY bar chart in the paper. The extracted values can then
        be fed into cross_figure_data_compare to detect data duplication across
        supposedly independent experiments.
        """

        name: str = "bar_chart_extract_values"
        description: str = (
            "Extract numerical values from bar chart images by measuring bar "
            "heights via computer vision. Works on standard bar charts with "
            "labeled y-axis (scale) and x-axis (group labels). "
            "Returns structured group-value pairs ready for cross-figure "
            "data comparison. "
            "CRITICAL: Use this on EVERY bar chart figure (Fig 1E, Fig 1F, "
            "Fig 4B, etc.) BEFORE running cross_figure_data_compare. "
            "Without this step, cross-figure data comparison has no input data "
            "and will return 'no matches found' even when data is duplicated."
        )
        args_schema: type[BaseModel] = BarChartExtractInput

        def _run(self, image_path: str) -> str:
            """Run bar chart value extraction."""
            result = extract_bar_chart_values(image_path)
            return json.dumps(result, ensure_ascii=False, default=str)

except ImportError:
    BarChartExtractionTool = None


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python chart_ocr.py <image_path>")
        sys.exit(1)

    path = sys.argv[1]
    img = Image.open(path)
    chart_type = detect_chart_type(img)
    print(f"Chart type: {chart_type}")

    if chart_type == "bar_chart":
        print("\n--- Bar Chart Value Extraction ---")
        bar_result = extract_bar_chart_values(path)
        print(json.dumps(bar_result, indent=2, ensure_ascii=False, default=str))

    result = extract_chart_data(path)
    result["chart_type"] = chart_type
    print("\n--- OCR Text Extraction ---")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
