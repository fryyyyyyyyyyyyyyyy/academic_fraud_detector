"""
Figure panel splitter — decompose composite academic figures into individual panels.

Academic papers often present multi-panel figures (e.g. Figure 1 with sub-panels
1A, 1B, 1C, 1D arranged in a grid). The existing cross-image comparison tools
compare whole composite images, missing partial reuse of individual panels.

This module provides:
1. `split_composite_figure()` — automatic detection of panel boundaries and
   decomposition of a composite figure into individual sub-panel images.
2. `detect_panel_grid()` — detect the grid layout (rows × cols) of a composite
   figure without actually splitting.
3. `extract_all_panels_from_pdf()` — pipeline that extracts images from PDF
   and splits composites into panels in one pass.

Algorithm:
- Horizontal & vertical projection: sum pixel intensities along each axis
  to find content regions separated by white gaps.
- Adaptive gap detection using Otsu thresholding on the projection profile.
- Connected component analysis as fallback for irregular layouts.
"""

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, List, Dict, Any, Optional, Tuple

from PIL import Image

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

# Default minimum panel dimension in pixels (panels smaller than this are ignored)
DEFAULT_MIN_PANEL_SIZE = 80

# Default gap width as fraction of image dimension
DEFAULT_GAP_FRACTION = 0.008  # gaps wider than 0.8% of image are panel separators


def split_composite_figure(
    image: Image.Image,
    min_panel_size: int = DEFAULT_MIN_PANEL_SIZE,
    gap_fraction: float = DEFAULT_GAP_FRACTION,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """
    Split a composite academic figure into its constituent panels.

    Uses projection profile analysis to detect white gaps between panels.
    Works with both grayscale and color images.

    Args:
        image: PIL Image object of the composite figure.
        min_panel_size: Minimum width/height in pixels for a detected panel.
        gap_fraction: Gaps wider than (gap_fraction * image_dimension) pixels
                      are considered panel separators.
        debug: If True, returns additional diagnostic info.

    Returns:
        List of dicts, each representing a detected panel:
        - 'bbox': (x1, y1, x2, y2) pixel coordinates
        - 'panel_image': PIL Image of the cropped panel
        - 'width', 'height': pixel dimensions
        - 'grid_position': (row, col) 0-indexed in the detected grid
        - 'area_fraction': fraction of total image area this panel occupies
    """
    import numpy as np

    if image.mode not in ("L", "RGB"):
        image = image.convert("RGB")

    gray = image.convert("L")
    arr = np.array(gray, dtype=np.uint8)
    h, w = arr.shape

    # ── Binarize: determine background type + content mask ──
    # Two modes:
    #   Light background (e.g., HE staining on white): content = dark pixels
    #   Dark background (e.g., immunofluorescence on black): content = bright pixels
    from collections import Counter
    hist = Counter(arr.flatten())
    total = h * w

    # Measure dark-pixel vs light-pixel dominance
    dark_range = range(0, 51)
    light_range = range(200, 256)
    dark_pixel_count = sum(hist.get(v, 0) for v in dark_range)
    light_pixel_count = sum(hist.get(v, 0) for v in light_range)
    dark_fraction = dark_pixel_count / total
    light_fraction = light_pixel_count / total

    if dark_fraction > 0.30 and dark_fraction > light_fraction:
        # ── Dark background mode (immunofluorescence, fluorescent microscopy) ──
        _is_dark_bg = True
        # Find the dominant dark background color (peak in 0-50 range)
        bg_range_dark = range(0, 51)
        bg_max_count_dark = max(hist.get(v, 0) for v in bg_range_dark)
        bg_threshold = next((v for v in bg_range_dark if hist.get(v, 0) == bg_max_count_dark), 5)
        # Content pixels are BRIGHTER than the dark background
        content_mask = arr > (bg_threshold + 20)
        logger.debug(
            f"Dark-background mode: dark_frac={dark_fraction:.3f}, "
            f"light_frac={light_fraction:.3f}, bg_threshold={bg_threshold}"
        )
    else:
        # ── Light background mode (HE staining, Western blots, most figures) ──
        _is_dark_bg = False
        # Find the dominant light background color (peak in 200-255 range)
        bg_max_count_light = max(hist.get(v, 0) for v in light_range)
        bg_threshold = next((v for v in light_range if hist.get(v, 0) == bg_max_count_light), 245)
        # Content pixels are DARKER than the light background
        content_mask = arr < (bg_threshold - 15)

    # ── Compute projections ──
    # Horizontal projection: dark pixel count per row
    h_proj = np.sum(content_mask, axis=1).astype(np.float64)
    # Vertical projection: dark pixel count per column
    v_proj = np.sum(content_mask, axis=0).astype(np.float64)

    # ── Find gap positions ──
    # A gap is a contiguous region where the projection is below threshold
    h_gap_threshold = max(w * 0.02, 5)  # at most 2% of width has content
    v_gap_threshold = max(h * 0.02, 5)  # at most 2% of height has content

    min_gap_px = max(int(min(w, h) * gap_fraction), 3)

    def find_gaps(proj, length, threshold, min_gap):
        """Find gap intervals in a 1D projection."""
        gaps = []
        in_gap = False
        gap_start = 0
        for i in range(length):
            if proj[i] <= threshold:
                if not in_gap:
                    gap_start = i
                    in_gap = True
            else:
                if in_gap:
                    gap_end = i
                    if gap_end - gap_start >= min_gap:
                        gaps.append((gap_start, gap_end))
                    in_gap = False
        if in_gap and length - gap_start >= min_gap:
            gaps.append((gap_start, length))
        return gaps

    h_gaps = find_gaps(h_proj, h, h_gap_threshold, min_gap_px)
    v_gaps = find_gaps(v_proj, w, v_gap_threshold, min_gap_px)

    # ── Determine panel boundaries ──
    # Content rows/columns are between gap intervals
    h_boundaries = [0]
    for gs, ge in h_gaps:
        h_boundaries.extend([gs, ge])
    h_boundaries.append(h)
    h_boundaries = sorted(set(h_boundaries))

    v_boundaries = [0]
    for gs, ge in v_gaps:
        v_boundaries.extend([gs, ge])
    v_boundaries.append(w)
    v_boundaries = sorted(set(v_boundaries))

    # ── Extract panels from content regions ──
    panels = []
    panel_idx = 0

    # Pair boundaries into content regions (skip gaps)
    h_content_ranges = []
    for i in range(len(h_boundaries) - 1):
        y1, y2 = h_boundaries[i], h_boundaries[i + 1]
        # Check if this region has significant content
        if y2 - y1 >= min_panel_size and np.mean(h_proj[y1:y2]) > threshold_for_region(h_proj[y1:y2], w):
            h_content_ranges.append((y1, y2))

    v_content_ranges = []
    for i in range(len(v_boundaries) - 1):
        x1, x2 = v_boundaries[i], v_boundaries[i + 1]
        if x2 - x1 >= min_panel_size and np.mean(v_proj[x1:x2]) > threshold_for_region(v_proj[x1:x2], h):
            v_content_ranges.append((x1, x2))

    # If gap detection didn't find clean splits, fall back to equal grid
    if len(h_content_ranges) <= 1 and len(v_content_ranges) <= 1:
        if debug:
            logger.info("Gap detection found no clean splits — trying contour-based fallback")
        return _contour_fallback(image, arr, content_mask, min_panel_size, debug)

    # Sort ranges by position
    h_content_ranges.sort(key=lambda r: r[0])
    v_content_ranges.sort(key=lambda r: r[0])

    if debug:
        logger.info(
            f"Detected grid: {len(h_content_ranges)} rows × {len(v_content_ranges)} cols "
            f"(h_gaps={len(h_gaps)}, v_gaps={len(v_gaps)})"
        )

    # Extract each panel
    for row_idx, (y1, y2) in enumerate(h_content_ranges):
        for col_idx, (x1, x2) in enumerate(v_content_ranges):
            # Trim whitespace within each panel cell
            cell = arr[y1:y2, x1:x2]
            cell_content = content_mask[y1:y2, x1:x2]
            trim_bbox = _trim_whitespace(cell, cell_content)

            if trim_bbox is None:
                continue  # empty panel cell

            tx1, ty1, tx2, ty2 = trim_bbox
            panel_x1 = x1 + tx1
            panel_y1 = y1 + ty1
            panel_x2 = x1 + tx2
            panel_y2 = y1 + ty2

            panel_w = panel_x2 - panel_x1
            panel_h = panel_y2 - panel_y1

            if panel_w < min_panel_size or panel_h < min_panel_size:
                continue

            panel_img = image.crop((panel_x1, panel_y1, panel_x2, panel_y2))
            panels.append({
                "bbox": (panel_x1, panel_y1, panel_x2, panel_y2),
                "panel_image": panel_img,
                "width": panel_w,
                "height": panel_h,
                "grid_position": (row_idx, col_idx),
                "area_fraction": round((panel_w * panel_h) / (w * h), 4),
                "panel_index": panel_idx,
            })
            panel_idx += 1

    if debug and not panels:
        logger.warning(f"Split found {len(h_content_ranges)}×{len(v_content_ranges)} grid but no panels > {min_panel_size}px")

    return panels


def _contour_fallback(
    image: Image.Image,
    arr: "np.ndarray",
    content_mask: "np.ndarray",
    min_panel_size: int,
    debug: bool,
) -> List[Dict[str, Any]]:
    """
    Fallback: use contour detection to find individual panel regions.
    This handles irregular layouts and non-grid arrangements.
    """
    import numpy as np

    h, w = arr.shape

    # Dilate to merge nearby content regions
    from scipy.ndimage import binary_dilation, binary_fill_holes

    kernel = np.ones((7, 7), dtype=bool)
    dilated = binary_dilation(content_mask, structure=kernel, iterations=3)
    filled = binary_fill_holes(dilated)

    # Find connected components
    from scipy.ndimage import label, find_objects

    labeled, num_features = label(filled)
    slices = find_objects(labeled)

    panels = []
    panel_idx = 0

    for i, sl in enumerate(slices):
        if sl is None:
            continue
        y1, y2 = sl[0].start, sl[0].stop
        x1, x2 = sl[1].start, sl[1].stop

        panel_w = x2 - x1
        panel_h = y2 - y1

        if panel_w < min_panel_size or panel_h < min_panel_size:
            continue

        # Skip if panel area is > 95% of the whole image (not a real panel)
        area_frac = (panel_w * panel_h) / (w * h)
        if area_frac > 0.90:
            # This is the whole image — try to find sub-panels within
            continue

        panel_img = image.crop((x1, y1, x2, y2))
        panels.append({
            "bbox": (x1, y1, x2, y2),
            "panel_image": panel_img,
            "width": panel_w,
            "height": panel_h,
            "grid_position": (0, panel_idx),
            "area_fraction": round(area_frac, 4),
            "panel_index": panel_idx,
            "method": "contour_fallback",
        })
        panel_idx += 1

    if debug and panels:
        logger.info(f"Contour fallback found {len(panels)} panels")
    elif debug:
        logger.warning("Contour fallback also found no panels — returning whole image as single panel")

    # If still nothing, return the whole image as a single panel
    if not panels:
        panels.append({
            "bbox": (0, 0, w, h),
            "panel_image": image.copy(),
            "width": w,
            "height": h,
            "grid_position": (0, 0),
            "area_fraction": 1.0,
            "panel_index": 0,
            "method": "whole_image_fallback",
        })

    return panels


def threshold_for_region(proj_slice: "np.ndarray", total_dim: int) -> float:
    """Determine a sensible threshold for whether a projection slice contains real content."""
    # If max projection is less than 1% of the dimension, it's a gap
    return max(total_dim * 0.005, 3.0)


def _trim_whitespace(
    gray_arr: "np.ndarray",
    content_mask: "np.ndarray",
    margin: int = 5,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Trim whitespace padding from a panel region. Returns (x1, y1, x2, y2)
    relative to the input array, or None if the region is empty.
    """
    import numpy as np

    h, w = gray_arr.shape
    rows_with_content = np.any(content_mask, axis=1)
    cols_with_content = np.any(content_mask, axis=0)

    if not np.any(rows_with_content) or not np.any(cols_with_content):
        return None

    y1 = int(np.argmax(rows_with_content))
    y2 = int(h - np.argmax(rows_with_content[::-1]))
    x1 = int(np.argmax(cols_with_content))
    x2 = int(w - np.argmax(cols_with_content[::-1]))

    # Add small margin
    y1 = max(0, y1 - margin)
    y2 = min(h, y2 + margin)
    x1 = max(0, x1 - margin)
    x2 = min(w, x2 + margin)

    return (x1, y1, x2, y2)


def detect_panel_grid(
    image: Image.Image,
    min_panel_size: int = DEFAULT_MIN_PANEL_SIZE,
    gap_fraction: float = DEFAULT_GAP_FRACTION,
) -> Dict[str, Any]:
    """
    Detect the panel grid layout of a composite figure without extracting panels.

    Returns metadata about the detected grid.

    Args:
        image: PIL Image of the composite figure.
        min_panel_size: Minimum panel dimension.
        gap_fraction: Gap detection sensitivity.

    Returns:
        Dict with 'rows', 'cols', 'total_panels', 'layout' description.
    """
    panels = split_composite_figure(
        image,
        min_panel_size=min_panel_size,
        gap_fraction=gap_fraction,
        debug=False,
    )

    if not panels:
        return {"rows": 1, "cols": 1, "total_panels": 1, "layout": "single"}

    # Determine grid dimensions from panel positions
    max_row = max(p["grid_position"][0] for p in panels) + 1
    max_col = max(p["grid_position"][1] for p in panels) + 1

    return {
        "rows": max_row,
        "cols": max_col,
        "total_panels": len(panels),
        "layout": f"{max_row}×{max_col}",
        "panel_sizes": [
            {"width": p["width"], "height": p["height"]} for p in panels
        ],
    }


def _split_images_to_panels(
    images: List[Dict[str, Any]],
    output_dir: str | Path,
    min_panel_size: int = DEFAULT_MIN_PANEL_SIZE,
    gap_fraction: float = DEFAULT_GAP_FRACTION,
) -> List[Dict[str, Any]]:
    """Split already-extracted figure images into panels without extracting from PDF."""
    output_dir = Path(output_dir)
    panels_dir = output_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    all_panels = []

    for image_idx, img_meta in enumerate(images):
        img_path = img_meta.get("filepath")
        if not img_path:
            all_panels.append({
                **img_meta,
                "panels": [],
                "is_composite": False,
                "panel_count": 0,
                "grid": "1×1",
                "error": "Image metadata missing filepath.",
            })
            continue

        try:
            with Image.open(img_path) as opened:
                img = opened.copy()
        except Exception as e:
            logger.warning(f"Could not open image for panel splitting: {img_path} — {e}")
            all_panels.append({
                **img_meta,
                "panels": [],
                "is_composite": False,
                "panel_count": 0,
                "grid": "1×1",
                "error": str(e),
            })
            continue

        # Detect and split panels
        try:
            panels = split_composite_figure(
                img,
                min_panel_size=min_panel_size,
                gap_fraction=gap_fraction,
                debug=False,
            )
        except Exception as e:
            logger.warning(f"Panel splitting failed for {img_path}: {e}")
            panels = []

        # Save panel images
        saved_panels = []
        base_filename = img_meta.get("filename") or Path(str(img_path)).name or f"image_{image_idx + 1}"
        base_name = Path(str(base_filename)).stem or f"image_{image_idx + 1}"
        for p in panels:
            panel_filename = f"{base_name}_panel_{p['panel_index']}.png"
            panel_path = panels_dir / panel_filename
            counter = 1
            while panel_path.exists():
                panel_filename = f"{base_name}_panel_{p['panel_index']}_{counter}.png"
                panel_path = panels_dir / panel_filename
                counter += 1

            # Save panel image
            p["panel_image"].save(panel_path, format="PNG")

            saved_panels.append({
                "panel_index": p["panel_index"],
                "grid_position": p["grid_position"],
                "bbox": p["bbox"],
                "width": p["width"],
                "height": p["height"],
                "filepath": str(panel_path.absolute()),
                "filename": panel_filename,
                "pdf_page": img_meta.get("pdf_page") or img_meta.get("page_number"),
                "page_number": img_meta.get("page_number") or img_meta.get("pdf_page"),
                "source": img_meta.get("source"),
                "source_image_filename": img_meta.get("filename"),
            })

        is_composite = len(panels) > 1
        all_panels.append({
            **img_meta,
            "panels": saved_panels,
            "is_composite": is_composite,
            "panel_count": len(saved_panels),
            "grid": f"{max((p['grid_position'][0] for p in panels), default=0) + 1}×"
                    f"{max((p['grid_position'][1] for p in panels), default=0) + 1}"
            if panels else "1×1",
        })

    total_panels = sum(entry.get("panel_count", 0) for entry in all_panels)
    composite_count = sum(1 for entry in all_panels if entry.get("is_composite"))
    logger.info(
        f"Panel extraction complete: {len(images)} figures → {total_panels} panels "
        f"({composite_count} composite figures split)"
    )

    return all_panels


def extract_all_panels_from_images(
    images: List[Dict[str, Any]],
    output_dir: Optional[str] = None,
    min_panel_size: int = DEFAULT_MIN_PANEL_SIZE,
    gap_fraction: float = DEFAULT_GAP_FRACTION,
) -> List[Dict[str, Any]]:
    """
    Split already-extracted figure images into panels.

    This reuses MinerU/PyMuPDF image metadata and never extracts images from the
    source PDF, preventing historical extracted_images content from being mixed in.
    """
    if output_dir is None:
        from ..utils.text_extraction import create_unique_image_output_dir
        source_name = None
        if images:
            source_name = images[0].get("filename") or images[0].get("filepath")
        output_dir = create_unique_image_output_dir(prefix="image_panels", source_name=source_name)

    return _split_images_to_panels(
        images,
        output_dir=output_dir,
        min_panel_size=min_panel_size,
        gap_fraction=gap_fraction,
    )


def extract_all_panels_from_pdf(
    pdf_path: str,
    output_dir: Optional[str] = None,
    min_panel_size: int = DEFAULT_MIN_PANEL_SIZE,
    min_image_size: int = 80,
    gap_fraction: float = DEFAULT_GAP_FRACTION,
    max_pages: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Complete pipeline: extract embedded images from a PDF, then split each
    composite figure into individual panels.

    Args:
        pdf_path: Path to the local PDF file.
        output_dir: Directory to save extracted images and panel images. If None,
            creates a unique cache directory for this extraction.
        min_panel_size: Minimum panel dimension in pixels.
        min_image_size: Minimum image dimension to keep from PDF extraction.
        gap_fraction: Gap detection sensitivity.
        max_pages: Maximum PDF pages to process.

    Returns:
        List of panel metadata dicts. Each dict includes:
        - All fields from extract_pdf_images (source page, etc.)
        - 'panels': list of sub-panel dicts
        - 'is_composite': True if figure was split into multiple panels
    """
    from ..utils.text_extraction import create_unique_image_output_dir, extract_pdf_images_from_file

    if output_dir is None:
        output_dir = create_unique_image_output_dir(prefix="pdf_panels", source_name=Path(pdf_path).name)

    # Extract images from PDF into this extraction's isolated directory.
    images = extract_pdf_images_from_file(
        pdf_path,
        output_dir=str(output_dir),
        min_size=min_image_size,
        max_pages=max_pages,
    )

    return _split_images_to_panels(
        images,
        output_dir=output_dir,
        min_panel_size=min_panel_size,
        gap_fraction=gap_fraction,
    )


def save_panels_for_tools(
    panels_data: List[Dict[str, Any]],
    output_dir: str,
) -> List[str]:
    """
    Convert panel extraction results into a flat list of panel image paths,
    suitable for passing to cross_image_duplicate_check and other tools.

    Args:
        panels_data: Output from extract_all_panels_from_pdf().
        output_dir: Directory for saved panel images.

    Returns:
        Flat list of absolute file paths to all panel images.
    """
    flat_paths = []
    for figure_entry in panels_data:
        for panel in figure_entry.get("panels", []):
            if "filepath" in panel:
                flat_paths.append(str(Path(panel["filepath"]).absolute()))
            elif "panel_image" in panel:
                # Save if not already saved
                save_dir = Path(output_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                fname = (
                    f"{os.path.splitext(figure_entry.get('filename', 'unknown'))[0]}"
                    f"_panel_{panel['panel_index']}.png"
                )
                fpath = save_dir / fname
                panel["panel_image"].save(fpath, format="PNG")
                flat_paths.append(str(fpath.absolute()))

    return flat_paths


# ═══════════════════════════════════════════════════════════════════════════
# Quick test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python figure_splitter.py <image_path_or_pdf_path>")
        sys.exit(1)

    path = sys.argv[1]
    if path.lower().endswith(".pdf"):
        panels = extract_all_panels_from_pdf(path, debug=True)
        for entry in panels:
            print(f"\nFigure: {entry['filename']} (page {entry['page_number']})")
            print(f"  Composite: {entry['is_composite']}")
            print(f"  Panels: {entry['panel_count']}")
            print(f"  Grid: {entry['grid']}")
            for p in entry.get("panels", []):
                print(f"    Panel {p['panel_index']}: {p['width']}×{p['height']} @ {p['bbox']}")
    else:
        img = Image.open(path)
        print(f"Image: {img.size}")
        panels = split_composite_figure(img, debug=True)
        print(f"Panels detected: {len(panels)}")
        for p in panels:
            print(f"  Panel {p['panel_index']}: {p['width']}×{p['height']} "
                  f"@ {p['bbox']}  grid={p['grid_position']}")
