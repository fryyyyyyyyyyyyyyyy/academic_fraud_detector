"""
Image forensics tools — detect manipulation in academic paper figures.

Three analysis dimensions:
1. ELA (Error Level Analysis): detects edited regions via JPEG compression artifacts.
2. Clone Detection: finds copy-moved regions within an image (common in blots/gels).
3. AI Image Detection: flags potential AI-generated/synthetic images.

All tools accept image URLs and return structured JSON findings.
"""

import json
import os
import logging
from io import BytesIO
from typing import Optional

from PIL import Image
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from ..utils.image_downloader import load_image

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Error Level Analysis (ELA)
# ═══════════════════════════════════════════════════════════════════════

class ELAInput(BaseModel):
    """Input for Error Level Analysis."""

    image_path_or_url: str = Field(..., description="Local file path or URL of the image to analyze.")
    quality: int = Field(
        default=90,
        ge=10,
        le=100,
        description="JPEG re-save quality for ELA. Lower = more sensitive to edits.",
    )


class ELATool(BaseTool):
    """
    Error Level Analysis (ELA) for detecting image manipulation.

    How it works:
    1. Download the image.
    2. Re-save it as JPEG at a specified quality.
    3. Compute the pixel-level difference between original and re-saved.
    4. Edited regions typically have different compression histories and show
       higher error levels than authentic regions.

    Common use cases:
    - Detecting spliced western blot bands
    - Finding copy-pasted cells in microscopy images
    - Identifying Photoshopped data visualizations
    """

    name: str = "error_level_analysis"
    description: str = (
        "Perform Error Level Analysis (ELA) on a scientific figure. ELA reveals "
        "regions that have been edited by analyzing JPEG compression error levels. "
        "Edited regions typically show different error levels (brighter in ELA) than "
        "the unedited background. Returns error statistics and flags anomalous regions. "
        "Use this on ALL figures in a suspect paper."
    )
    args_schema: type[BaseModel] = ELAInput

    def _run(self, image_path_or_url: str, quality: int = 90) -> str:
        """Execute ELA on the image."""
        try:
            img, meta = load_image(image_path_or_url, use_cache=True)
            if img is None:
                return json.dumps({
                    "error": meta.get("error", "Failed to download image"),
                    "flagged": False,
                })
        except Exception as e:
            return json.dumps({"error": str(e), "flagged": False})

        try:
            # Convert to RGB if not already
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            # Save at specified quality
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)

            from PIL import Image as PILImage
            recompressed = PILImage.open(buffer)

            # Compute absolute difference
            from PIL import ImageChops, ImageStat
            diff = ImageChops.difference(img, recompressed)
            extrema = diff.getextrema()
            stat = ImageStat.Stat(diff)

            # Per-channel statistics
            channels = ["R", "G", "B"] if len(stat.mean) >= 3 else ["L"]
            channel_stats = {}
            for i, ch in enumerate(channels[:min(3, len(stat.mean))]):
                channel_stats[ch] = {
                    "mean_error": round(stat.mean[i], 2),
                    "std_error": round(stat.stddev[i], 2),
                    "max_error": round(extrema[i][1], 2) if i < len(extrema) else 0,
                }

            # Overall metrics
            mean_error = sum(stat.mean) / len(stat.mean)
            max_error = max(e[1] for e in extrema) if extrema else 0
            std_error = sum(stat.stddev) / len(stat.stddev)

            # Heuristic thresholds (empirically calibrated)
            suspicious = mean_error > 15.0 or max_error > 80.0 or std_error > 20.0

            return json.dumps({
                "analysis_type": "Error Level Analysis (ELA)",
                "image_source": image_path_or_url,
                "image_dimensions": f"{meta.get('width')}×{meta.get('height')}",
                "image_format": meta.get("format"),
                "ela_quality_used": quality,
                "overall_stats": {
                    "mean_error": round(mean_error, 2),
                    "max_error": round(max_error, 2),
                    "std_error": round(std_error, 2),
                },
                "per_channel_stats": channel_stats,
                "suspicious_regions_detected": suspicious,
                "flagged": suspicious,
                "interpretation": (
                    "Potentially edited: high error levels suggest regions with different "
                    "compression history (possible splices, insertions, or erasures). "
                    "Manual review of the ELA difference image is recommended."
                    if suspicious
                    else "No significant ELA anomalies detected. Image appears consistent."
                ),
            })

        except Exception as e:
            logger.error(f"ELA analysis failed: {e}")
            return json.dumps({"error": str(e), "flagged": False})


# ═══════════════════════════════════════════════════════════════════════
# Clone Detection
# ═══════════════════════════════════════════════════════════════════════

class CloneDetectionInput(BaseModel):
    """Input for clone detection."""

    image_path_or_url: str = Field(..., description="Local file path or URL of the image to check for cloned regions.")
    block_size: int = Field(
        default=32,
        ge=8,
        le=128,
        description="Block size for clone detection in pixels. Smaller = finer detection but slower.",
    )


class CloneDetectionTool(BaseTool):
    """
    Detect cloned/duplicated regions within an image (copy-move forgery).

    How it works:
    1. Convert image to grayscale.
    2. Divide into overlapping blocks.
    3. Compute perceptual hash (average hash) for each block.
    4. Blocks with identical hashes that are spatially separated are flagged.

    This is especially effective for:
    - Duplicated western blot bands
    - Copy-pasted cells or tissue regions
    - Reproduced graph elements
    """

    name: str = "clone_detection"
    description: str = (
        "Detect cloned/duplicated regions within a scientific image. Used to find "
        "copy-move forgery where parts of an image are copied and pasted elsewhere — "
        "common in western blots, microscopy, and flow cytometry figures. "
        "Uses block-level perceptual hashing to find identical regions. "
        "Use this after ELA to confirm specific manipulation types."
    )
    args_schema: type[BaseModel] = CloneDetectionInput

    def _run(self, image_path_or_url: str, block_size: int = 32) -> str:
        """Execute clone detection on the image."""
        try:
            img, meta = load_image(image_path_or_url, use_cache=True)
            if img is None:
                return json.dumps({"error": meta.get("error"), "flagged": False})
        except Exception as e:
            return json.dumps({"error": str(e), "flagged": False})

        try:
            gray = img.convert("L")
            w, h = gray.size

            if w < block_size * 2 or h < block_size * 2:
                return json.dumps({
                    "error": f"Image too small for clone detection (min: {block_size * 2}px).",
                    "flagged": False,
                })

            pixels = list(gray.getdata())
            hashes: dict[str, tuple] = {}
            clones = []

            step = max(block_size // 2, 1)  # 50% overlap, min step 1px
            for y in range(0, h - block_size + 1, step):
                for x in range(0, w - block_size + 1, step):
                    # Extract block and compute average hash
                    block_pixels = []
                    for by in range(block_size):
                        for bx in range(block_size):
                            idx = (y + by) * w + (x + bx)
                            block_pixels.append(pixels[idx])

                    avg = sum(block_pixels) / len(block_pixels)
                    ahash = "".join("1" if p > avg else "0" for p in block_pixels)

                    if ahash in hashes:
                        prev_x, prev_y = hashes[ahash]
                        distance = ((x - prev_x) ** 2 + (y - prev_y) ** 2) ** 0.5
                        if distance > block_size * 1.5:  # Not adjacent
                            clones.append({
                                "region1": [prev_x, prev_y],
                                "region2": [x, y],
                                "distance_px": round(distance),
                            })
                    else:
                        hashes[ahash] = (x, y)

                    # Safety: don't let hash dict grow unbounded
                    if len(hashes) > 500000:
                        break
                if len(hashes) > 500000:
                    break

            return json.dumps({
                "analysis_type": "Clone Detection (Copy-Move Forgery)",
                "image_source": image_path_or_url,
                "image_dimensions": f"{w}×{h}",
                "block_size": block_size,
                "blocks_analyzed": len(hashes),
                "clone_regions_detected": len(clones),
                "flagged": len(clones) > 0,
                "clones": clones[:30],
                "interpretation": (
                    f"Found {len(clones)} potential clone region(s). "
                    "Copy-move forgery suspected — manual review required."
                    if clones
                    else "No cloned regions detected."
                ),
            })

        except Exception as e:
            logger.error(f"Clone detection failed: {e}")
            return json.dumps({"error": str(e), "flagged": False})


# ═══════════════════════════════════════════════════════════════════════
# AI Image Detection
# ═══════════════════════════════════════════════════════════════════════

class AIImageDetectionInput(BaseModel):
    """Input for AI image detection."""

    image_path_or_url: str = Field(..., description="Local file path or URL of the image to check for AI generation.")


class AIImageDetectionTool(BaseTool):
    """
    Detect whether a scientific figure is AI-generated (DALL-E, Midjourney, SDXL, etc.).

    AI-generated figures in academic papers are a red flag because:
    - They may depict fabricated experimental results.
    - They often contain subtle artifacts (impossible equipment, nonsensical labels).
    - Their presence suggests the authors did not perform real experiments.

    Current implementation uses noise residual analysis + frequency domain checks.
    For production use, integrate with a dedicated model (e.g., DIRE, NPR, SDXL-Detector).
    """

    name: str = "ai_image_detection"
    description: str = (
        "Check whether a scientific figure appears to be AI-generated (DALL-E, Midjourney, "
        "Stable Diffusion, etc.). Analyzes noise patterns and statistical artifacts common "
        "in synthetic images. AI-generated figures in papers are a strong red flag for fraud."
    )
    args_schema: type[BaseModel] = AIImageDetectionInput

    def _run(self, image_path_or_url: str) -> str:
        """Execute AI image detection."""
        try:
            img, meta = load_image(image_path_or_url, use_cache=True)
            if img is None:
                return json.dumps({"error": meta.get("error"), "flagged": False})
        except Exception as e:
            return json.dumps({"error": str(e), "flagged": False})

        try:
            import numpy as np

            gray = img.convert("L")
            arr = np.array(gray, dtype=np.float64)

            # ── Noise residual analysis ──
            # Compute local variance (noise texture)
            from scipy.ndimage import uniform_filter

            mean = uniform_filter(arr, size=5)
            mean_sq = uniform_filter(arr ** 2, size=5)
            local_var = mean_sq - mean ** 2
            noise_std = float(np.sqrt(np.mean(local_var)))

            # ── Frequency domain check ──
            # AI images often have distinctive frequency signatures
            fft = np.fft.fft2(arr)
            fft_shifted = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shifted)

            # High-frequency energy ratio
            h, w = magnitude.shape
            center_y, center_x = h // 2, w // 2
            radius = min(h, w) // 4
            y_indices, x_indices = np.ogrid[:h, :w]
            dist = np.sqrt((y_indices - center_y) ** 2 + (x_indices - center_x) ** 2)

            high_freq_mask = dist > radius
            high_freq_energy = float(np.sum(magnitude[high_freq_mask]))
            total_energy = float(np.sum(magnitude))
            high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0

            # Heuristic: AI images often have unusually low high-freq energy
            # (overly smooth) or specific noise signatures
            is_suspicious = (
                high_freq_ratio < 0.15  # unusually smooth (common in AI images)
                or noise_std < 5.0       # very low noise
            )

            return json.dumps({
                "analysis_type": "AI Image Detection (Noise + Frequency Domain)",
                "image_source": image_path_or_url,
                "image_dimensions": f"{meta.get('width')}×{meta.get('height')}",
                "noise_std": round(noise_std, 4),
                "high_frequency_energy_ratio": round(high_freq_ratio, 4),
                "suspicious_for_ai_generation": is_suspicious,
                "flagged": is_suspicious,
                "note": (
                    "This is a heuristic analysis. For definitive AI image detection, "
                    "integrate with a dedicated model (DIRE, SDXL-Detector, or NPR). "
                    "Low high-frequency energy + low noise can also indicate heavy compression "
                    "or simple vector graphics — not definitive proof of AI generation."
                ),
                "recommendation": (
                    "Image shows suspicious noise/frequency patterns consistent with AI "
                    "generation. Recommend cross-referencing with a specialized detector "
                    "and manual review of figure details (labels, equipment depictions)."
                    if is_suspicious
                    else "No AI generation artifacts detected with current heuristics."
                ),
            })

        except Exception as e:
            logger.error(f"AI image detection failed: {e}")
            return json.dumps({"error": str(e), "flagged": False})


# ═══════════════════════════════════════════════════════════════════════
# Cross-Image Duplicate Detection
# ═══════════════════════════════════════════════════════════════════════

class CrossImageDuplicateInput(BaseModel):
    """Input for cross-image duplicate detection."""

    image_paths: str = Field(
        ...,
        description=(
            "JSON-encoded list of image file paths to compare. "
            'Example: \'["/path/to/fig1.png", "/path/to/fig2.png"]\'.'
        ),
    )
    hamming_threshold: int = Field(
        default=10,
        ge=0,
        le=64,
        description="Maximum Hamming distance for two hashes to be considered a match. Lower = stricter.",
    )


class CrossImageDuplicateTool(BaseTool):
    """
    Detect if the same image is reused across different figures, even after
    rotation, flipping, or mild cropping.

    How it works:
    1. Compute a DCT-based perceptual hash (pHash) for each image.
    2. For each image, compute 7 variant hashes:
       - original, rotated 90°, 180°, 270°
       - flipped horizontally, flipped vertically
       - flipped horizontally + rotated 90°
    3. Compare all pairs of (image, variant) hashes using Hamming distance.
    4. Flag pairs with distance below threshold.

    This catches the most common image reuse pattern in fraudulent papers:
    the same Western blot / microscopy image presented as different experiments
    after rotation or mirroring.
    """

    name: str = "cross_image_duplicate_check"
    description: str = (
        "Compare ALL images extracted from a paper to detect if the same image "
        "is reused across different figures — even after rotation (90/180/270 deg) "
        "or flipping (horizontal/vertical). Uses DCT perceptual hashing with "
        "Hamming distance comparison. Catches the classic fraud pattern where "
        "a single Western blot or microscopy image is rotated/flipped and "
        "presented as multiple different experimental results. "
        "Input: a JSON list of image file paths. "
        "Returns: matched pairs with Hamming distance and detected transformation."
    )
    args_schema: type[BaseModel] = CrossImageDuplicateInput

    def _run(self, image_paths: str, hamming_threshold: int = 10) -> str:
        """Execute cross-image duplicate detection."""
        try:
            paths = json.loads(image_paths)
            if not isinstance(paths, list) or len(paths) < 2:
                return json.dumps({
                    "error": "Need at least 2 image paths as a JSON list.",
                    "flagged": False,
                })
        except json.JSONDecodeError:
            return json.dumps({
                "error": "image_paths must be a valid JSON list of strings.",
                "flagged": False,
            })

        try:
            import numpy as np
            from scipy.fft import dct
        except ImportError as e:
            return json.dumps({"error": f"Required library not available: {e}", "flagged": False})

        images_loaded = []
        for path in paths:
            img, meta = load_image(path, use_cache=True)
            if img is None:
                logger.warning(f"Could not load image: {path} — {meta.get('error')}")
                continue
            images_loaded.append({
                "path": path,
                "filename": os.path.basename(path),
                "img": img,
            })

        if len(images_loaded) < 2:
            return json.dumps({
                "analysis_type": "Cross-Image Duplicate Detection",
                "images_compared": len(images_loaded),
                "pairs_checked": 0,
                "flagged": False,
                "matches": [],
                "interpretation": "Not enough valid images to compare.",
            })

        # Compute pHash for each image + all variants
        hashes_by_image = {}
        for entry in images_loaded:
            variants = self._compute_variant_hashes(entry["img"])
            hashes_by_image[entry["filename"]] = variants

        # Compare all pairs
        filenames = list(hashes_by_image.keys())
        matches = []
        for i in range(len(filenames)):
            for j in range(i + 1, len(filenames)):
                variants_a = hashes_by_image[filenames[i]]
                variants_b = hashes_by_image[filenames[j]]

                best_dist = 64
                best_transform = "none"
                for transform_a, hash_a in variants_a.items():
                    for transform_b, hash_b in variants_b.items():
                        dist = self._hamming_distance(hash_a, hash_b)
                        if dist < best_dist:
                            best_dist = dist
                            if transform_a == "original" and transform_b == "original":
                                best_transform = "identical"
                            elif transform_a != "original":
                                best_transform = f"{filenames[i]}:{transform_a}"
                            else:
                                best_transform = f"{filenames[j]}:{transform_b}"

                if best_dist <= hamming_threshold:
                    matches.append({
                        "image_a": filenames[i],
                        "image_b": filenames[j],
                        "hamming_distance": best_dist,
                        "suggested_transformation": best_transform,
                        "confidence": (
                            "very_high" if best_dist <= 3 else
                            "high" if best_dist <= 6 else
                            "medium" if best_dist <= 10 else
                            "low"
                        ),
                    })

        return json.dumps({
            "analysis_type": "Cross-Image Duplicate Detection",
            "images_compared": len(images_loaded),
            "pairs_checked": len(filenames) * (len(filenames) - 1) // 2,
            "flagged": len(matches) > 0,
            "match_count": len(matches),
            "hamming_threshold": hamming_threshold,
            "matches": sorted(matches, key=lambda m: m["hamming_distance"]),
            "interpretation": (
                f"Found {len(matches)} image pair(s) with Hamming distance <= {hamming_threshold}. "
                "These images may be the same image reused with rotation/flipping — "
                "a common fraud pattern in Western blots and microscopy figures."
                if matches
                else "No cross-image duplicates detected. All images appear distinct."
            ),
        })

    def _compute_phash(self, img: "Image.Image") -> str:
        """Compute a DCT-based perceptual hash (64-bit binary string)."""
        import numpy as np
        from scipy.fft import dct

        # Convert to grayscale and resize to 32x32
        gray = img.convert("L")
        resized = gray.resize((32, 32), Image.LANCZOS)
        pixels = np.array(resized, dtype=np.float64)

        # 2D DCT
        dct_coeffs = dct(dct(pixels.T, norm="ortho").T, norm="ortho")

        # Take top-left 8x8 (lowest frequencies)
        low_freq = dct_coeffs[:8, :8]

        # Compare to median (more robust than mean)
        median = np.median(low_freq)
        hash_bits = (low_freq > median).flatten()

        return "".join("1" if b else "0" for b in hash_bits)

    def _compute_variant_hashes(self, img: "Image.Image") -> dict[str, str]:
        """Compute pHash for original + rotated/flipped variants."""
        variants = {}
        variants["original"] = self._compute_phash(img)

        # Rotations
        for angle, label in [(90, "rot_90"), (180, "rot_180"), (270, "rot_270")]:
            rotated = img.rotate(angle, expand=True)
            variants[label] = self._compute_phash(rotated)

        # Flips
        flipped_h = img.transpose(Image.FLIP_LEFT_RIGHT)
        variants["flip_h"] = self._compute_phash(flipped_h)
        flipped_v = img.transpose(Image.FLIP_TOP_BOTTOM)
        variants["flip_v"] = self._compute_phash(flipped_v)

        # Flip + rotate (catches combined transforms)
        variants["flip_h_rot_90"] = self._compute_phash(
            flipped_h.rotate(90, expand=True)
        )

        return variants

    @staticmethod
    def _hamming_distance(hash_a: str, hash_b: str) -> int:
        """Compute Hamming distance between two binary hash strings."""
        return sum(c1 != c2 for c1, c2 in zip(hash_a, hash_b))


# ═══════════════════════════════════════════════════════════════════════
# Background Consistency Analysis (Splicing Detection)
# ═══════════════════════════════════════════════════════════════════════

class BackgroundConsistencyInput(BaseModel):
    """Input for background consistency analysis."""

    image_path_or_url: str = Field(
        ..., description="Local file path or URL of the image to analyze."
    )
    num_lanes: int = Field(
        default=0,
        ge=0,
        le=20,
        description=(
            "Number of vertical lanes/regions to divide the image into. "
            "0 = auto-detect based on image width (typical Western blot spacing)."
        ),
    )
    lane_width_ratio: float = Field(
        default=0.12,
        ge=0.05,
        le=0.5,
        description="Width of each lane as fraction of image width (for auto-detect).",
    )


class BackgroundConsistencyTool(BaseTool):
    """
    Detect image splicing by comparing background consistency across lanes/regions.

    Why this works:
    When images from different sources are spliced together (e.g., Western blot
    lanes from different gels), the background noise, brightness, and texture
    are subtly different in each spliced region. These differences are often
    invisible to the naked eye but detectable via statistical comparison.

    How it works:
    1. Divide the image into vertical lanes (like Western blot lanes).
    2. For each lane, compute:
       - Mean background intensity
       - Standard deviation of background (noise level)
       - Local variance (texture granularity)
    3. Compare stats pairwise between lanes — flag outliers.
    4. Also checks for sharp vertical edges that could indicate splice boundaries.
    """

    name: str = "background_consistency_check"
    description: str = (
        "Analyze whether an image has consistent background across all regions, "
        "or whether some regions have different noise/brightness/texture — a key "
        "indicator of image splicing (e.g., Western blot lanes from different gels "
        "pasted together). Divides the image into vertical lanes, computes per-lane "
        "background statistics, and flags statistically deviant lanes. "
        "Also checks for unnatural vertical edges at lane boundaries. "
        "Use this for Western blots, gel images, and any figure with distinct lanes."
    )
    args_schema: type[BaseModel] = BackgroundConsistencyInput

    def _run(
        self,
        image_path_or_url: str,
        num_lanes: int = 0,
        lane_width_ratio: float = 0.12,
    ) -> str:
        """Execute background consistency analysis."""
        try:
            img, meta = load_image(image_path_or_url, use_cache=True)
            if img is None:
                return json.dumps({"error": meta.get("error"), "flagged": False})
        except Exception as e:
            return json.dumps({"error": str(e), "flagged": False})

        try:
            import numpy as np

            gray = img.convert("L")
            arr = np.array(gray, dtype=np.float64)
            h, w = arr.shape

            if w < 100 or h < 50:
                return json.dumps({
                    "error": f"Image too small for background analysis ({w}x{h}).",
                    "flagged": False,
                })

            # Determine lanes
            if num_lanes <= 0:
                lane_w = int(w * lane_width_ratio)
                num_lanes = max(2, w // lane_w)
            lane_w = w // num_lanes

            if lane_w < 10:
                return json.dumps({
                    "error": f"Lane width too narrow ({lane_w}px). Use fewer lanes.",
                    "flagged": False,
                })

            # Compute per-lane background stats
            lane_stats = []
            for i in range(num_lanes):
                x_start = i * lane_w
                x_end = (i + 1) * lane_w if i < num_lanes - 1 else w
                lane_region = arr[:, x_start:x_end]

                lane_mean = float(np.mean(lane_region))
                lane_std = float(np.std(lane_region))
                # Local variance as texture measure
                local_var = float(np.mean(np.abs(np.diff(lane_region, axis=0))))
                # Gradient magnitude at boundaries (for edge detection)
                if i > 0:
                    left_boundary = arr[:, x_start:min(x_start + 5, x_end)]
                    right_of_boundary = arr[:, max(0, x_start - 5):x_start]
                    if left_boundary.shape[1] > 0 and right_of_boundary.shape[1] > 0:
                        boundary_gradient = float(
                            np.mean(np.abs(
                                left_boundary[:, 0] - right_of_boundary[:, -1]
                            ))
                        )
                    else:
                        boundary_gradient = 0.0
                else:
                    boundary_gradient = 0.0

                lane_stats.append({
                    "lane": i + 1,
                    "x_range": [x_start, x_end],
                    "mean_intensity": round(lane_mean, 2),
                    "std_intensity": round(lane_std, 2),
                    "texture_roughness": round(local_var, 4),
                    "left_boundary_gradient": round(boundary_gradient, 2),
                })

            # Find outliers: lanes where background stats deviate > 2σ from the median
            means = [s["mean_intensity"] for s in lane_stats]
            stds = [s["std_intensity"] for s in lane_stats]
            textures = [s["texture_roughness"] for s in lane_stats]
            gradients = [s["left_boundary_gradient"] for s in lane_stats]

            med_mean = np.median(means)
            med_std = np.median(stds)
            mad_mean = np.median(np.abs(np.array(means) - med_mean)) * 1.4826 or 1.0
            mad_std = np.median(np.abs(np.array(stds) - med_std)) * 1.4826 or 1.0

            anomalies = []
            for s in lane_stats:
                reasons = []
                z_mean = abs(s["mean_intensity"] - med_mean) / mad_mean
                z_std = abs(s["std_intensity"] - med_std) / mad_std

                if z_mean > 2.5:
                    reasons.append(f"Brightness deviates {z_mean:.1f}σ from other lanes")
                if z_std > 2.5:
                    reasons.append(f"Noise level deviates {z_std:.1f}σ from other lanes")
                if s["left_boundary_gradient"] > 2.0 * np.median(gradients) + 5:
                    reasons.append(
                        f"Sharp edge at left boundary "
                        f"(gradient={s['left_boundary_gradient']:.1f})"
                    )

                if reasons:
                    anomalies.append({
                        "lane": s["lane"],
                        "reasons": reasons,
                        "stats": s,
                    })

            return json.dumps({
                "analysis_type": "Background Consistency Analysis",
                "image_source": image_path_or_url,
                "image_dimensions": f"{w}x{h}",
                "lanes_analyzed": num_lanes,
                "lane_width_px": lane_w,
                "lane_stats": lane_stats,
                "anomalies_detected": len(anomalies),
                "flagged": len(anomalies) > 0,
                "anomalies": anomalies,
                "interpretation": (
                    f"{len(anomalies)} of {num_lanes} lane(s) show inconsistent "
                    "background characteristics. This may indicate image splicing — "
                    "lanes from different original images pasted together. "
                    "Manual review of the flagged lane boundaries is recommended. "
                    f"Anomalous lanes: {[a['lane'] for a in anomalies]}."
                    if anomalies
                    else f"All {num_lanes} lanes show consistent background. "
                    "No splicing detected."
                ),
            })

        except Exception as e:
            logger.error(f"Background consistency analysis failed: {e}")
            return json.dumps({"error": str(e), "flagged": False})


# ═══════════════════════════════════════════════════════════════════════════
# Feature-Based Cross-Image Duplicate Detection (SIFT/ORB + RANSAC)
# ═══════════════════════════════════════════════════════════════════════════

class FeatureBasedDuplicateInput(BaseModel):
    """Input for feature-based cross-image duplicate detection."""

    image_paths: str = Field(
        ...,
        description=(
            "JSON-encoded list of image file paths (or panel image paths) to compare. "
            'Example: \'["/path/to/fig1_panel_0.png", "/path/to/fig4_panel_0.png"]\'.'
        ),
    )
    min_inliers: int = Field(
        default=12,
        ge=4,
        le=1000,
        description="Minimum number of RANSAC inliers to flag as a match. "
        "Lower = more sensitive. For HE staining / histology images, consider 8-10. "
        "This value OVERRIDES the dynamic threshold when it is lower (i.e., setting "
        "min_inliers=8 forces the tool to flag weaker partial overlaps).",
    )
    use_sift: bool = Field(
        default=True,
        description="Prefer SIFT (more accurate) over ORB. Falls back to ORB if SIFT unavailable.",
    )
    ratio_threshold: float = Field(
        default=0.75,
        ge=0.60,
        le=0.95,
        description="Lowe's ratio test threshold for descriptor matching. "
        "0.75 (default) = standard strict matching. "
        "0.80-0.85 recommended for histology/microscopy images with repetitive cell textures. "
        "Higher = more matches pass the ratio test (more sensitive, but may introduce noise).",
    )
    sift_contrast_threshold: float = Field(
        default=0.03,
        ge=0.01,
        le=0.10,
        description="SIFT contrastThreshold for keypoint detection. "
        "0.03 (default) = standard. 0.02 recommended for low-contrast biological images "
        "(HE staining, fluorescence microscopy). Lower = more keypoints detected.",
    )


class FeatureBasedDuplicateTool(BaseTool):
    """
    Detect partial image reuse across different figures using feature point matching.

    This is the KEY tool for catching the most common fraud pattern: reusing the
    same tissue section / cell image / Western blot region in different figures,
    even when the region is cropped, scaled, rotated by small angles, or
    presented with different surrounding content.

    How it works:
    1. For each image pair, detect SIFT (or ORB) keypoints and descriptors.
    2. Match descriptors using Lowe's ratio test (mutual nearest neighbor with
       distance ratio < 0.75).
    3. Compute a homography via RANSAC on the matched point coordinates.
    4. If the number of geometric inliers exceeds the threshold, the images
       share a common region — a red flag for image reuse.

    Unlike pHash-based comparison (CrossImageDuplicateTool), this catches:
    - **Partial overlaps**: when only a sub-region of image A appears in image B.
    - **Scale changes**: same region at different magnifications.
    - **Small rotations**: not just 90° multiples, but arbitrary angles.
    - **Cropping + pasting**: the most realistic fraud scenario.

    Use this AFTER extracting individual panels from composite figures (via
    figure_splitter module) for panel-level comparison.
    """

    name: str = "feature_based_duplicate_check"
    description: str = (
        "Advanced cross-image duplicate detection using SIFT/ORB feature point "
        "matching + RANSAC geometric verification. Catches partial image reuse "
        "(partial overlaps, cropping, scaling, small rotations) that pHash misses. "
        "This is ESSENTIAL for detecting reused tissue sections, cells, or blot "
        "regions across different figures — the #1 fraud pattern in biology papers. "
        "Input: JSON list of image file paths (ideally individual panel images from "
        "figure_splitter). Returns: matched pairs with inlier count, match quality, "
        "and estimated transform type."
    )
    args_schema: type[BaseModel] = FeatureBasedDuplicateInput

    def _compute_dynamic_threshold(self, image_size: tuple) -> int:
        """
        Compute adaptive min_inliers based on image area.

        Rationale: larger images have more pixels → more SIFT keypoints →
        more random false matches. A fixed threshold over-penalizes large
        panels and under-penalizes small panels.

        Calibration (empirical, tuned for biological/scientific images):
        - Small panels (< 50k px²): min_inliers = 8
        - Medium panels (50k-200k px²): min_inliers = 10
        - Large panels (200k-500k px²): min_inliers = 12
        - Very large (> 500k px²): min_inliers = 15
        """
        w, h = image_size
        area = w * h
        if area < 50000:
            return 8
        elif area < 200000:
            return 10
        elif area < 500000:
            return 12
        else:
            return 15

    def _run(
        self,
        image_paths: str,
        min_inliers: int = 12,
        use_sift: bool = True,
        ratio_threshold: float = 0.75,
        sift_contrast_threshold: float = 0.03,
    ) -> str:
        """Execute feature-based cross-image duplicate detection."""
        try:
            paths = json.loads(image_paths)
            if not isinstance(paths, list) or len(paths) < 2:
                return json.dumps({
                    "error": "Need at least 2 image paths as a JSON list.",
                    "flagged": False,
                })
        except json.JSONDecodeError:
            return json.dumps({
                "error": "image_paths must be a valid JSON list of strings.",
                "flagged": False,
            })

        # ── Initialize feature detector ──
        detector, feat_matcher = self._init_feature_detector(
            use_sift, contrast_threshold=sift_contrast_threshold
        )
        if detector is None:
            return json.dumps({
                "error": "No feature detector available. Install opencv-contrib-python for SIFT, "
                         "or opencv-python-headless for ORB.",
                "flagged": False,
            })

        # Store matcher as instance variable for use in _match_descriptors
        self._feat_matcher = feat_matcher

        # ── Load images ──
        images_loaded = []
        for path in paths:
            img, meta = load_image(path, use_cache=True)
            if img is None:
                logger.warning(f"Could not load image: {path} — {meta.get('error')}")
                continue
            # Convert to grayscale for feature detection
            gray = img.convert("L")
            images_loaded.append({
                "path": path,
                "filename": os.path.basename(path),
                "img": img,
                "gray": gray,
                "size": img.size,
            })

        if len(images_loaded) < 2:
            return json.dumps({
                "analysis_type": "Feature-Based Cross-Image Duplicate Detection",
                "detector_type": "SIFT" if use_sift else "ORB",
                "images_compared": len(images_loaded),
                "flagged": False,
                "matches": [],
                "interpretation": "Not enough valid images to compare.",
            })

        # ── Detect keypoints and descriptors for all images ──
        keypoints_by_image = {}
        descriptors_by_image = {}
        for entry in images_loaded:
            kp, desc = self._detect_keypoints(detector, entry["gray"])
            keypoints_by_image[entry["filename"]] = kp
            descriptors_by_image[entry["filename"]] = desc

        # ── Compare all pairs ──
        import numpy as np

        filenames = list(descriptors_by_image.keys())
        matches = []
        for i in range(len(filenames)):
            for j in range(i + 1, len(filenames)):
                name_a, name_b = filenames[i], filenames[j]
                desc_a = descriptors_by_image[name_a]
                desc_b = descriptors_by_image[name_b]
                kp_a = keypoints_by_image[name_a]
                kp_b = keypoints_by_image[name_b]

                if desc_a is None or desc_b is None:
                    continue
                if len(kp_a) < 4 or len(kp_b) < 4:
                    continue  # need at least 4 points for homography

                # ── Dynamic threshold based on image sizes ──
                size_a = images_loaded[i]["size"]
                size_b = images_loaded[j]["size"]
                # Use the smaller image's threshold (more conservative)
                dynamic_min_inliers = min(
                    self._compute_dynamic_threshold(size_a),
                    self._compute_dynamic_threshold(size_b),
                )
                # Agent-specified min_inliers can LOWER the threshold (more sensitive)
                # but cannot raise it above the dynamic threshold (safety ceiling).
                # This fixes the bug where agent's explicit min_inliers was silently ignored.
                effective_min_inliers = min(dynamic_min_inliers, min_inliers)

                # ── Match descriptors ──
                good_matches = self._match_descriptors(desc_a, desc_b, ratio_threshold)

                if len(good_matches) < effective_min_inliers:
                    continue

                # ── Geometric verification via RANSAC homography ──
                src_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                try:
                    H, mask = self._find_homography(src_pts, dst_pts, effective_min_inliers)
                except Exception:
                    continue

                if H is None or mask is None:
                    continue

                inlier_count = int(np.sum(mask))
                if inlier_count < effective_min_inliers:
                    continue

                # ── Analyze the transform ──
                transform_type = self._classify_transform(H)
                overlap_fraction = self._estimate_overlap(
                    H, images_loaded[i]["size"], images_loaded[j]["size"]
                )

                # ── Compute match quality metrics ──
                avg_distance = float(np.mean([
                    m.distance for m, msk in zip(good_matches, mask) if msk
                ]))

                confidence = (
                    "very_high" if inlier_count >= 30 else
                    "high" if inlier_count >= 20 else
                    "medium" if inlier_count >= 15 else
                    "low"
                )

                matches.append({
                    "image_a": name_a,
                    "image_b": name_b,
                    "keypoints_a": len(kp_a),
                    "keypoints_b": len(kp_b),
                    "total_good_matches": len(good_matches),
                    "ransac_inliers": inlier_count,
                    "inlier_ratio": round(inlier_count / max(len(good_matches), 1), 4),
                    "avg_match_distance": round(avg_distance, 2),
                    "transform_type": transform_type,
                    "overlap_fraction": round(overlap_fraction, 4),
                    "confidence": confidence,
                })

        # ── Sort by inlier count descending (strongest matches first) ──
        matches.sort(key=lambda m: -m["ransac_inliers"])

        return json.dumps({
            "analysis_type": "Feature-Based Cross-Image Duplicate Detection",
            "detector_type": "SIFT" if use_sift else "ORB",
            "images_compared": len(images_loaded),
            "pairs_checked": len(filenames) * (len(filenames) - 1) // 2,
            "min_inliers_threshold": min_inliers,
            "flagged": len(matches) > 0,
            "match_count": len(matches),
            "matches": matches[:50],  # top 50 matches
            "interpretation": (
                f"CRITICAL: Found {len(matches)} image pair(s) with significant "
                f"feature-point correspondence (≥{min_inliers} RANSAC inliers). "
                "These images share a common visual region — strong evidence of "
                "image reuse across different figures. This is a MAJOR red flag "
                "for academic fraud. Immediate investigation required."
                if matches
                else "No feature-level duplicates detected across images. "
                "All image pairs appear to contain genuinely distinct content."
            ),
        })

    # ── Helper methods ──

    def _init_feature_detector(self, use_sift: bool, contrast_threshold: float = 0.03) -> tuple:
        """Initialize SIFT or ORB detector and FLANN/BF matcher."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            logger.error("OpenCV (cv2) is not installed.")
            return None, None

        detector = None
        matcher = None

        if use_sift:
            # Try SIFT first (more accurate for scientific images)
            try:
                detector = cv2.SIFT_create(
                    nfeatures=2000,
                    contrastThreshold=contrast_threshold,
                )
                # FLANN matcher for SIFT
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                matcher = cv2.FlannBasedMatcher(index_params, search_params)
                logger.info("Using SIFT detector with FLANN matcher")
            except (AttributeError, cv2.error) as e:
                logger.warning(f"SIFT not available: {e}. Falling back to ORB.")
                use_sift = False

        if not use_sift:
            # Fall back to ORB (free, always available)
            try:
                detector = cv2.ORB_create(
                    nfeatures=2000,
                    scaleFactor=1.2,
                    nlevels=8,
                    edgeThreshold=15,
                )
                # Brute-Force Hamming matcher for ORB binary descriptors
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                logger.info("Using ORB detector with BF-Hamming matcher")
            except Exception as e:
                logger.error(f"ORB initialization failed: {e}")
                return None, None

        return detector, matcher

    def _detect_keypoints(self, detector, gray_img: "Image.Image") -> tuple:
        """Detect keypoints and compute descriptors for a grayscale image."""
        import numpy as np

        arr = np.array(gray_img, dtype=np.uint8)
        try:
            keypoints, descriptors = detector.detectAndCompute(arr, None)
            if descriptors is None:
                return [], None
            return keypoints, descriptors
        except Exception as e:
            logger.warning(f"Keypoint detection failed: {e}")
            return [], None

    def _match_descriptors(self, desc_a, desc_b, ratio_threshold: float = 0.75) -> list:
        """Match descriptors using Lowe's ratio test (mutual NN).

        Args:
            desc_a, desc_b: SIFT/ORB descriptor arrays.
            ratio_threshold: Lowe's ratio test threshold. 0.75 (default) = standard
                strict matching. 0.80-0.85 recommended for histology/microscopy images
                with repetitive textures (cell nuclei, tissue patterns).
        """
        import cv2

        try:
            # Get 2 nearest neighbors for each descriptor
            raw_matches = self._get_knn_matches(desc_a, desc_b, k=2)
        except Exception:
            return []

        # Lowe's ratio test: keep matches where the best match is significantly
        # better than the second-best (ratio < threshold)
        good = []
        for match_pair in raw_matches:
            if len(match_pair) < 2:
                continue
            m, n = match_pair[0], match_pair[1]
            if m.distance < ratio_threshold * n.distance:
                good.append(m)

        return good

    def _get_knn_matches(self, desc_a, desc_b, k: int = 2) -> list:
        """Get k-nearest-neighbor matches. Handles both FLANN and BF matchers."""
        import cv2

        try:
            # Try FLANN-style knnMatch
            raw = self._feat_matcher.knnMatch(desc_a, desc_b, k=k)
            return raw
        except Exception:
            pass

        # Fall back to BF-style: match all, then sort by distance
        try:
            all_matches = self._feat_matcher.match(desc_a, desc_b)
            all_matches = sorted(all_matches, key=lambda m: m.distance)
            # Group by query index to simulate knn
            from collections import defaultdict
            groups = defaultdict(list)
            for m in all_matches:
                groups[m.queryIdx].append(m)
            result = []
            for qidx in sorted(groups.keys()):
                result.append(groups[qidx][:k])
            return result
        except Exception:
            return []

    def _find_homography(
        self,
        src_pts: "np.ndarray",
        dst_pts: "np.ndarray",
        min_inliers: int,
    ) -> tuple:
        """Find homography via RANSAC. Returns (H, mask)."""
        import cv2
        import numpy as np

        if len(src_pts) < 4:
            return None, None

        try:
            H, mask = cv2.findHomography(
                src_pts, dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0,
                confidence=0.99,
                maxIters=2000,
            )
            return H, mask
        except cv2.error:
            return None, None

    def _classify_transform(self, H: "np.ndarray") -> str:
        """Classify the type of geometric transform from the homography matrix."""
        import numpy as np

        if H is None:
            return "unknown"

        # Extract rotation, scale, translation components
        # For a similarity transform, H ≈ [[s*cos, -s*sin, tx], [s*sin, s*cos, ty], [0, 0, 1]]
        a, b = H[0, 0], H[0, 1]
        c, d = H[1, 0], H[1, 1]
        tx, ty = H[0, 2], H[1, 2]

        # Estimate rotation angle
        angle = np.arctan2(c, a) * 180.0 / np.pi
        # Estimate scale
        scale = np.sqrt(a**2 + c**2)

        # Classify
        parts = []
        if abs(angle) < 1.0 and abs(scale - 1.0) < 0.05:
            parts.append("near_identical")
        elif abs(angle) < 3.0:
            if abs(scale - 1.0) < 0.10:
                parts.append("translation_only")
            else:
                parts.append(f"scale_{scale:.1f}x")
        elif 85 < abs(angle) < 95:
            parts.append("rotation_~90deg")
        elif 170 < abs(angle) < 190:
            parts.append("rotation_~180deg")
        else:
            parts.append(f"rotation_{angle:.0f}deg")

        if abs(scale - 1.0) >= 0.10:
            parts.append(f"scale_{scale:.2f}x")

        if abs(tx) > 10 or abs(ty) > 10:
            parts.append(f"translation_({tx:.0f},{ty:.0f})px")

        return "+".join(parts) if parts else "affine"

    def _estimate_overlap(
        self,
        H: "np.ndarray",
        size_a: tuple,
        size_b: tuple,
    ) -> float:
        """Estimate the fraction of image A's area that overlaps with image B."""
        import numpy as np

        w_a, h_a = size_a
        w_b, h_b = size_b

        # Project the four corners of image A through H to get their positions in image B
        corners_a = np.float32([
            [0, 0, 1],
            [w_a, 0, 1],
            [w_a, h_a, 1],
            [0, h_a, 1],
        ]).T  # 3×4

        projected = H @ corners_a  # 3×4
        projected = projected / projected[2, :]  # normalize
        projected_pts = projected[:2, :].T  # 4×2

        # Clip to image B bounds
        clipped = np.clip(projected_pts, [0, 0], [w_b, h_b])
        # Approximate area of the projected quadrilateral
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(clipped)
            overlap_area = hull.volume  # in 2D, "volume" = area
        except Exception:
            # Degenerate: use bounding box area
            min_xy = np.min(clipped, axis=0)
            max_xy = np.max(clipped, axis=0)
            overlap_area = max(0, (max_xy[0] - min_xy[0]) * (max_xy[1] - min_xy[1]))

        total_area = w_a * h_a
        return min(1.0, overlap_area / max(total_area, 1))
