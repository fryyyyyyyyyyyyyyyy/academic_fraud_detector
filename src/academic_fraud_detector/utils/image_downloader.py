"""
Image download and caching utilities.

Handles batch downloading of paper figures with caching,
format detection, and basic validation.
"""

import os
import logging
import hashlib
from typing import Optional, List, Tuple
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

logger = logging.getLogger(__name__)

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "academic_fraud_detector" / "images"


def set_cache_dir(path: str) -> None:
    """Change the default image cache directory."""
    global DEFAULT_CACHE_DIR
    DEFAULT_CACHE_DIR = Path(path)


def _url_to_cache_key(url: str) -> str:
    """Generate a cache filename from a URL."""
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    return f"{url_hash}"


def download_image(
    url: str,
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
) -> Tuple[Optional[Image.Image], dict]:
    """
    Download an image from a URL, with optional caching.

    Args:
        url: Image URL.
        cache_dir: Directory for cached images. Uses default if None.
        use_cache: Whether to check cache before downloading.

    Returns:
        Tuple of (PIL.Image or None, metadata dict).
        metadata includes: 'url', 'format', 'width', 'height', 'file_size_bytes',
        'cached', 'error' (if any).
    """
    metadata = {
        "url": url,
        "format": None,
        "width": None,
        "height": None,
        "file_size_bytes": None,
        "cached": False,
        "error": None,
    }

    cache_path = None
    if use_cache:
        cache_root = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_path = cache_root / _url_to_cache_key(url)

        if cache_path.exists():
            try:
                img = Image.open(cache_path)
                img.load()  # Force load to verify integrity
                metadata["cached"] = True
                metadata["format"] = img.format
                metadata["width"], metadata["height"] = img.size
                metadata["file_size_bytes"] = cache_path.stat().st_size
                return img, metadata
            except Exception:
                logger.warning(f"Cached image corrupted, re-downloading: {url}")

    # Download
    try:
        resp = requests.get(url, timeout=30, stream=True)
        resp.raise_for_status()

        content = resp.content
        metadata["file_size_bytes"] = len(content)

        img = Image.open(BytesIO(content))
        img.load()
        metadata["format"] = img.format
        metadata["width"], metadata["height"] = img.size

        # Cache it
        if cache_path:
            with open(cache_path, "wb") as f:
                f.write(content)

        return img, metadata

    except requests.RequestException as e:
        metadata["error"] = f"Download failed: {e}"
        logger.error(f"Failed to download image {url}: {e}")
        return None, metadata
    except Exception as e:
        metadata["error"] = f"Image processing failed: {e}"
        logger.error(f"Failed to process image {url}: {e}")
        return None, metadata


def download_images(
    urls: List[str],
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
    max_images: int = 50,
) -> List[Tuple[Optional[Image.Image], dict]]:
    """
    Batch download images from a list of URLs.

    Args:
        urls: List of image URLs.
        cache_dir: Cache directory.
        use_cache: Whether to use cache.
        max_images: Maximum number of images to download.

    Returns:
        List of (Image, metadata) tuples, one per URL.
    """
    results = []
    for i, url in enumerate(urls[:max_images]):
        logger.info(f"Downloading image {i + 1}/{min(len(urls), max_images)}: {url[:80]}...")
        img, meta = download_image(url, cache_dir=cache_dir, use_cache=use_cache)
        results.append((img, meta))
    return results


def load_image(
    image_path_or_url: str,
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
) -> tuple[Optional[Image.Image], dict]:
    """
    Load an image from either a local file path or a remote URL.

    This is a unified interface that all image forensics tools should use.
    It auto-detects whether the input is a local path or a URL.

    Args:
        image_path_or_url: Local file path OR remote URL to the image.
        cache_dir: Cache directory (only used for URLs).
        use_cache: Whether to use cache (only used for URLs).

    Returns:
        Tuple of (PIL.Image or None, metadata dict).
    """
    # Detect if it's a local file path
    if os.path.exists(image_path_or_url):
        metadata = {
            "url": image_path_or_url,
            "format": None,
            "width": None,
            "height": None,
            "file_size_bytes": None,
            "cached": False,
            "error": None,
            "source": "local",
        }
        try:
            img = Image.open(image_path_or_url)
            img.load()
            metadata["format"] = img.format
            metadata["width"], metadata["height"] = img.size
            metadata["file_size_bytes"] = os.path.getsize(image_path_or_url)
            return img, metadata
        except Exception as e:
            metadata["error"] = f"Failed to load local image: {e}"
            logger.error(f"Failed to load local image {image_path_or_url}: {e}")
            return None, metadata

    # Otherwise treat as URL
    return download_image(image_path_or_url, cache_dir=cache_dir, use_cache=use_cache)


def validate_image(img: Image.Image, min_size: int = 100) -> bool:
    """
    Validate that an image is usable for analysis.

    Args:
        img: PIL Image to validate.
        min_size: Minimum acceptable width or height in pixels.

    Returns:
        True if the image passes validation.
    """
    if img is None:
        return False
    if img.width < min_size or img.height < min_size:
        return False
    return True


def image_to_bytes(img: Image.Image, format: str = "PNG") -> bytes:
    """Convert a PIL Image to bytes."""
    buf = BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()
