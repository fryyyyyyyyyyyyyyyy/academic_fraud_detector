"""
Text extraction utilities for PDF and HTML content.

Uses PyMuPDF (fitz) for PDFs and BeautifulSoup for HTML.
Provides a unified extract() interface that all tools can use.
"""

import logging
import os
from typing import Optional, List, Dict, Any
from io import BytesIO
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# Default output directory for extracted images
DEFAULT_IMAGE_OUTPUT_DIR = Path.home() / ".cache" / "academic_fraud_detector" / "extracted_images"


def extract_pdf_text(pdf_content: bytes, max_pages: Optional[int] = None) -> str:
    """
    Extract text from a PDF byte buffer.

    Args:
        pdf_content: Raw PDF bytes.
        max_pages: Maximum number of pages to extract (None = all).

    Returns:
        Extracted text with page markers.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error("PyMuPDF (fitz) is not installed. Install with: pip install pymupdf")
        return ""

    doc = fitz.open(stream=pdf_content, filetype="pdf")
    pages = []
    page_limit = max_pages or doc.page_count
    for i in range(min(doc.page_count, page_limit)):
        page = doc[i]
        text = page.get_text("text")
        if text.strip():
            pages.append(f"[Page {i + 1}]\n{text}")
    doc.close()
    return "\n\n".join(pages)


def extract_pdf_text_from_url(url: str, max_pages: Optional[int] = None) -> str:
    """
    Download a PDF from a URL and extract its text.

    Args:
        url: URL to the PDF file.
        max_pages: Maximum pages to extract.

    Returns:
        Extracted text.
    """
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return extract_pdf_text(resp.content, max_pages=max_pages)
    except Exception as e:
        logger.error(f"Failed to extract PDF from {url}: {e}")
        return ""


def extract_pdf_images(
    pdf_content: bytes,
    output_dir: Optional[str] = None,
    min_size: int = 100,
    max_pages: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Extract embedded images from a PDF byte buffer.

    Uses PyMuPDF to locate and extract images embedded in the PDF pages.
    Images smaller than min_size pixels (width or height) are skipped.

    Args:
        pdf_content: Raw PDF bytes.
        output_dir: Directory to save extracted images. Uses default cache if None.
        min_size: Minimum image dimension in pixels to keep.
        max_pages: Maximum number of pages to process (None = all).

    Returns:
        List of dicts, each with:
        - 'filename': saved filename
        - 'filepath': absolute path to the saved image
        - 'format': image format (png, jpeg, etc.)
        - 'width', 'height': pixel dimensions
        - 'page_number': which PDF page the image came from
        - 'xref': internal PDF xref number
        - 'size_bytes': file size
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error("PyMuPDF (fitz) is not installed. Install with: pip install pymupdf")
        return []

    save_dir = Path(output_dir) if output_dir else DEFAULT_IMAGE_OUTPUT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(stream=pdf_content, filetype="pdf")
    extracted = []
    page_limit = max_pages or doc.page_count

    for page_idx in range(min(doc.page_count, page_limit)):
        page = doc[page_idx]
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]  # xref number
            try:
                base_image = doc.extract_image(xref)
            except Exception:
                logger.debug(f"Failed to extract image xref={xref} on page {page_idx + 1}")
                continue

            image_bytes = base_image.get("image")
            if not image_bytes:
                continue

            ext = base_image.get("ext", "png")
            width = base_image.get("width", 0)
            height = base_image.get("height", 0)

            # Skip tiny images (icons, decorative elements)
            if width < min_size or height < min_size:
                continue

            # Generate unique filename
            filename = f"page{page_idx + 1}_img{img_idx + 1}.{ext}"
            filepath = save_dir / filename

            # Avoid overwriting: add suffix if file exists
            counter = 1
            while filepath.exists():
                filename = f"page{page_idx + 1}_img{img_idx + 1}_{counter}.{ext}"
                filepath = save_dir / filename
                counter += 1

            with open(filepath, "wb") as f:
                f.write(image_bytes)

            extracted.append({
                "filename": filename,
                "filepath": str(filepath.absolute()),
                "format": ext.upper(),
                "width": width,
                "height": height,
                "page_number": page_idx + 1,
                "xref": xref,
                "size_bytes": len(image_bytes),
            })

    doc.close()
    logger.info(f"Extracted {len(extracted)} images from PDF ({page_limit} pages)")
    return extracted


def extract_pdf_images_from_file(
    filepath: str,
    output_dir: Optional[str] = None,
    min_size: int = 100,
    max_pages: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Extract embedded images from a local PDF file.

    Args:
        filepath: Path to the local PDF file.
        output_dir: Directory to save extracted images.
        min_size: Minimum image dimension in pixels to keep.
        max_pages: Maximum number of pages to process.

    Returns:
        List of image metadata dicts.
    """
    with open(filepath, "rb") as f:
        return extract_pdf_images(f.read(), output_dir=output_dir, min_size=min_size, max_pages=max_pages)


def extract_html_text(html_content: str) -> str:
    """
    Extract readable text from HTML content.

    Strips scripts, styles, and navigation elements.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.error("BeautifulSoup is not installed. Install with: pip install beautifulsoup4")
        return html_content

    soup = BeautifulSoup(html_content, "html.parser")

    # Remove non-content elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Collapse excessive whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def extract_text(source: str, source_type: str = "auto", max_pages: Optional[int] = None) -> str:
    """
    Unified text extraction interface.

    Args:
        source: Path to file, URL, or raw text.
        source_type: One of 'auto', 'pdf', 'html', 'text', 'url'.
        max_pages: Max pages for PDF extraction.

    Returns:
        Extracted plain text.
    """
    if source_type == "auto":
        source_lower = source.lower()
        if source_lower.endswith(".pdf") or "pdf" in source_lower:
            source_type = "pdf"
        elif source_lower.endswith((".htm", ".html")):
            source_type = "html"
        elif source.startswith(("http://", "https://")):
            # Try to detect from content-type
            source_type = "url"
        else:
            source_type = "text"

    if source_type == "pdf":
        if source.startswith(("http://", "https://")):
            return extract_pdf_text_from_url(source, max_pages=max_pages)
        with open(source, "rb") as f:
            return extract_pdf_text(f.read(), max_pages=max_pages)

    elif source_type == "html":
        if source.startswith(("http://", "https://")):
            resp = requests.get(source, timeout=30)
            resp.raise_for_status()
            return extract_html_text(resp.text)
        with open(source, "r", encoding="utf-8") as f:
            return extract_html_text(f.read())

    elif source_type == "url":
        resp = requests.get(source, timeout=30)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "pdf" in content_type:
            return extract_pdf_text(resp.content, max_pages=max_pages)
        elif "html" in content_type:
            return extract_html_text(resp.text)
        else:
            return resp.text

    else:
        # Plain text
        if source.startswith(("http://", "https://")):
            resp = requests.get(source, timeout=30)
            return resp.text
        try:
            with open(source, "r", encoding="utf-8") as f:
                return f.read()
        except (FileNotFoundError, OSError):
            return source
