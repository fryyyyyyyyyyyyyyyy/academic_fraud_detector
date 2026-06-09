"""
Supplementary Information (SI) Downloader — auto-download and parse
supplementary materials linked from academic papers.

Many papers store raw data in supplementary files hosted on:
- figshare (PLOS ONE, Nature, etc.) — API-based download
- PubMed Central (PMC) — open access articles
- PLOS ONE direct SI PDFs (journal.pone.XXXXXXX.sXXX.pdf)
- bioRxiv/medRxiv supplementary material

This module extracts SI links from PDF text or DOI, downloads files,
and parses structured data (XLSX, CSV) into a format compatible with
CrossFigureDataComparisonTool.

Key functions:
- extract_si_links_from_pdf_text() — find SI URLs in PDF text content
- download_si_files() — download SI files from found URLs
- parse_si_data() — parse XLSX/CSV into structured group-value format
- download_and_parse_si_data() — full pipeline convenience function
"""

import json
import logging
import os
import re
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse

import requests

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}


def extract_si_links_from_pdf_text(pdf_text: str) -> List[Dict[str, str]]:
    """
    Extract supplementary information links from PDF text content.

    Searches for:
    - figshare URLs (common for PLOS ONE, Nature, etc.)
    - "S1 File", "S2 File", "Supplementary" mentions with adjacent URLs
    - DOI-based SI links
    - General http/https URLs containing supplementary-related keywords

    Args:
        pdf_text: Full text content extracted from the PDF.

    Returns:
        List of dicts with 'url', 'label' (e.g., 'S2 File'), 'host'.
    """
    links = []
    seen_urls = set()

    # ── Pattern 1: Explicit SI labels with adjacent URLs ──
    si_patterns = [
        r'(S\d+\s*(?:File|Table|Figure|Dataset|Data|Text|Appendix)[^.]*?\.)\s*(https?://[^\s]+)',
        r'(Supporting\s+Information[^.]*?\.?)\s*(https?://[^\s]+)',
        r'(Supplementary\s+(?:Material|Data|File|Figure|Table)[^.]*?)\s*(https?://[^\s]+)',
        r'(Data\s+Availability[^.]*?)\s*(https?://[^\s]+)',
    ]

    for pattern in si_patterns:
        for match in re.finditer(pattern, pdf_text, re.IGNORECASE):
            label_part = match.group(1).strip()
            url = match.group(2).strip().rstrip('.,;:')
            if url not in seen_urls:
                seen_urls.add(url)
                links.append({
                    "url": url,
                    "label": label_part[:100],
                    "host": _identify_host(url),
                    "source": "labeled_si_pattern",
                })

    # ── Pattern 2: Any figshare URL in the text ──
    for match in re.finditer(r'https?://(?:www\.)?figshare\.com/\S+', pdf_text):
        url = match.group(0).rstrip('.,;:')
        if url not in seen_urls:
            seen_urls.add(url)
            links.append({
                "url": url,
                "label": "figshare_link",
                "host": "figshare.com",
                "source": "figshare_url_scan",
            })

    # ── Pattern 3: DOI links near SI mentions ──
    for match in re.finditer(r'(?:doi\.org/|DOI:?\s*)(10\.\d{4,}/[^\s.,;:]+)', pdf_text):
        doi = match.group(1)
        url = f"https://doi.org/{doi}"
        if url not in seen_urls:
            seen_urls.add(url)
            links.append({
                "url": url,
                "label": f"DOI: {doi[:60]}",
                "host": "doi.org",
                "source": "doi_pattern",
            })

    # ── Pattern 4: S* File labels WITHOUT adjacent URLs ─────────────────
    # PLOS ONE papers often list "S1 File. (PDF)" or "S2 File. (XLSX)"
    # without a direct URL. These need to be resolved via DOI/figshare API.
    unresolved_labels = []
    si_label_no_url_pattern = re.compile(
        r'(S\d+\s*(?:File|Table|Figure|Dataset|Data|Text|Appendix))'
        r'(?:[^.]*?)\.?\s*(?:\(([^)]+)\))?',
        re.IGNORECASE,
    )
    for match in si_label_no_url_pattern.finditer(pdf_text):
        label = match.group(1).strip()
        file_type = match.group(2).strip() if match.group(2) else "unknown"
        # Only capture if no URL was found adjacent to this label
        # (check that this label isn't already part of a Pattern 1 match)
        already_found = any(
            label.lower() in l.get("label", "").lower() for l in links
        )
        if not already_found:
            unresolved_labels.append({
                "label": label,
                "file_type": file_type,
            })

    # Deduplicate unresolved labels
    seen_labels = set()
    unique_unresolved = []
    for ul in unresolved_labels:
        if ul["label"].lower() not in seen_labels:
            seen_labels.add(ul["label"].lower())
            unique_unresolved.append(ul)

    logger.info(
        f"Found {len(links)} SI links with URLs and "
        f"{len(unique_unresolved)} SI labels without URLs in PDF text"
    )

    # Attach unresolved labels to result for downstream resolution
    # (stored as a special attribute on the first link or as metadata)
    if unique_unresolved and links:
        links[0]["_unresolved_si_labels"] = unique_unresolved
    elif unique_unresolved:
        links.append({
            "url": "",
            "label": "unresolved_si_labels",
            "host": "unknown",
            "source": "unresolved_si_labels",
            "_unresolved_si_labels": unique_unresolved,
        })

    return links


def _identify_host(url: str) -> str:
    """Identify the hosting platform from a URL."""
    try:
        parsed = urlparse(url)
        hostname = parsed.netloc.lower()
        for known_host in ["figshare.com", "doi.org", "ncbi.nlm.nih.gov",
                           "journals.plos.org", "biorxiv.org"]:
            if known_host in hostname:
                return known_host
        return hostname
    except Exception:
        return "unknown"


def download_si_files(
    si_links: List[Dict[str, str]],
    output_dir: str,
    timeout: int = 30,
) -> List[Dict[str, Any]]:
    """
    Download supplementary files from the found SI links.

    Handles:
    - figshare: Uses figshare API to get download URLs
    - Direct links: Downloads via HTTP GET
    - DOI links: Resolves to actual file URLs (follows redirects)

    Args:
        si_links: Output from extract_si_links_from_pdf_text().
        output_dir: Directory to save downloaded files.
        timeout: HTTP request timeout in seconds.

    Returns:
        List of download results with local file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    session = requests.Session()
    session.headers.update(HEADERS)

    for link_info in si_links:
        url = link_info["url"]
        host = link_info.get("host", "")
        label = link_info.get("label", "unknown")

        try:
            if "figshare.com" in host:
                result = _download_figshare(session, url, output_dir, label, timeout)
            else:
                result = _download_direct(session, url, output_dir, label, timeout)

            if result:
                result["source_link"] = link_info
                results.append(result)

        except Exception as e:
            logger.warning(f"Failed to download SI from {url}: {e}")
            results.append({
                "url": url, "label": label,
                "error": str(e), "downloaded": False,
            })

    logger.info(f"Downloaded {sum(1 for r in results if r.get('downloaded'))} "
                f"of {len(si_links)} SI files")
    return results


def _download_figshare(
    session: requests.Session,
    url: str,
    output_dir: Path,
    label: str,
    timeout: int,
) -> Optional[Dict[str, Any]]:
    """Download files from a figshare article page via the figshare API."""
    article_match = re.search(r'figshare\.com/(?:articles|files)/(?:[^/]+/)?(\d+)', url)
    if not article_match:
        return _download_direct(session, url, output_dir, label, timeout)

    article_id = article_match.group(1)
    api_url = f"https://api.figshare.com/v2/articles/{article_id}"

    try:
        resp = session.get(api_url, timeout=timeout)
        resp.raise_for_status()
        article_data = resp.json()
    except Exception as e:
        logger.warning(f"Figshare API lookup failed for article {article_id}: {e}")
        return _download_direct(session, url, output_dir, label, timeout)

    files = article_data.get("files", [])
    if not files:
        logger.warning(f"No files found in figshare article {article_id}")
        return None

    results = []
    for file_info in files:
        file_name = file_info.get("name", f"figshare_{article_id}_file")
        file_url = file_info.get("download_url") or \
                   f"https://figshare.com/ndownloader/files/{file_info.get('id', '')}"

        try:
            file_resp = session.get(file_url, timeout=timeout, stream=True)
            file_resp.raise_for_status()
            safe_name = re.sub(r'[<>:"/\\|?*]', '_', file_name)
            file_path = output_dir / safe_name
            with open(file_path, 'wb') as f:
                for chunk in file_resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            results.append({
                "url": file_url, "label": f"{label} — {file_name}",
                "filename": safe_name, "filepath": str(file_path.absolute()),
                "size_bytes": file_path.stat().st_size, "downloaded": True,
            })
            logger.info(f"Downloaded figshare file: {file_name} ({file_path.stat().st_size} bytes)")
        except Exception as e:
            logger.warning(f"Failed to download figshare file {file_name}: {e}")
            results.append({
                "url": file_url, "filename": file_name,
                "error": str(e), "downloaded": False,
            })

    return results[0] if len(results) == 1 else {
        "url": url, "label": label, "article_id": article_id,
        "files": results,
        "downloaded": any(r.get("downloaded") for r in results),
    }


def _download_direct(
    session: requests.Session,
    url: str,
    output_dir: Path,
    label: str,
    timeout: int,
) -> Optional[Dict[str, Any]]:
    """Download a file from a direct URL."""
    resp = session.get(url, timeout=timeout, allow_redirects=True, stream=True)

    content_type = resp.headers.get("Content-Type", "").lower()
    if "text/html" in content_type and resp.status_code == 200:
        cd = resp.headers.get("Content-Disposition", "")
        if "attachment" not in cd.lower():
            logger.debug(f"URL returned HTML page: {url}")
            return {
                "url": url, "label": label, "downloaded": False,
                "error": "URL returned HTML page, not a downloadable file",
            }

    resp.raise_for_status()

    file_name = None
    cd = resp.headers.get("Content-Disposition", "")
    filename_match = re.search(r'filename[*]?=["\']?([^"\';]+)', cd)
    if filename_match:
        file_name = filename_match.group(1)
    else:
        parsed = urlparse(url)
        path_name = os.path.basename(parsed.path)
        if path_name and '.' in path_name:
            file_name = path_name
        else:
            ext = _guess_extension(content_type)
            file_name = f"si_download_{hash(url) % 10000}{ext}"

    safe_name = re.sub(r'[<>:"/\\|?*]', '_', file_name)
    file_path = output_dir / safe_name

    with open(file_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    logger.info(f"Downloaded: {safe_name} ({file_path.stat().st_size} bytes)")
    return {
        "url": url, "label": label, "filename": safe_name,
        "filepath": str(file_path.absolute()),
        "size_bytes": file_path.stat().st_size,
        "content_type": content_type, "downloaded": True,
    }


def _resolve_si_from_doi(
    doi: str,
    output_dir: str,
    timeout: int = 30,
) -> List[Dict[str, Any]]:
    """
    Resolve supplementary information files for a paper via its DOI.

    This handles the common PLOS ONE pattern where S* Files are listed in
    the PDF text but without direct URLs — the files are accessible through
    the journal's article page or figshare collection.

    Strategy (tried in order):
    1. PLOS ONE article API: https://journals.plos.org/plosone/article/file?id=...
    2. Crossref API to get supplementary data links
    3. Construct figshare search URL from DOI

    Args:
        doi: Paper DOI (e.g., "10.1371/journal.pone.0313446").
        output_dir: Directory to save downloaded files.
        timeout: HTTP request timeout in seconds.

    Returns:
        List of download results.
    """
    results = []
    session = requests.Session()
    session.headers.update(HEADERS)
    output_path = Path(output_dir)

    logger.info(f"Resolving SI files for DOI: {doi}")

    # ── Strategy 1: PLOS ONE article file API ─────────────────────────
    if "journal.pone" in doi.lower():
        try:
            # PLOS ONE articles often have supplementary files at predictable URLs
            # e.g., https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0313446.s001
            article_id = doi.split("/")[-1]  # e.g., "pone.0313446"
            for suffix in [".s001", ".s002", ".s003", ".s004"]:
                file_id = f"10.1371/journal.{article_id}{suffix}"
                plos_url = (
                    f"https://journals.plos.org/plosone/article/file"
                    f"?id={file_id}&type=supplementary"
                )
                try:
                    resp = session.head(plos_url, timeout=timeout, allow_redirects=True)
                    if resp.status_code == 200:
                        # Try to download
                        dl_resp = session.get(plos_url, timeout=timeout, stream=True)
                        dl_resp.raise_for_status()

                        content_type = dl_resp.headers.get("Content-Type", "")
                        ext = _guess_extension(content_type)
                        file_name = f"{article_id}{suffix}{ext}"
                        file_path = output_path / file_name

                        with open(file_path, "wb") as f:
                            for chunk in dl_resp.iter_content(chunk_size=8192):
                                f.write(chunk)

                        results.append({
                            "url": plos_url,
                            "label": f"S{suffix[2:]} File",
                            "filename": file_name,
                            "filepath": str(file_path.absolute()),
                            "size_bytes": file_path.stat().st_size,
                            "downloaded": True,
                            "source": "plos_one_article_api",
                        })
                        logger.info(
                            f"Downloaded PLOS ONE SI: {file_name} "
                            f"({file_path.stat().st_size} bytes)"
                        )
                except Exception as e:
                    logger.debug(f"PLOS ONE SI {suffix} not found: {e}")
                    continue
        except Exception as e:
            logger.warning(f"PLOS ONE article API strategy failed: {e}")

    # ── Strategy 2: Crossref API ──────────────────────────────────────
    if not results:
        try:
            crossref_url = f"https://api.crossref.org/works/{doi}"
            resp = session.get(crossref_url, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            msg = data.get("message", {})

            # Check for supplementary data in Crossref record
            link_list = msg.get("link", [])
            for link in link_list:
                link_url = link.get("URL", "")
                content_type = link.get("content-type", "")
                if any(kw in (content_type + link_url).lower()
                       for kw in ["supplement", "data", "xlsx", "xls", "csv"]):
                    try:
                        dl_resp = session.get(link_url, timeout=timeout, stream=True)
                        dl_resp.raise_for_status()
                        file_name = os.path.basename(link_url) or f"si_crossref_{hash(link_url) % 10000}"
                        file_path = output_path / file_name
                        with open(file_path, "wb") as f:
                            for chunk in dl_resp.iter_content(chunk_size=8192):
                                f.write(chunk)
                        results.append({
                            "url": link_url,
                            "label": "Supplementary Data (Crossref)",
                            "filename": file_name,
                            "filepath": str(file_path.absolute()),
                            "size_bytes": file_path.stat().st_size,
                            "downloaded": True,
                            "source": "crossref_api",
                        })
                    except Exception as e:
                        logger.debug(f"Crossref SI download failed: {link_url}: {e}")
        except Exception as e:
            logger.warning(f"Crossref API strategy failed: {e}")

    # ── Strategy 3: Search figshare by DOI ────────────────────────────
    if not results:
        try:
            # figshare search API: search for articles linked to this DOI
            figshare_search = (
                f"https://api.figshare.com/v2/articles/search"
                f"?search_for={doi}"
            )
            resp = session.get(figshare_search, timeout=timeout)
            resp.raise_for_status()
            articles = resp.json()

            for article in articles[:3]:  # Check top 3 results
                article_id = article.get("id")
                if not article_id:
                    continue
                # Verify this article is linked to our DOI
                article_url = f"https://api.figshare.com/v2/articles/{article_id}"
                try:
                    art_resp = session.get(article_url, timeout=timeout)
                    art_resp.raise_for_status()
                    art_data = art_resp.json()
                    # Check if DOI matches
                    art_doi = art_data.get("doi", "")
                    if art_doi and doi.lower() in art_doi.lower():
                        # Download files from this figshare article
                        for file_info in art_data.get("files", []):
                            file_name = file_info.get("name", f"figshare_{article_id}_file")
                            file_url = (
                                file_info.get("download_url")
                                or f"https://figshare.com/ndownloader/files/{file_info.get('id', '')}"
                            )
                            try:
                                dl_resp = session.get(file_url, timeout=timeout, stream=True)
                                dl_resp.raise_for_status()
                                safe_name = re.sub(r'[<>:"/\\|?*]', '_', file_name)
                                file_path = output_path / safe_name
                                with open(file_path, "wb") as f:
                                    for chunk in dl_resp.iter_content(chunk_size=8192):
                                        f.write(chunk)
                                results.append({
                                    "url": file_url,
                                    "label": f"Figshare: {file_name}",
                                    "filename": safe_name,
                                    "filepath": str(file_path.absolute()),
                                    "size_bytes": file_path.stat().st_size,
                                    "downloaded": True,
                                    "source": "figshare_doi_search",
                                })
                                logger.info(f"Downloaded figshare SI via DOI: {safe_name}")
                            except Exception as e:
                                logger.debug(f"Figshare file download failed: {e}")
                except Exception as e:
                    logger.debug(f"Figshare article check failed: {e}")
        except Exception as e:
            logger.warning(f"Figshare DOI search strategy failed: {e}")

    if not results:
        logger.info(f"No SI files resolved for DOI: {doi}")

    return results


def _guess_extension(content_type: str) -> str:
    """Guess file extension from content type."""
    mapping = {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        "application/vnd.ms-excel": ".xls",
        "text/csv": ".csv",
        "application/pdf": ".pdf",
        "application/zip": ".zip",
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "text/plain": ".txt",
        "application/json": ".json",
        "application/octet-stream": ".bin",
    }
    return mapping.get(content_type.split(";")[0].strip(), ".bin")


def parse_si_data(
    downloaded_files: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Parse downloaded SI files into structured data for cross-figure comparison.

    Handles:
    - XLSX/Excel: reads all sheets, looks for group-value columns
    - CSV: reads as DataFrame, looks for group-value structure
    - TXT/other: attempts basic numeric extraction

    Args:
        downloaded_files: Output from download_si_files().

    Returns:
        List of parsed datasets, each compatible with CrossFigureDataComparisonTool
        input format: {label, group_labels, values, source_file}.
    """
    datasets = []

    for dl in downloaded_files:
        if not dl.get("downloaded"):
            continue
        # Handle figshare multi-file downloads
        if "files" in dl:
            for sub_file in dl["files"]:
                if sub_file.get("downloaded"):
                    _parse_file(sub_file, datasets)
        else:
            _parse_file(dl, datasets)

    logger.info(f"Parsed {len(datasets)} datasets from SI files")
    return datasets


def _parse_file(dl: Dict[str, Any], datasets: List[Dict[str, Any]]):
    """Parse a single downloaded file into datasets."""
    filepath = dl.get("filepath")
    if not filepath or not os.path.exists(filepath):
        return
    filename = dl.get("filename", os.path.basename(filepath))
    try:
        if filepath.endswith(('.xlsx', '.xls')):
            datasets.extend(_parse_excel(filepath, filename))
        elif filepath.endswith('.csv'):
            datasets.extend(_parse_csv(filepath, filename))
        elif filepath.endswith('.txt'):
            datasets.extend(_parse_text_data(filepath, filename))
    except Exception as e:
        logger.warning(f"Failed to parse SI file {filename}: {e}")


def _parse_excel(filepath: str, filename: str) -> List[Dict[str, Any]]:
    """Parse Excel SI file into structured datasets."""
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas not installed — cannot parse Excel files")
        return []

    datasets = []
    try:
        xl = pd.ExcelFile(filepath)
    except Exception as e:
        logger.warning(f"Cannot open Excel file {filename}: {e}")
        return []

    for sheet_name in xl.sheet_names:
        try:
            df = pd.read_excel(filepath, sheet_name=sheet_name)
        except Exception:
            continue
        if df.empty or df.shape[1] < 2:
            continue

        first_col = df.iloc[:, 0]
        if first_col.dtype == 'object':
            group_labels = [str(v).strip() for v in first_col.dropna().tolist()]
        else:
            group_labels = [f"Row_{i}" for i in range(len(df))]

        for col_idx in range(1, df.shape[1]):
            col_data = df.iloc[:, col_idx]
            col_header = str(df.columns[col_idx]) if col_idx < len(df.columns) else f"Col_{col_idx}"
            numeric_vals = pd.to_numeric(col_data, errors='coerce').dropna().tolist()
            if len(numeric_vals) >= 2:
                datasets.append({
                    "label": f"{filename}/{sheet_name}/{col_header}",
                    "group_labels": group_labels[:len(numeric_vals)],
                    "values": numeric_vals,
                    "source_file": filename,
                    "source_sheet": sheet_name,
                    "source_column": col_header,
                })
    return datasets


def _parse_csv(filepath: str, filename: str) -> List[Dict[str, Any]]:
    """Parse CSV SI file into structured datasets."""
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas not installed")
        return []

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logger.warning(f"Cannot read CSV {filename}: {e}")
        return []

    if df.empty or df.shape[1] < 2:
        return []

    datasets = []
    first_col = df.iloc[:, 0]
    if first_col.dtype == 'object':
        group_labels = [str(v).strip() for v in first_col.dropna().tolist()]
    else:
        group_labels = [f"Row_{i}" for i in range(len(df))]

    for col_idx in range(1, df.shape[1]):
        col_data = df.iloc[:, col_idx]
        col_header = str(df.columns[col_idx]) if col_idx < len(df.columns) else f"Col_{col_idx}"
        numeric_vals = pd.to_numeric(col_data, errors='coerce').dropna().tolist()
        if len(numeric_vals) >= 2:
            datasets.append({
                "label": f"{filename}/{col_header}",
                "group_labels": group_labels[:len(numeric_vals)],
                "values": numeric_vals,
                "source_file": filename,
                "source_column": col_header,
            })
    return datasets


def _parse_text_data(filepath: str, filename: str) -> List[Dict[str, Any]]:
    """Parse plain-text data file (tab/space separated values)."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception:
        return []
    if not lines:
        return []

    data_rows = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('//'):
            continue
        parts = re.split(r'[\t,;|]\s*', line)
        numeric_parts = []
        text_parts = []
        for p in parts:
            try:
                numeric_parts.append(float(p))
            except ValueError:
                if p.strip():
                    text_parts.append(p.strip())
        if numeric_parts:
            data_rows.append({"text": text_parts, "values": numeric_parts})

    if not data_rows:
        return []

    group_labels = [r["text"][0] if r["text"] else f"Row_{i}"
                    for i, r in enumerate(data_rows)]
    max_cols = max(len(r["values"]) for r in data_rows)
    datasets = []
    for col_idx in range(max_cols):
        col_values = []
        for i, row in enumerate(data_rows):
            if col_idx < len(row["values"]):
                col_values.append(row["values"][col_idx])
        if len(col_values) >= 2:
            datasets.append({
                "label": f"{filename}/Col_{col_idx + 1}",
                "group_labels": group_labels[:len(col_values)],
                "values": col_values,
                "source_file": filename,
            })
    return datasets


def download_and_parse_si_data(
    pdf_text: str,
    output_dir: str,
    doi: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full pipeline: extract SI links from PDF text, download files, parse data.

    Args:
        pdf_text: Full text extracted from the PDF document.
        output_dir: Directory for downloaded files.
        doi: Optional paper DOI. Used to resolve S* File labels that don't
             have direct URLs in the PDF text (common in PLOS ONE papers).

    Returns:
        Dict with 'si_links', 'downloads', 'datasets', 'unresolved_si_labels',
        ready for CrossFigureDataComparisonTool input.
    """
    links = extract_si_links_from_pdf_text(pdf_text)

    # Extract unresolved SI labels (S* File mentions without URLs)
    unresolved_labels = []
    for link in links:
        if "_unresolved_si_labels" in link:
            unresolved_labels = link.pop("_unresolved_si_labels")
            break
    # Clean up empty "unresolved_si_labels" placeholder entries
    links = [l for l in links if l.get("source") != "unresolved_si_labels"]

    if not links and not unresolved_labels:
        return {
            "si_links": [], "downloads": [], "datasets": [],
            "unresolved_si_labels": [],
            "note": "No supplementary information links found in PDF text.",
        }

    # ── Resolve SI labels without URLs via DOI ────────────────────────
    doi_downloads = []
    if unresolved_labels and doi:
        logger.info(
            f"Attempting DOI-based SI resolution for {len(unresolved_labels)} "
            f"labels: {[ul['label'] for ul in unresolved_labels]}"
        )
        doi_downloads = _resolve_si_from_doi(doi, output_dir)

    # ── Download from found URLs ─────────────────────────────────────
    downloads = download_si_files(links, output_dir) if links else []

    # ── Combine all downloads ────────────────────────────────────────
    all_downloads = downloads + doi_downloads
    datasets = parse_si_data(all_downloads)

    result = {
        "si_links": links,
        "downloads": all_downloads,
        "datasets": datasets,
        "total_datasets": len(datasets),
        "unresolved_si_labels": [
            {"label": ul["label"], "file_type": ul["file_type"],
             "resolved": any(ul["label"].lower() in str(d.get("label", "")).lower()
                            for d in doi_downloads)}
            for ul in unresolved_labels
        ],
    }

    # Note which labels couldn't be resolved
    still_unresolved = [
        ul for ul in result["unresolved_si_labels"] if not ul["resolved"]
    ]
    if still_unresolved:
        result["note"] = (
            f"{len(still_unresolved)} SI label(s) found in PDF text but "
            f"could not be resolved to download URLs: "
            f"{[ul['label'] for ul in still_unresolved]}. "
            f"Try downloading manually from the journal website."
        )

    return result
