"""
Table extraction from PDFs and HTML.

Uses camelot-py for PDF table extraction and BeautifulSoup for HTML tables.
Extracted data feeds the statistical analysis tools (Benford, GRIM, etc.).
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PAGE_MARKER_RE = re.compile(r"\[Page\s+(\d+)\]", re.IGNORECASE)
NUMBER_RE = r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
SECTION_ALIASES = {
    "abstract": "Abstract",
    "summary": "Abstract",
    "摘要": "Abstract",
    "introduction": "Introduction",
    "background": "Introduction",
    "引言": "Introduction",
    "前言": "Introduction",
    "methods": "Methods",
    "materials and methods": "Methods",
    "methodology": "Methods",
    "experimental procedures": "Methods",
    "方法": "Methods",
    "材料与方法": "Methods",
    "results": "Results",
    "result": "Results",
    "结果": "Results",
    "discussion": "Discussion",
    "讨论": "Discussion",
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
    "结论": "Conclusion",
    "references": "References",
    "reference": "References",
    "参考文献": "References",
    "figure legends": "Figure legends",
    "figures": "Figures",
    "tables": "Tables",
}


def extract_pdf_tables(
    pdf_content: bytes,
    page_numbers: Optional[str] = None,
    method: str = "lattice",
) -> List[Dict[str, Any]]:
    """
    Extract tables from a PDF byte buffer using Camelot.

    Args:
        pdf_content: Raw PDF bytes.
        page_numbers: Comma-separated page numbers, e.g. '1,2,3' or '1-5'.
        method: Camelot extraction method: 'lattice' (bordered tables) or 'stream' (borderless).

    Returns:
        List of dicts, each with 'page', 'table_number', 'rows', 'columns', 'data'.
    """
    try:
        import camelot
    except ImportError:
        logger.error("camelot-py is not installed. Install with: pip install camelot-py[cv]")
        return []

    # Write PDF to temp file (camelot needs a file path)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_content)
        tmp_path = tmp.name

    try:
        tables = camelot.read_pdf(
            tmp_path,
            pages=page_numbers or "all",
            flavor=method,
            suppress_stdout=True,
        )

        results = []
        for i, table in enumerate(tables):
            parsed = table.df.values.tolist()  # DataFrame -> list of lists
            # Remove empty rows/columns
            parsed = [[str(cell).strip() if cell else "" for cell in row] for row in parsed]
            parsed = [row for row in parsed if any(row)]
            results.append({
                "table_number": i + 1,
                "page": table.page,
                "rows": len(parsed),
                "columns": len(parsed[0]) if parsed else 0,
                "data": parsed,
                "accuracy": float(table.parsing_report.get("accuracy", 0)),
            })

        return results

    finally:
        os.unlink(tmp_path)


def extract_html_tables(html_content: str) -> List[Dict[str, Any]]:
    """
    Extract tables from HTML content using BeautifulSoup.

    Args:
        html_content: Raw HTML string.

    Returns:
        List of dicts with 'table_number', 'rows', 'columns', 'data'.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.error("BeautifulSoup is not installed.")
        return []

    soup = BeautifulSoup(html_content, "html.parser")
    tables = soup.find_all("table")
    results = []

    for i, table in enumerate(tables):
        rows = table.find_all("tr")
        data = []
        for row in rows:
            cells = row.find_all(["td", "th"])
            data.append([cell.get_text(strip=True) for cell in cells])

        # Filter empty rows
        data = [row for row in data if any(row)]
        if not data:
            continue

        results.append({
            "table_number": i + 1,
            "rows": len(data),
            "columns": len(data[0]) if data else 0,
            "data": data,
        })

    return results


def extract_numeric_values(table_data: List[List[str]]) -> List[float]:
    """
    Extract all numeric values from parsed table data.

    Attempts to parse cells as floats, handling common formatting:
    - Commas as thousands separators: '1,234' -> 1234.0
    - Parentheses for negative: '(5.2)' -> -5.2
    - Percentage signs: '12.5%' -> 0.125
    - Scientific notation: '1.23e-4'

    Args:
        table_data: List of rows, each row is a list of cell strings.

    Returns:
        List of successfully parsed float values.
    """
    numbers = []
    for row in table_data:
        for cell in row:
            cell = cell.strip()
            if not cell:
                continue

            # Remove commas (thousands separators)
            cleaned = cell.replace(",", "")

            # Handle percentage
            is_pct = cleaned.endswith("%")
            if is_pct:
                cleaned = cleaned[:-1]

            # Handle parentheses for negative numbers
            is_negative = cleaned.startswith("(") and cleaned.endswith(")")
            if is_negative:
                cleaned = cleaned[1:-1]

            # Remove common noise: units, asterisks, etc.
            cleaned = cleaned.rstrip("*^{}[] ")

            try:
                value = float(cleaned)
                if is_pct:
                    value = value / 100.0
                if is_negative:
                    value = -value
                numbers.append(value)
            except ValueError:
                continue

    return numbers


def _page_ranges(text: str) -> list[dict[str, int]]:
    """Return text intervals covered by explicit [Page N] markers."""
    matches = list(PAGE_MARKER_RE.finditer(text or ""))
    ranges: list[dict[str, int]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        ranges.append({"page": int(match.group(1)), "start": start, "end": end})
    return ranges


def _page_for_index(page_ranges: list[dict[str, int]], index: int) -> int | None:
    for item in page_ranges:
        if item["start"] <= index < item["end"]:
            return item["page"]
    if page_ranges and index < page_ranges[0]["start"]:
        return page_ranges[0]["page"]
    return None


def _normalize_heading(text: str) -> str:
    heading = re.sub(r"^#+\s*", "", text.strip())
    heading = re.sub(r"^[\dIVXivx]+[.)\s-]+", "", heading)
    heading = heading.strip(" ：:.-—\t")
    return heading


def _detect_section_at(text: str, index: int) -> str | None:
    """Best-effort section lookup by scanning recent headings before index."""
    window_start = max(0, index - 15000)
    prefix = text[window_start:index]
    best_section: str | None = None
    for line in prefix.splitlines():
        stripped = line.strip()
        if not stripped or PAGE_MARKER_RE.fullmatch(stripped):
            continue
        candidate = _normalize_heading(stripped)
        lowered = candidate.lower()
        if lowered in SECTION_ALIASES:
            best_section = SECTION_ALIASES[lowered]
            continue
        markdown_heading = stripped.startswith("#") and len(candidate) <= 80
        plain_heading = len(candidate) <= 45 and not candidate.endswith((".", ",", ";"))
        if markdown_heading or plain_heading:
            for alias, canonical in SECTION_ALIASES.items():
                if re.search(rf"\b{re.escape(alias)}\b", lowered):
                    best_section = canonical
                    break
    return best_section


def _source_for_match(
    text: str,
    match: re.Match[str],
    *,
    document_role: str = "main_pdf",
    file_name: str | None = None,
    extraction_method: str = "unknown",
    page_ranges: list[dict[str, int]] | None = None,
    context_chars: int = 80,
) -> dict[str, Any]:
    ranges = _page_ranges(text) if page_ranges is None else page_ranges
    page = _page_for_index(ranges, match.start())
    if ranges:
        page_detection = "page_marker"
    elif text:
        page_detection = "missing_page_marker"
    else:
        page_detection = "not_available"
    return {
        "document_role": document_role,
        "file_name": file_name,
        "char_start": match.start(),
        "char_end": match.end(),
        "page": page,
        "section": _detect_section_at(text, match.start()),
        "context_before": text[max(0, match.start() - context_chars):match.start()],
        "context_after": text[match.end():match.end() + context_chars],
        "extraction_method": extraction_method,
        "page_detection": page_detection,
    }


def extract_p_values_with_sources(
    text: str,
    *,
    document_role: str = "main_pdf",
    file_name: str | None = None,
    extraction_method: str = "unknown",
) -> List[Dict[str, Any]]:
    """Extract p-values with source spans, preserving the legacy float API separately."""
    pattern = re.compile(
        rf"\b(?:p\s*[-\s]?value|p)\s*"
        rf"(?P<operator><=|>=|=|<|>|≤|≥)\s*"
        rf"(?P<value>{NUMBER_RE})(?![A-Za-z0-9_])",
        re.IGNORECASE,
    )
    records: list[dict[str, Any]] = []
    seen_spans: set[tuple[int, int]] = set()
    page_ranges = _page_ranges(text)
    for match in pattern.finditer(text or ""):
        if match.span() in seen_spans:
            continue
        seen_spans.add(match.span())
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        if not 0 <= value <= 1:
            continue
        source = _source_for_match(
            text,
            match,
            document_role=document_role,
            file_name=file_name,
            extraction_method=extraction_method,
            page_ranges=page_ranges,
        )
        records.append({
            "value": value,
            "operator": match.group("operator"),
            "raw_text": match.group(0),
            "context": source["context_before"] + match.group(0) + source["context_after"],
            "source": source,
        })
    return records


def extract_p_values(text: str) -> List[float]:
    """
    Extract p-values from academic text.

    Keeps the historical return contract: a list of floats.
    """
    return [item["value"] for item in extract_p_values_with_sources(text)]


def extract_means_and_sds_with_sources(
    text: str,
    *,
    document_role: str = "main_pdf",
    file_name: str | None = None,
    extraction_method: str = "unknown",
) -> List[Dict[str, Any]]:
    """Extract mean±SD style reported statistics with source spans."""
    patterns = [
        (
            "mean_plusminus_sd",
            re.compile(
                rf"(?P<mean>{NUMBER_RE})\s*(?:±|\+/-|\\pm)\s*(?P<sd>{NUMBER_RE})"
            ),
        ),
        (
            "mean_label_sd_label",
            re.compile(
                rf"\b(?:M|Mean|mean)\s*[=:]\s*(?P<mean>{NUMBER_RE})"
                rf"[^\n.;]{{0,100}}?\b(?:SD|sd|std|Std|standard deviation)"
                rf"(?:\s*[=:])?\s*(?P<sd>{NUMBER_RE})"
            ),
        ),
        (
            "mean_prose_sd_parenthetical",
            re.compile(
                rf"\b(?:Mean|mean)\b[^\n.;]{{0,80}}?\b(?:was|were|is|are)\s+"
                rf"(?P<mean>{NUMBER_RE})[\s,]*(?:\()?\s*"
                rf"(?:SD|sd|std|Std|standard deviation)(?:\s*[=:])?\s*(?P<sd>{NUMBER_RE})"
            ),
        ),
    ]
    results: list[dict[str, Any]] = []
    seen_spans: set[tuple[int, int]] = set()
    page_ranges = _page_ranges(text)
    for pattern_type, pattern in patterns:
        for match in pattern.finditer(text or ""):
            if match.span() in seen_spans:
                continue
            seen_spans.add(match.span())
            try:
                mean = float(match.group("mean"))
                sd = float(match.group("sd"))
            except ValueError:
                continue
            if sd < 0:
                continue
            source = _source_for_match(
                text,
                match,
                document_role=document_role,
                file_name=file_name,
                extraction_method=extraction_method,
                page_ranges=page_ranges,
            )
            results.append({
                "mean": mean,
                "sd": sd,
                "context": source["context_before"] + match.group(0) + source["context_after"],
                "raw_text": match.group(0),
                "pattern_type": pattern_type,
                "source": source,
            })
    results.sort(key=lambda item: item.get("source", {}).get("char_start", 0))
    return results


def extract_means_and_sds(text: str) -> List[Dict[str, Any]]:
    """
    Extract mean ± SD patterns from academic text.

    Returns dictionaries that keep the historical 'mean', 'sd', and 'context'
    keys, with additional source metadata for downstream grounding.
    """
    return extract_means_and_sds_with_sources(text)


def extract_sample_sizes_with_sources(
    text: str,
    *,
    document_role: str = "main_pdf",
    file_name: str | None = None,
    extraction_method: str = "unknown",
) -> List[Dict[str, Any]]:
    """Extract simple n=... statements with source spans."""
    pattern = re.compile(r"\b(?P<label>[nN])\s*=\s*(?P<n>\d{1,4})\b")
    records: list[dict[str, Any]] = []
    page_ranges = _page_ranges(text)
    for match in pattern.finditer(text or ""):
        n = int(match.group("n"))
        if n <= 0:
            continue
        source = _source_for_match(
            text,
            match,
            document_role=document_role,
            file_name=file_name,
            extraction_method=extraction_method,
            page_ranges=page_ranges,
        )
        records.append({
            "n": n,
            "raw_text": match.group(0),
            "context": source["context_before"] + match.group(0) + source["context_after"],
            "source": source,
        })
    return records


def extract_paper_claims(
    text: str,
    *,
    file_name: str | None = None,
    document_role: str = "main_pdf",
    extraction_method: str = "unknown",
    max_claims: int = 300,
) -> dict[str, Any]:
    """Extract deterministic paper claim anchors for later LLM contextual review."""
    claims: list[dict[str, Any]] = []
    warnings: list[str] = []

    def add_claim(claim_type: str, values: dict[str, Any], raw_text: str, source: dict[str, Any]) -> None:
        claims.append({
            "claim_id": "",
            "claim_type": claim_type,
            "values": values,
            "raw_text": raw_text,
            "normalized_text": " ".join(raw_text.split()),
            "source": source,
            "confidence": "deterministic_regex",
            "reportable": False,
        })

    for item in extract_means_and_sds_with_sources(
        text,
        document_role=document_role,
        file_name=file_name,
        extraction_method=extraction_method,
    ):
        add_claim(
            "reported_mean_sd",
            {"mean": item["mean"], "sd": item["sd"]},
            item["raw_text"],
            item["source"],
        )
    for item in extract_p_values_with_sources(
        text,
        document_role=document_role,
        file_name=file_name,
        extraction_method=extraction_method,
    ):
        add_claim(
            "reported_p_value",
            {"p_value": item["value"], "operator": item["operator"]},
            item["raw_text"],
            item["source"],
        )
    for item in extract_sample_sizes_with_sources(
        text,
        document_role=document_role,
        file_name=file_name,
        extraction_method=extraction_method,
    ):
        add_claim("reported_n", {"n": item["n"]}, item["raw_text"], item["source"])

    claims.sort(key=lambda item: item.get("source", {}).get("char_start", 0))
    total_claim_count = len(claims)
    if total_claim_count > max_claims:
        warnings.append(f"论文 claim 超过 {max_claims} 条，仅保留文档顺序前 {max_claims} 条。")
        claims = claims[:max_claims]
    for index, item in enumerate(claims, start=1):
        item["claim_id"] = f"PCL-{index:04d}"

    counts = _count_claim_types(claims)
    return {
        "schema_version": "paper_claims.v1",
        "claims": claims,
        "summary": {
            "claim_count": len(claims),
            "reported_stat_count": counts.get("reported_mean_sd", 0) + counts.get("reported_p_value", 0),
            "reported_mean_sd_count": counts.get("reported_mean_sd", 0),
            "reported_p_value_count": counts.get("reported_p_value", 0),
            "reported_n_count": counts.get("reported_n", 0),
        },
        "warnings": warnings,
    }


def _count_claim_types(claims: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in claims:
        claim_type = str(item.get("claim_type", ""))
        counts[claim_type] = counts.get(claim_type, 0) + 1
    return counts
