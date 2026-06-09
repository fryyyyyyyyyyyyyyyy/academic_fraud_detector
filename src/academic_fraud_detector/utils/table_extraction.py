"""
Table extraction from PDFs and HTML.

Uses camelot-py for PDF table extraction and BeautifulSoup for HTML tables.
Extracted data feeds the statistical analysis tools (Benford, GRIM, etc.).
"""

import logging
from typing import List, Optional, Dict, Any
import tempfile
import os

import requests

logger = logging.getLogger(__name__)


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


def extract_p_values(text: str) -> List[float]:
    """
    Extract p-values from academic text.

    Looks for patterns like:
    - 'p = 0.032'
    - 'p < 0.001'
    - 'P = 0.05'
    - 'p-value = 0.123'

    Args:
        text: Paper text to search.

    Returns:
        List of extracted p-values as floats.
    """
    import re

    patterns = [
        r"[pP]\s*[=<>]\s*(\d+\.\d+)",
        r"[pP][- ]?value\s*[=<>]\s*(\d+\.\d+)",
        r"[pP]\s*[=<>]\s*\.?(\d+)",  # p = .05 notation
    ]

    values = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for m in matches:
            try:
                v = float(m)
                if 0 <= v <= 1:
                    values.append(v)
            except ValueError:
                continue

    return values


def extract_means_and_sds(text: str) -> List[Dict[str, Any]]:
    """
    Extract mean ± SD patterns from academic text.

    Looks for patterns like:
    - '23.5 ± 2.1'
    - 'M = 23.5, SD = 2.1'
    - 'mean = 23.5 (SD = 2.1)'

    Args:
        text: Paper text to search.

    Returns:
        List of dicts with 'mean', 'sd', 'context' keys.
    """
    import re

    results = []

    # Pattern: number ± number
    pattern1 = r"(\d+\.?\d*)\s*[±±]\s*(\d+\.?\d*)"
    for match in re.finditer(pattern1, text):
        results.append({
            "mean": float(match.group(1)),
            "sd": float(match.group(2)),
            "context": text[max(0, match.start() - 50):match.end() + 50],
        })

    # Pattern: M = number, SD = number
    pattern2 = r"[mM](?:ean)?\s*=\s*(\d+\.?\d*).*?(?:SD|sd|std)(?:\s*=|:)?\s*(\d+\.?\d*)"
    for match in re.finditer(pattern2, text):
        results.append({
            "mean": float(match.group(1)),
            "sd": float(match.group(2)),
            "context": text[max(0, match.start() - 30):match.end() + 30],
        })

    return results
