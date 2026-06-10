"""本地案例目录发现工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

PDF_HINTS = ("paper", "article", "manuscript", "main", "论文", "正文")
RAW_DATA_EXTENSIONS = {".xlsx", ".xlsm", ".xltx", ".xltm"}


def _file_entry(path: Path) -> dict[str, Any]:
    """将本地文件转换为可 JSON 序列化的 manifest 条目。"""
    try:
        size_bytes = path.stat().st_size
    except OSError:
        size_bytes = 0
    return {
        "path": str(path),
        "name": path.name,
        "suffix": path.suffix.lower(),
        "size_bytes": size_bytes,
    }


def _select_primary_pdf(pdf_paths: list[Path]) -> tuple[Path | None, list[str]]:
    """选择最可能的论文主体 PDF，并返回选择过程警告。"""
    warnings: list[str] = []
    if not pdf_paths:
        return None, warnings
    if len(pdf_paths) == 1:
        return pdf_paths[0], warnings

    hinted = [p for p in pdf_paths if any(hint in p.stem.lower() for hint in PDF_HINTS)]
    if hinted:
        selected = sorted(hinted, key=lambda p: p.name.lower())[0]
        warnings.append(
            f"案例目录包含 {len(pdf_paths)} 个 PDF；按文件名提示选择 {selected.name}。"
        )
        return selected, warnings

    selected = max(pdf_paths, key=lambda p: p.stat().st_size if p.exists() else 0)
    warnings.append(
        f"案例目录包含 {len(pdf_paths)} 个 PDF；未发现主体论文命名提示，选择最大文件 {selected.name}。"
    )
    return selected, warnings


def discover_case_folder(case_dir: str) -> dict[str, Any]:
    """
    发现本地案例目录中的论文 PDF 和 XLSX 原始数据文件。

    函数只返回结构化 manifest，不直接读取 PDF/XLSX 内容。调用方应检查 errors；
    若存在错误，本地案例流程不应继续执行。
    """
    root = Path(case_dir).expanduser()
    manifest: dict[str, Any] = {
        "case_dir": str(root),
        "case_name": root.name,
        "pdf_files": [],
        "selected_pdf": None,
        "raw_data_files": [],
        "warnings": [],
        "errors": [],
    }

    if not root.exists():
        manifest["errors"].append(f"案例目录不存在: {root}")
        return manifest
    if not root.is_dir():
        manifest["errors"].append(f"local_case 必须是目录: {root}")
        return manifest

    pdf_paths = sorted(
        [p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"],
        key=lambda p: p.name.lower(),
    )
    raw_paths = sorted(
        [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in RAW_DATA_EXTENSIONS],
        key=lambda p: p.name.lower(),
    )

    manifest["pdf_files"] = [_file_entry(p) for p in pdf_paths]
    manifest["raw_data_files"] = [_file_entry(p) for p in raw_paths]

    selected_pdf, warnings = _select_primary_pdf(pdf_paths)
    manifest["warnings"].extend(warnings)
    if selected_pdf is not None:
        manifest["selected_pdf"] = str(selected_pdf)

    if not pdf_paths:
        manifest["errors"].append("案例目录中未发现论文 PDF 文件。")
    if not raw_paths:
        manifest["errors"].append("案例目录中未发现 .xlsx/.xlsm/.xltx/.xltm 原始数据文件。")

    return manifest
