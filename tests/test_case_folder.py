"""本地案例目录发现测试。"""

from academic_fraud_detector.utils.case_folder import discover_case_folder


def test_discover_case_folder_valid_chinese_path(tmp_path):
    case_dir = tmp_path / "上海大学院长，代表作论文，严重造假！"
    case_dir.mkdir()
    pdf_path = case_dir / "上海大学院长，代表作论文，严重造假！.pdf"
    xlsx_path = case_dir / "Source Data Fig.1.xlsx"
    pdf_path.write_bytes(b"%PDF-1.7")
    xlsx_path.write_bytes(b"placeholder")

    manifest = discover_case_folder(str(case_dir))

    assert manifest["errors"] == []
    assert manifest["selected_pdf"] == str(pdf_path)
    assert manifest["pdf_files"][0]["name"] == pdf_path.name
    assert manifest["raw_data_files"][0]["path"] == str(xlsx_path)


def test_discover_case_folder_requires_directory(tmp_path):
    file_path = tmp_path / "paper.pdf"
    file_path.write_bytes(b"%PDF-1.7")

    manifest = discover_case_folder(str(file_path))

    assert manifest["errors"]
    assert "必须是目录" in manifest["errors"][0]


def test_discover_case_folder_requires_pdf_and_xlsx(tmp_path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()

    manifest = discover_case_folder(str(case_dir))

    assert any("PDF" in error for error in manifest["errors"])
    assert any("原始数据" in error for error in manifest["errors"])


def test_discover_case_folder_rejects_legacy_xls_without_supported_workbook(tmp_path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "paper.pdf").write_bytes(b"%PDF-1.7")
    (case_dir / "legacy.xls").write_bytes(b"placeholder")

    manifest = discover_case_folder(str(case_dir))

    assert any("原始数据" in error for error in manifest["errors"])
    assert manifest["raw_data_files"] == []


def test_discover_case_folder_selects_hinted_pdf_when_multiple(tmp_path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    main_pdf = case_dir / "main article.pdf"
    other_pdf = case_dir / "supplement.pdf"
    main_pdf.write_bytes(b"small")
    other_pdf.write_bytes(b"large" * 100)
    (case_dir / "data.xlsx").write_bytes(b"placeholder")

    manifest = discover_case_folder(str(case_dir))

    assert manifest["selected_pdf"] == str(main_pdf)
    assert manifest["warnings"]
