"""XLSX 原始数据加载测试。"""

from openpyxl import Workbook

from academic_fraud_detector.utils.raw_data_loader import load_raw_data_files


def test_load_raw_data_files_extracts_column_and_decimal_metadata(tmp_path):
    xlsx_path = tmp_path / "Source Data Fig.1.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "Fig1"
    ws["A1"] = "Sample"
    ws["B1"] = "Treatment response"
    for row, value in enumerate([1.25, 2.35, 3.45, 4.55], start=2):
        ws.cell(row=row, column=1).value = f"S{row - 1}"
        cell = ws.cell(row=row, column=2)
        cell.value = value
        cell.number_format = "0.00"
    wb.save(xlsx_path)

    payload = load_raw_data_files([str(xlsx_path)])

    assert payload["profile"]["loaded_file_count"] == 1
    assert payload["profile"]["sheet_count"] == 1
    assert payload["profile"]["dataset_count"] >= 1

    column_dataset = next(
        dataset for dataset in payload["datasets"]
        if dataset["source"]["orientation"] == "column" and dataset["source"]["range"] == "B2:B5"
    )
    assert column_dataset["values"] == [1.25, 2.35, 3.45, 4.55]
    assert column_dataset["raw_values"] == ["1.25", "2.35", "3.45", "4.55"]
    assert column_dataset["decimal_places"] == [2, 2, 2, 2]
    assert column_dataset["last_decimal_digits"] == [5, 5, 5, 5]
    assert column_dataset["decimal_suffixes"]["2"] == ["25", "35", "45", "55"]
    assert column_dataset["row_labels"] == ["S1", "S2", "S3", "S4"]


def test_load_raw_data_files_uses_excel_percent_display_for_decimal_metadata(tmp_path):
    xlsx_path = tmp_path / "percent.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "Percent"
    ws["A1"] = "Sample"
    ws["B1"] = "Rate"
    for row, value in enumerate([0.123, 0.456, 0.789], start=2):
        ws.cell(row=row, column=1).value = f"S{row - 1}"
        cell = ws.cell(row=row, column=2)
        cell.value = value
        cell.number_format = "0.0%"
    wb.save(xlsx_path)

    payload = load_raw_data_files([str(xlsx_path)])
    column_dataset = next(
        dataset for dataset in payload["datasets"]
        if dataset["source"]["orientation"] == "column" and dataset["source"]["range"] == "B2:B4"
    )

    assert column_dataset["raw_values"] == ["12.3%", "45.6%", "78.9%"]
    assert column_dataset["last_decimal_digits"] == [3, 6, 9]


def test_load_raw_data_files_counts_gaps_inside_reported_range(tmp_path):
    xlsx_path = tmp_path / "gaps.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "Gaps"
    ws["B1"] = "Signal"
    ws["B2"] = 1.1
    ws["B3"] = "N/A"
    ws["B4"] = 2.2
    ws["B6"] = 3.3
    wb.save(xlsx_path)

    payload = load_raw_data_files([str(xlsx_path)])
    column_dataset = next(
        dataset for dataset in payload["datasets"]
        if dataset["source"]["orientation"] == "column" and dataset["source"]["range"] == "B2:B6"
    )

    assert column_dataset["missing_count"] == 2
    assert column_dataset["source"]["analyzed_cells"] == ["B2", "B4", "B6"]


def test_load_raw_data_files_reports_missing_file(tmp_path):
    missing = tmp_path / "missing.xlsx"

    payload = load_raw_data_files([str(missing)])

    assert payload["profile"]["loaded_file_count"] == 0
    assert payload["profile"]["errors"]
    assert payload["files"][0]["status"] == "error"
