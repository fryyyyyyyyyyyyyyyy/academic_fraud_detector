"""
Academic Fraud Detection System — CLI Entry Point.

Usage:
    python -m academic_fraud_detector.main <paper_identifier> [identifier_type]

Examples:
    python -m academic_fraud_detector.main 10.1038/nature12345 doi
    python -m academic_fraud_detector.main 2301.12345 arxiv_id
    python -m academic_fraud_detector.main "Attention Is All You Need" title
    python -m academic_fraud_detector.main https://arxiv.org/abs/2301.12345 url
    python -m academic_fraud_detector.main /path/to/paper.pdf local_pdf

Output:
    Produces 3 files: .md (Markdown), .html (styled HTML), .pdf (PDF, if available)

Environment variables:
    MANAGER_MODEL       — LLM model for Lead Investigator
    AGENT_MODEL         — LLM model for specialist agents
    OPENAI_API_KEY      — API key
    OPENAI_API_BASE     — Custom API base URL (DeepSeek, etc.)
"""

import sys
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env from project root
load_dotenv()

# Workaround for Windows GBK terminal
if sys.platform == "win32":
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace") # type: ignore
        sys.stdout.reconfigure(encoding="utf-8", errors="replace") # type: ignore
    except Exception:
        pass

from .crew import AcademicFraudDetectionCrew

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# HTML Template for Report Rendering
# ═══════════════════════════════════════════════════════════════════════════

REPORT_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>学术诚信调查报告</title>
<style>
  @page {{
    size: A4;
    margin: 2cm 2.5cm;
    @bottom-center {{
      content: "第 " counter(page) " 页";
      font-size: 9pt;
      color: #999;
    }}
  }}
  body {{
    font-family: "Microsoft YaHei", "SimSun", "Noto Sans SC", sans-serif;
    font-size: 12pt;
    line-height: 1.8;
    color: #222;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
  }}
  h1 {{
    font-size: 22pt;
    text-align: center;
    border-bottom: 3px solid #1a5276;
    padding-bottom: 12px;
    margin-bottom: 8px;
    color: #1a5276;
  }}
  h2 {{
    font-size: 16pt;
    border-bottom: 2px solid #2980b9;
    padding-bottom: 6px;
    margin-top: 32px;
    color: #1a5276;
  }}
  h3 {{
    font-size: 13pt;
    margin-top: 24px;
    color: #2471a3;
  }}
  h4 {{
    font-size: 11pt;
    margin-top: 18px;
    color: #2c3e50;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 11pt;
  }}
  th {{
    background-color: #1a5276;
    color: white;
    padding: 8px 12px;
    text-align: left;
    font-weight: bold;
  }}
  td {{
    border: 1px solid #ccc;
    padding: 6px 12px;
  }}
  tr:nth-child(even) {{ background-color: #f2f4f7; }}
  blockquote {{
    border-left: 4px solid #2980b9;
    background: #eaf2f8;
    margin: 12px 0;
    padding: 8px 16px;
    color: #444;
  }}
  hr {{
    border: none;
    border-top: 1px solid #ddd;
    margin: 28px 0;
  }}
  strong {{ color: #1a3a4a; }}
  .risk-critical {{ color: #c0392b; font-weight: bold; }}
  .risk-high {{ color: #e67e22; font-weight: bold; }}
  .risk-medium {{ color: #d4ac0d; font-weight: bold; }}
  .risk-low {{ color: #27ae60; }}
  .risk-none {{ color: #7f8c8d; }}
  ul, ol {{ padding-left: 24px; }}
  li {{ margin: 4px 0; }}
  code {{
    background: #f4f4f4;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 10pt;
  }}
  .footer {{
    margin-top: 40px;
    padding-top: 12px;
    border-top: 1px solid #ccc;
    font-size: 9pt;
    color: #999;
    text-align: center;
    font-style: italic;
  }}
</style>
</head>
<body>
{body}
<p class="footer">
  本报告由 Academic Fraud Detection System (CrewAI) 自动生成。检测结果仅供参考，不作为最终学术不端判定依据。建议由相关领域专家进行人工复核。
</p>
</body>
</html>"""


def _extract_raw_text(result) -> str:
    """Extract raw text from CrewAI result object (handles multiple types)."""
    if hasattr(result, 'raw'):
        return str(result.raw)
    elif hasattr(result, 'json_dict'):
        return json.dumps(result.json_dict, ensure_ascii=False, indent=2)
    elif isinstance(result, str):
        return result
    else:
        return str(result)


def _extract_title_from_markdown(markdown_text: str) -> str:
    """Try to extract paper title from the markdown report."""
    for line in markdown_text.splitlines():
        line = line.strip()
        if line.startswith("**论文标题**"):
            match = re.search(r"\*\*论文标题\*\*\s*\|\s*(.+?)(?:\s*\|)?$", line)
            if match:
                return match.group(1).strip()
        if line.startswith("| **论文标题**"):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                return parts[2].strip().removeprefix("**").removesuffix("**")
    return ""


def _extract_risk_level_from_markdown(markdown_text: str) -> str:
    """Extract risk level from the markdown report."""
    for line in markdown_text.splitlines():
        if "风险等级" in line and "**" in line:
            match = re.search(r"\*\*风险等级\*\*\s*\|\s*(.+?)(?:\s*\|)?$", line)
            if match:
                return match.group(1).strip()
        if "| **风险等级**" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                return parts[2].strip().removeprefix("**").removesuffix("**")
    return ""


def _extract_risk_score_from_markdown(markdown_text: str) -> str:
    """Extract risk score from the markdown report."""
    for line in markdown_text.splitlines():
        if "风险评分" in line:
            match = re.search(r"\*\*风险评分\*\*\s*\|\s*(.+?)(?:\s*\|)?$", line)
            if match:
                return match.group(1).strip()
        if "| **风险评分**" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                return parts[2].strip().removeprefix("**").removesuffix("**")
    return ""


def _extract_executive_summary(markdown_text: str) -> list:
    """Extract executive summary bullets from markdown."""
    bullets = []
    in_summary = False
    for line in markdown_text.splitlines():
        if line.strip().startswith("## 执行摘要"):
            in_summary = True
            continue
        if in_summary:
            if line.strip().startswith("## "):
                break
            if re.match(r"^\d+\.\s+\*\*", line.strip()):
                bullets.append(line.strip())
            elif re.match(r"^\d+\.\s+", line.strip()):
                bullets.append(line.strip())
    return bullets


def _extract_recommendations(markdown_text: str) -> list:
    """Extract recommendations from markdown."""
    recs = []
    in_recs = False
    for line in markdown_text.splitlines():
        if line.strip().startswith("## 建议"):
            in_recs = True
            continue
        if in_recs:
            if line.strip().startswith("## "):
                break
            if re.match(r"^\d+\.\s+\*\*", line.strip()):
                recs.append(line.strip())
    return recs


def markdown_to_html(markdown_text: str) -> str:
    """Convert Markdown text to a styled HTML page."""
    import markdown as md_lib

    # Convert Markdown to HTML body
    md_extensions = ["tables", "fenced_code", "codehilite", "nl2br"]
    try:
        html_body = md_lib.markdown(markdown_text, extensions=md_extensions)
    except Exception:
        html_body = md_lib.markdown(markdown_text, extensions=["tables"])

    return REPORT_HTML_TEMPLATE.format(body=html_body)




def save_report_files(
    markdown_text: str,
    output_dir: Path,
    base_name: str,
) -> dict:
    """
    Save the report as .md and .html, then open HTML in browser.

    Returns a dict with paths:
        {"md": Path, "html": Path}
    """
    results = {}

    # 1. Save Markdown (.md)
    md_path = output_dir / f"{base_name}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)
    results["md"] = md_path
    logger.info(f"[MD]   Markdown 报告已保存: {md_path}")

    # 2. Generate and save HTML (.html)
    html_text = markdown_to_html(markdown_text)
    html_path = output_dir / f"{base_name}.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_text)
    results["html"] = html_path
    logger.info(f"[HTML] HTML 报告已保存: {html_path}")

    # 3. Auto-open HTML in browser
    try:
        import webbrowser
        webbrowser.open(str(html_path))
        logger.info(f"[HTML] 已在浏览器中打开报告。")
    except Exception:
        pass

    return results


def run_investigation(
    paper_identifier: str,
    identifier_type: str = "doi",
    output_dir: Optional[str] = None,
) -> dict:
    """
    Run a full academic fraud investigation on a target paper.

    Produces 3 report formats: .md (Markdown), .html (styled HTML), .pdf (PDF).

    Args:
        paper_identifier: DOI, arXiv ID, title, URL, or local PDF path.
        identifier_type: One of 'doi', 'arxiv_id', 'title', 'url', 'local_pdf'.
        output_dir: Directory for output files. Defaults to ./reports/

    Returns:
        Dict with keys: "markdown" (str), "files" (dict of paths), "elapsed" (float).
    """
    logger.info(f"[START] Starting investigation: {paper_identifier} (type: {identifier_type})")
    start_time = datetime.now()

    # ── Kickoff the crew ──
    is_local = (identifier_type == "local_pdf")
    crew_instance = AcademicFraudDetectionCrew(local_only=is_local).crew()
    inputs = {
        "paper_identifier": paper_identifier,
        "identifier_type": identifier_type,
    }

    try:
        result = crew_instance.kickoff(inputs=inputs)
    except Exception as e:
        logger.error(f"Investigation failed: {e}")
        raise

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"[DONE] Investigation completed in {elapsed:.1f} seconds")

    # ── Get raw markdown output ──
    markdown_text = _extract_raw_text(result)

    # ── Save reports in all formats ──
    if output_dir is None:
        output_dir = str(Path.cwd() / "reports")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    safe_id = paper_identifier.replace("/", "_").replace(":", "_").replace("\\", "_")[:80]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"investigation_{safe_id}_{timestamp}"
    # Also save JSON as raw backup
    json_path = output_path / f"{base_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"raw_markdown": markdown_text}, f, indent=2, ensure_ascii=False)
    logger.info(f"[JSON] Raw output backup saved to: {json_path}")

    files = save_report_files(markdown_text, output_path, base_name)

    # ── Print terminal summary ──
    title = _extract_title_from_markdown(markdown_text) or paper_identifier
    risk_level = _extract_risk_level_from_markdown(markdown_text)
    risk_score = _extract_risk_score_from_markdown(markdown_text)

    print("\n" + "=" * 70)
    print(f"  学术诚信调查报告")
    print("=" * 70)
    print(f"  论文       : {title[:55]}")
    if risk_level:
        print(f"  风险等级   : {risk_level}")
    if risk_score:
        print(f"  风险评分   : {risk_score}/100")
    print(f"  耗时       : {elapsed:.1f} 秒")
    print("-" * 70)
    print(f"  Markdown   : {files['md']}")
    print(f"  HTML       : {files['html']}  (已在浏览器中打开)")
    print("=" * 70 + "\n")

    # Print executive summary
    summary_bullets = _extract_executive_summary(markdown_text)
    if summary_bullets:
        print("执行摘要：")
        for b in summary_bullets[:5]:
            print(f"  {b}")
        print()

    # Print recommendations
    recs = _extract_recommendations(markdown_text)
    if recs:
        print("建议：")
        for r in recs[:5]:
            print(f"  {r}")
        print()

    return {
        "markdown": markdown_text,
        "files": files,
        "elapsed": elapsed,
    }


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\n参数：")
        print("  paper_identifier   — DOI, arXiv ID, 论文标题, URL, 或本地 PDF 路径")
        print("  identifier_type    — doi, arxiv_id, title, url, local_pdf 之一 (默认: doi)")
        print("\n选项：")
        print("  --pdf PATH         — 本地 PDF 快捷方式 (等同于 local_pdf)")
        print("  --output-dir DIR   — 输出目录 (默认: ./reports/)")
        print("  --quiet            — 静默模式")
        print("\n示例：")
        print("  python -m academic_fraud_detector.main 10.1038/nature12345 doi")
        print("  python -m academic_fraud_detector.main --pdf /path/to/paper.pdf")
        print("  python -m academic_fraud_detector.main /path/to/paper.pdf local_pdf")
        print("\n输出文件：")
        print("  .md   — 中文 Markdown 报告（主格式，可直接阅读）")
        print("  .html — 带样式的 HTML 报告（可在浏览器中打开并打印为 PDF）")
        print("  .pdf  — PDF 报告（需安装 weasyprint）")
        sys.exit(1)

    # Handle --pdf shorthand: --pdf /path/to/paper.pdf
    paper_id = sys.argv[1]
    id_type = "doi"  # default

    if paper_id == "--pdf":
        if len(sys.argv) < 3:
            print("[错误] --pdf 需要提供文件路径。")
            print("   示例: python -m academic_fraud_detector.main --pdf /path/to/paper.pdf")
            sys.exit(1)
        paper_id = sys.argv[2]
        id_type = "local_pdf"
    elif len(sys.argv) > 2 and not sys.argv[2].startswith("--"):
        id_type = sys.argv[2]

    # Parse optional flags
    output_dir = None
    quiet = False
    for i, arg in enumerate(sys.argv):
        if arg == "--output-dir" and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
        if arg == "--quiet":
            quiet = True

    # Early validation for local PDF files
    if id_type == "local_pdf":
        if not os.path.exists(paper_id):
            print(f"[错误] 文件不存在: {paper_id}")
            sys.exit(1)
        if not paper_id.lower().endswith(".pdf"):
            print(f"[错误] 文件必须是 PDF 格式: {paper_id}")
            sys.exit(1)
        print(f"[PDF] 正在加载本地 PDF: {paper_id}")

    if quiet:
        logging.getLogger().setLevel(logging.WARNING)
        os.environ["CREWAI_VERBOSE"] = "false"

    try:
        run_investigation(paper_id, id_type, output_dir=output_dir)
    except KeyboardInterrupt:
        print("\n[警告] 用户中断调查。")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Investigation failed with error: {e}")
        print(f"\n[错误] 调查失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
