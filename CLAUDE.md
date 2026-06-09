# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 环境要求

所有 Python 命令必须在项目 `.venv` 中运行。Windows 上使用 `.venv/Scripts/python` 和 `.venv/Scripts/pip`（bash 格式：`.venv/Scripts/python`，不要用 `\`）。

```bash
# 安装依赖（项目根目录）
.venv/Scripts/pip install -e ".[dev]"

# 运行所有测试
.venv/Scripts/python -m pytest tests/ -v

# 运行单个测试文件
.venv/Scripts/python -m pytest tests/test_tools/test_statistical_analysis.py -v

# 代码检查
.venv/Scripts/python -m ruff check src/
```

## 运行检测

```bash
# 本地 PDF（最常用，不需要 API，仅用图片+数据检测）
.venv/Scripts/python -m academic_fraud_detector.main --pdf paper2test/xxx.pdf

# 等价写法
.venv/Scripts/python -m academic_fraud_detector.main paper2test/xxx.pdf local_pdf

# 通过 DOI（需要 API）
.venv/Scripts/python -m academic_fraud_detector.main 10.1038/nature12345 doi

# 静默模式 + 自定义输出目录
.venv/Scripts/python -m academic_fraud_detector.main --pdf paper2test/xxx.pdf --output-dir ./my_reports/ --quiet
```

输出产物：
- `.md` — 中文 Markdown 报告（主格式）
- `.html` — 带样式 HTML（自动在浏览器中打开）
- `.json` — 原始输出备份

## 架构概览

```
Lead Investigator (Manager, 层级式进程)
    │
    ├── Plagiarism Detective       ← 抄袭检测 (25%)
    ├── Image Forensics Analyst    ← 图像取证 (25%)
    ├── Data Integrity Auditor     ← 数据审计 (30%)
    ├── Citation Network Investigator ← 引用操纵 (10%)
    ├── Peer Review Inspector      ← 审稿欺诈 (10%)
    │   (以上5个并行执行)
    │
    └── Evidence Synthesizer       ← 汇总评分 + 生成中文报告
```

**关键架构决策：**
- **Process 类型**: `Process.hierarchical` — Manager 动态调度，发现可疑时要求深入调查
- **两种运行模式**: `local_only=True`（仅图像+数据，4 agents/4 tasks）和完整模式（7 agents/7 tasks）
- **模型分层**: Manager 用强模型 (`MANAGER_MODEL`)，Specialist 用轻量模型 (`AGENT_MODEL`)
- **LLM 提供商**: 通过 `OPENAI_API_BASE` 支持 DeepSeek 等兼容 OpenAI API 的服务
- **中文输出**: 所有 agent backstory 和 task 描述要求使用中文。最终报告为 Markdown 格式（非 JSON）

## 代码组织

```
src/academic_fraud_detector/
├── main.py              # CLI + run_investigation() API + report→HTML 转换
├── crew.py              # @CrewBase 组装：初始化工具、创建 agents、组装 crew
├── config/
│   ├── agents.yaml      # 7 个 Agent 的 role/goal/backstory（纯 YAML，不含 tools）
│   └── tasks.yaml       # 7 个 Task 的详细调查流程和 expected_output 模板
├── tools/               # CrewAI BaseTool 子类（每个工具一个类 + Pydantic Input Schema）
│   ├── paper_fetching.py      # arXiv/CrossRef/S2 搜索 + 本地 PDF 加载
│   ├── text_similarity.py     # 语义相似度 (sentence-transformers) + 词法复制检测
│   ├── image_forensics.py     # ELA, 克隆检测, AI图片检测, 跨图pHash, SIFT+RANSAC
│   ├── statistical_analysis.py # Benford, p-value caliper, GRIM, 异常精度, 统计一致性
│   ├── citation_analysis.py   # NetworkX 引用图 (SCC, PageRank) + 自引率
│   └── peer_review_analysis.py # 审稿文本分析, 审稿人资质, 模板检测
├── utils/
│   ├── chart_ocr.py           # pytesseract → easyocr 回退，提取图中数值
│   ├── figure_splitter.py     # 投影轮廓分析拆分复合图 → 独立面板（SIFT比对的前提）
│   ├── text_extraction.py     # PyMuPDF 文本/图片/表格提取
│   ├── table_extraction.py    # camelot-py 表格数据提取
│   ├── image_downloader.py    # URL 图片下载与本地缓存
│   └── api_client.py          # 限流 API 客户端
└── models/
    ├── investigation_report.py # 最终报告 Pydantic Schema
    ├── paper.py                # 论文元数据
    └── evidence.py             # 证据项 Schema
```

## 工具注册与装配

工具**不在 YAML 中配置**，而是在 `crew.py` 的 `@agent` 方法中通过 `tools=[...]` 列表注入。添加新工具的步骤：

1. 在 `tools/` 或 `utils/` 下写工具类（继承 `BaseTool`，定义 `name`/`description`/`args_schema`）
2. 在 `crew.py` 的 `__init__` 中实例化它
3. 在对应 `@agent` 方法中把实例加入 `tools` 列表
4. 如果需要在 task description 中引导 agent 调用，更新 `tasks.yaml`

`tools/__init__.py` 使用了**延迟导入**（guarded imports），允许在不安装完整 CrewAI 的情况下单独测试工具模块。

## 图像检测管线（关键路径）

PDF 加载后自动执行以下管线：
1. **提取嵌入图片** → `LocalPaperLoaderTool` (PyMuPDF)
2. **拆分复合图** → `figure_splitter.extract_all_panels_from_pdf()` 将 Figure 1A/1B/1C/1D 拆为独立面板
3. **面板级 SIFT 比对** → `FeatureBasedDuplicateTool` (SIFT + FLANN + RANSAC, `min_inliers=15`)
4. **整图级 pHash 比对** → `CrossImageDuplicateTool` (DCT 感知哈希, 汉明距离, 检测旋转/翻转复用)
5. **单图取证** → ELA + 克隆检测 + AI检测 + 背景一致性

**设计意图**: 步骤 3 是 P0 修复的核心 — 之前的系统只做整图 pHash，遗漏了面板级局部复用（如 Fig 1D 的部分区域在 Fig 4A 中重复出现）。

## 配置

- `.env` — API keys 和模型选择。必填：`OPENAI_API_KEY`（或 `ANTHROPIC_API_KEY`），选填：`OPENAI_API_BASE`（DeepSeek 等）、`MANAGER_MODEL`、`AGENT_MODEL`
- `agents.yaml` — Agent 角色定义（中文 persona，含 `max_iter` 和 `allow_delegation`）
- `tasks.yaml` — 调查流程的详细指引和 `expected_output` 模板（JSON schema 和 Markdown 模板）
- `.env.example` — 环境变量参考

## 待办事项

`todo.txt` 中记录了 13 项待办任务，分为 4 个阶段。背景：与纯 prompt 技能 `geng-academic-fraud-detector` 对比后，发现缺少"方法矛盾检测"、"产出异常分析"、"实验设计逻辑"、"引用主张验证"等 LLM 推理维度。
