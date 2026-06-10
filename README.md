# Academic Fraud Detection System — 学术造假检测多 Agent 系统

基于 **CrewAI** 的本地学术数据造假审计系统。当前本地检测流程已改为：**论文 PDF + XLSX 原始数据**，通过确定性统计检验和数据比对生成证据，再由 Agent 汇总为中文 Markdown/HTML 报告。

> 当前本地模式 **不做图片处理分析**：不提取图片、不拆分 panels、不做图像取证、不做图像 OCR/柱状图 OCR。数据造假判断必须来自代码层确定性证据，不能让大模型凭主观臆断。

---

## 快速开始

### 1. 安装依赖

所有 Python 命令都应在项目 `.venv` 中运行：

```bash
.venv/Scripts/pip install -e ".[dev]"
```

### 2. 本地案例目录检测（推荐）

案例目录结构应类似：

```text
案例目录/
├── 论文主体.pdf
├── Source Data Fig.1.xlsx
├── Source Data Fig.2.xlsx
└── ...
```

启动检测：

```bash
.venv/Scripts/python -m academic_fraud_detector.main --case "path/to/case_folder"
```

等价 positional 写法：

```bash
.venv/Scripts/python -m academic_fraud_detector.main "path/to/case_folder" local_case
```

### 3. 自定义输出目录 / 静默模式

```bash
.venv/Scripts/python -m academic_fraud_detector.main --case "path/to/case_folder" --output-dir ./my_reports/ --quiet
```

### 4. PDF-only 兼容模式

仍保留单 PDF 输入，但该模式没有 XLSX 原始数据，不能完成核心原始数据审计；报告会如实说明数据限制。

```bash
.venv/Scripts/python -m academic_fraud_detector.main --pdf paper2test/xxx.pdf
```

### 5. DOI / 标题 / URL 模式（需要 API）

```bash
.venv/Scripts/python -m academic_fraud_detector.main 10.1038/nature12345 doi
```

---

## 输出产物

默认输出到 `./reports/`：

| 格式 | 说明 |
|------|------|
| `.md` | 中文 Markdown 报告（主格式，可直接阅读） |
| `.html` | 带样式的 HTML 报告（自动在浏览器中打开） |
| `.json` | 原始 Markdown 输出备份 |

---

## 整体架构

```text
                    ┌─────────────────────────────────┐
                    │ Lead Investigator (Manager)     │
                    │ 学术诚信首席调查员               │
                    │ 强模型：MANAGER_MODEL            │
                    └──────────────┬──────────────────┘
                                   │ 层级式调度 + 证据约束 + 报告质量控制
            ┌──────────────────────┼──────────────────────┐
            │                      │                      │
            ▼                      ▼                      ▼
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│ Paper Acquisition    │  │ Raw Data Precheck     │  │ Methodology Audit    │
│ PDF 文本/表格提取     │  │ XLSX 确定性统计预检    │  │ 方法学一致性审查       │
└──────────┬───────────┘  └──────────┬───────────┘  └──────────┬───────────┘
           │                         │                         │
           └─────────────────────────┼─────────────────────────┘
                                     ▼
                    ┌─────────────────────────────────┐
                    │ Evidence Synthesizer            │
                    │ 只汇总 evidence_id 支撑的证据    │
                    │ 生成中文 Markdown / HTML 报告    │
                    └─────────────────────────────────┘
```

**关键架构决策：**

- **Process 类型**：`Process.hierarchical`，由 Manager 统一调度和汇总。
- **本地案例模式**：`local_case`，输入为“1 个论文 PDF + 若干 XLSX 原始数据”。
- **确定性证据优先**：数据造假结论来自 `raw_data_precheck_json` / `deterministic_evidence_json`，大模型只负责解释和汇总。
- **本地模式不做图片处理**：不提取图片、不拆分 panels、不做图像取证、不做图像 OCR/柱状图 OCR。
- **模型分层**：Manager + Synthesizer 使用 `MANAGER_MODEL`，Specialist 使用 `AGENT_MODEL`。
- **中文输出**：Agent 任务与最终报告均使用中文，最终报告为 Markdown 主格式。

---

## 当前本地检测流程

```text
用户输入 --case 案例目录
        │
        ▼
发现案例材料
- 选择论文 PDF
- 收集 Source Data *.xlsx/*.xlsm/*.xltx/*.xltm
        │
        ▼
PDF 解析（不提取图片）
- full_text
- tables
- p-values / mean±SD / numeric values
        │
        ▼
XLSX 原始数据标准化
- workbook / sheet / row / column / range
- raw_values / numeric values
- decimal places / last digits / decimal suffixes
        │
        ▼
确定性 raw-data precheck
- 近似等差数列
- 重复/相似数据
- 打乱顺序后的重复/相似
- 小数部分/小数后缀复用
- 小数最后一位频率异常/缺失异常
- 论文报告统计量与原始数据候选对齐
        │
        ▼
CrewAI 汇总
- Data Integrity Auditor：只解释确定性证据
- Methodology Reviewer：方法学一致性审查
- Evidence Synthesizer：生成中文报告
```

---

## 本地 raw-data 检测项目

### 1. 近似等差数列

检测某列/某行数据是否近乎线性递增或递减。

输出指标包括：

- `cv_of_diffs`：相邻差值变异系数；越低越接近等差。
- `r_squared`：线性拟合优度。
- `order_mode`：原始顺序或排序后检测。
- `evidence_id`：可追溯证据编号。

### 2. 重复或相似数据

检测不同组、不同列、不同 sheet 或不同 workbook 之间是否存在：

- 同序完全重复；
- 同序近似重复；
- 打乱顺序后的多重集合重复；
- 子集重复；
- 线性缩放/平移后的高度相似。

### 3. 小数部分 / 小数后缀异常

检测：

- 某个小数后缀在同一数据集中异常高频；
- 不同数据集之间共享高度相似的小数后缀集合；
- 整数部分不同但小数后缀高度复用。

### 4. 小数最后一位异常

对小数最后一位 `0-9` 做概率检验：

- 单个数字高频：`P(X >= k), X ~ Binomial(n, 0.1)`；
- 指定数字缺失：`P(X = 0) = 0.9^n`；
- 任一数字缺失：occupancy / inclusion-exclusion 概率；
- 整体分布异常：chi-square goodness-of-fit；
- 多重检验修正：BH-FDR，输出 `q_value`。

例如：若 70 个数据中 26 个小数最后一位是 `5`，系统会计算 `P(X >= 26), X ~ Binomial(70, 0.1)`，并把概率写入证据 JSON 和最终报告。

### 5. 论文内容与原始数据对齐

系统会从 PDF 文本/表格提取 `mean ± SD`、p 值和数值表格，并与 XLSX 数据集的描述统计做候选对齐。没有明确标签映射时，未匹配项不会被直接当作造假证据。

---

## 防止大模型主观臆断

最终报告必须遵守：

1. 每条数据造假风险都必须引用 `evidence_id`。
2. 每条证据必须包含文件、sheet、单元格范围、统计方法、p-value/q-value/effect size 等客观数据。
3. 没有进入 `allowed_claims_json` 的内容不能写成造假发现。
4. 如果没有 high/critical 证据，报告必须明确写“未发现高置信度确定性数据造假证据”。
5. 统计异常不等于单独证明造假，必须列出可能的非造假解释和人工复核建议。
6. 当前本地流程不做图像处理，不得声称执行了图像取证或图像 OCR。

---

## Agent 清单

### Manager 层

| Agent | 模型 | max_iter | 说明 |
|-------|------|:--------:|------|
| **Lead Investigator** | `MANAGER_MODEL` | 20 | 学术诚信首席调查员，负责层级式调度、检查证据约束、控制最终报告质量。 |

### Specialist 层

| Agent | 本地模式 | 完整/API 模式 | max_iter | 核心职责 | 关键输入/工具 |
|-------|:--------:|:------------:|:--------:|----------|---------------|
| **Raw Data Integrity Auditor** | ✅ | ✅ | 25 | 原始数据完整性审计，只解释确定性 evidence | `raw_data_precheck_json`、Benford、p-value、GRIM、统计一致性 |
| **Methodology Consistency Reviewer** | ✅ | ✅ | 15 | 方法学一致性审查 | 试剂校验、伦理审批、实验时间线、Methods 自洽性 |
| **Citation Network Investigator** | — | ✅ | 15 | 引用网络、自引和引用主张验证 | Semantic Scholar、NetworkX、引用主张验证 |
| **Peer Review Inspector** | — | ✅ | 15 | 审稿文本、模板复用、审稿人资质审查 | 审稿文本分析、模板检测、资质检查 |
| **Productivity Anomaly Analyst** | — | ✅ | 15 | 发表频率、Salami Slicing、论文工厂线索 | 发表频率、文本相似度、外部文献数据库 |
| **Image Forensics Analyst** | — | 可保留/不推荐 | 15 | 历史图像取证能力；当前本地模式不运行 | ELA、克隆检测、pHash、SIFT 等旧工具 |
| ~~**Plagiarism Detective**~~ | — | — | — | 已禁用，YAML/代码保留用于加载或未来复用 | 语义相似度、词法复制检测 |

### Synthesizer 层

| Agent | 模型 | max_iter | 说明 |
|-------|------|:--------:|------|
| **Evidence Synthesizer** | `MANAGER_MODEL` | 10 | 只汇总带 `evidence_id` 的证据，生成中文 Markdown / HTML 报告。 |

---

## Task 流水线

### 本地案例模式：`local_case`

```text
Phase 0 — 启动前确定性预检
┌─────────────────────────────────────────────────────────────────┐
│ discover_case_folder                                             │
│ load_raw_data_files                                              │
│ run_raw_data_precheck                                            │
│ 输出：case_manifest_json / raw_data_precheck_json / evidence JSON │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
Phase 1 — 论文获取
┌─────────────────────────────────────────────────────────────────┐
│ acquire_target_paper                                             │
│ 功能：读取本地 PDF 文本、表格、p 值、mean±SD 等                    │
│ 注意：extract_images=False，不提取图片                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
Phase 2 — 专项审计
┌─────────────────────────────────────────────────────────────────┐
│ data_integrity_investigation                                     │
│ methodology_audit                                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
Phase 3 — 汇总报告
┌─────────────────────────────────────────────────────────────────┐
│ local_synthesize_findings                                        │
│ 只允许报告 deterministic_evidence_json 中的 evidence_id 证据       │
└─────────────────────────────────────────────────────────────────┘
```

### 完整/API 模式

完整/API 模式仍保留 DOI、标题、URL 等入口，用于需要外部数据库的调查；但当前 README 的推荐路径是本地 `--case` 原始数据审计。

---

## 评分方法

| 维度 | 本地模式权重 | 评分依据 |
|------|:----------:|---------|
| 数据完整性 | **85%** | XLSX 原始数据确定性证据：近等差、重复/相似、末位异常、后缀复用、统计不一致等 |
| 方法论一致性 | 15% | Methods 自洽性、伦理审批、试剂/设备、实验时间线 |
| 图像取证 | — | 当前本地模式不执行 |
| 引用操纵 | — | 需外部引用数据库 API |
| 同行评审 | — | 需公开审稿元数据 |
| 产出异常 | — | 需外部文献数据库 API |

> **风险等级映射**：0-15 无风险｜16-35 低风险｜36-55 中风险｜56-75 高风险｜76-100 严重风险。

关键原则：

- 评分必须来自 `confidence_summary_json` 和带 `evidence_id` 的确定性证据。
- 多条同源证据不应机械重复加权。
- 统计异常不等于正式造假结论，报告必须给出可能的非造假解释。

---

## 工具清单

### 本地 raw-data 工具

| 工具/模块 | 作用 |
|----------|------|
| `case_folder.py` | 发现案例目录中的 PDF 和 XLSX 原始数据文件 |
| `raw_data_loader.py` | 标准化 XLSX workbook/sheet/行/列/range，保留小数位、末位、后缀和来源定位 |
| `raw_data_precheck.py` | 运行近似等差、重复/相似、小数后缀、末位数字概率等确定性预检 |
| `paper_fetching.py` | 本地 PDF 文本/表格加载，local_case 下关闭图片提取 |
| `table_extraction.py` | PDF 表格、p 值、mean±SD、数值提取 |
| `statistical_analysis.py` | Benford、p-value、GRIM、异常精度、统计一致性等统计工具 |

### 外部/API 工具（非本地核心路径）

| 工具类别 | 作用 |
|----------|------|
| Citation tools | 引用网络、自引、引用主张验证 |
| Peer review tools | 审稿文本、模板复用、审稿人资质检查 |
| Productivity tools | 发表频率、Salami Slicing、Methods 雷同 |
| Image forensics tools | 历史图像取证能力；当前本地模式不调用 |

---

## 报告输出结构

```markdown
# 学术诚信调查报告

## 一、输入材料
- PDF
- XLSX 原始数据文件
- Sheet 数量 / dataset 数量 / 可分析数值数量

## 二、执行摘要
- 总体风险等级
- 最高置信度证据
- 主要局限性

## 三、总体风险评估
- 风险评分
- 置信度
- critical/high/medium/low 证据数量

## 四、确定性数据证据
- 每条证据必须包含 evidence_id
- 文件 / sheet / range
- 检测方法和统计量
- p_value / q_value / effect size

## 五、专项分析
### 5.1 近似等差数列
### 5.2 重复或相似数据
### 5.3 小数部分/小数后缀异常
### 5.4 小数最后一位频率异常
### 5.5 论文内容与原始数据对齐

## 六、方法论一致性审查

## 七、未调查维度

## 八、局限性与人工复核建议

## 九、证据清单
```

---

## 本地模式未执行的维度

| 维度 | 原因 |
|------|------|
| 图像取证 | 用户要求当前不做图片处理分析 |
| 引用网络分析 | 需要 Semantic Scholar 等外部引用数据库 API |
| 同行评审审查 | 需要公开审稿元数据 |
| 产出异常分析 | 需要外部文献数据库 API |

---

## 关键文件

```text
src/academic_fraud_detector/
├── main.py                      # CLI 入口，支持 --case / local_case
├── crew.py                      # CrewAI 组装，本地模式注入 raw-data evidence
├── config/
│   ├── agents.yaml              # Agent 角色设定
│   └── tasks.yaml               # 任务提示与报告模板
├── tools/
│   ├── paper_fetching.py        # 本地 PDF 加载，local_case 下 extract_images=False
│   └── statistical_analysis.py  # Benford、p-value、GRIM、异常精度等统计工具
├── utils/
│   ├── case_folder.py           # 本地案例目录发现：PDF + XLSX
│   ├── raw_data_loader.py       # XLSX 原始数据标准化
│   ├── raw_data_precheck.py     # 确定性原始数据预检与 evidence 输出
│   ├── text_extraction.py       # PDF 文本提取
│   └── table_extraction.py      # PDF 表格/统计值提取
└── models/
    ├── investigation_report.py
    ├── paper.py
    └── evidence.py
```

---

## 环境配置

在项目根目录创建 `.env` 文件：

```bash
# OpenAI 兼容服务，例如 DeepSeek
OPENAI_API_KEY=sk-xxxxx
OPENAI_API_BASE=https://api.deepseek.com

# 模型选择
MANAGER_MODEL=deepseek-chat
AGENT_MODEL=deepseek-chat

# 或使用 Anthropic
# ANTHROPIC_API_KEY=sk-ant-xxxxx
```

---

## 运行测试

```bash
# 本次 raw-data 改造相关测试
.venv/Scripts/python -m pytest tests/test_case_folder.py tests/test_raw_data_loader.py tests/test_raw_data_precheck.py tests/test_local_pdf_kickoff.py -v

# 全部测试
.venv/Scripts/python -m pytest tests/ -v

# 代码检查
.venv/Scripts/python -m ruff check src/
```

---

## 局限性

- `.xls` 旧 Excel 格式暂不作为本地案例原始数据输入；请转换为 `.xlsx`。
- 小数末位检验依赖“连续测量值末位近似均匀”的假设；若仪器或记录规则限制末位，需人工解释。
- Excel 存储值与显示格式可能不同；系统会尽量利用 `number_format` 重建显示小数位。
- 自动检测结果只能作为学术诚信初筛证据，正式结论仍需领域专家结合实验设计、原始记录和实验上下文复核。
