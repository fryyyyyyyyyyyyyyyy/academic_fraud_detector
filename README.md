# Academic Fraud Detection System — 学术造假检测多 Agent 系统

基于 **CrewAI** 框架构建的层级式多 Agent 学术造假检测系统。通过 7 个专业 Agent 并行调查论文的 6 个造假维度，最终由法务证据整合师生成结构化的中文 Markdown 调查报告。

## 快速开始

```bash
# 安装依赖
.venv/Scripts/pip install -e ".[dev]"

# 本地 PDF 检测（最常用，无需 API）
.venv/Scripts/python -m academic_fraud_detector.main --pdf paper2test/xxx.pdf

# 通过 DOI 检测（需要 API）
.venv/Scripts/python -m academic_fraud_detector.main 10.1038/nature12345 doi

# 静默模式 + 自定义输出目录
.venv/Scripts/python -m academic_fraud_detector.main --pdf paper2test/xxx.pdf --output-dir ./my_reports/ --quiet
```

**输出产物：**

| 格式 | 说明 |
|------|------|
| `.md` | 中文 Markdown 报告（主格式，可直接阅读） |
| `.html` | 带样式的 HTML 报告（自动在浏览器中打开） |
| `.json` | 原始输出备份 |

---

## 整体架构

```
                    ┌─────────────────────────────────┐
                    │  Lead Investigator (Manager)     │
                    │  学术诚信首席调查员               │
                    │  强模型: Claude Sonnet / GPT-4o   │
                    └──────────────┬──────────────────┘
                                   │ 动态委派 + 评估发现 + 要求深入
            ┌──────────────────────┼──────────────────────┐
            │                      │                      │
    ┌───────┴───────┐      ┌──────┴──────┐      ┌───────┴───────┐
    │  Phase 1: 获取  │      │ Phase 2: 并行  │      │ Phase 3: 汇总  │
    │  acquire_paper  │      │  5-6 个维度    │      │  synthesize    │
    └───────────────┘      └──────────────┘      └───────────────┘
```

**关键架构决策：**

- **Process 类型**: `Process.hierarchical` — Manager 动态调度 Specialist，发现可疑时要求深入调查
- **两种运行模式**: 完整模式（6 维度）和本地 PDF 模式（3 维度，离线运行）
- **模型分层**: Manager + Synthesizer 用强模型 (`MANAGER_MODEL`)，Specialist 用轻量模型 (`AGENT_MODEL`)
- **LLM 提供商**: 通过 `OPENAI_API_BASE` 支持 DeepSeek 等兼容 OpenAI API 的服务
- **中文输出**: 所有 agent backstory 和 task 描述要求使用中文，最终报告为 Markdown 格式

---

## Agent 清单

### Manager 层

| Agent | 模型 | max_iter | 说明 |
|-------|------|:--------:|------|
| **Lead Investigator** | 强模型 | 20 | 学术诚信首席调查员，20 年科研诚信调查经验，主导 500+ 起学术不端调查。协调全局、委派任务、评估发现、做最终判定 |

### Specialist 层（并行执行）

| Agent | 权重（完整/本地） | max_iter | 核心职责 | 关键工具 |
|-------|:------------:|:--------:|---------|---------|
| **Image Forensics Analyst** | 30% / 30% | 15 | 数字图像取证，6 工具防捏造管线 | ELA、克隆检测、AI 检测、pHash 跨图比对、SIFT+RANSAC、背景一致性 |
| **Data Integrity Auditor** | 30% / **55%** | 25 | 统计数据造假检测，强制执行顺序 | 本福特、p 值分布、GRIM、异常精度、跨图数据比对、柱状图提取 |
| **Citation Network Investigator** | 15% / — | 15 | 引用圈、自引、编造引用检测 | NetworkX 图分析、自引率、引用主张验证 |
| **Peer Review Inspector** | 10% / — | 15 | 审稿人资质、模板复用、审稿时间线 | 审稿文本分析、审稿人资质校验、模板检测 |
| **Methodology Consistency Reviewer** | 10% / 15% | 15 | 试剂货号真伪、伦理审批号格式、实验时间线矛盾 | 试剂校验、伦理审批、时间线、Methods 自洽性 |
| **Productivity Anomaly Analyst** | 5% / — | 15 | 论文工厂、发表频率、Salami Slicing | 发表频率、Salami Slicing、Methods 雷同检测 |
| ~~**Plagiarism Detective**~~ | ~~25%~~ | — | ❌ **已禁用**（YAML 和代码保留，crew 中不注册） | 语义相似度、词法复制检测 |

### Synthesizer 层

| Agent | 模型 | max_iter | 说明 |
|-------|------|:--------:|------|
| **Evidence Synthesizer** | 强模型 | 10 | 法务证据整合师 & 调查报告撰写人。跨维度交叉验证 → 加权评分 → 中文 Markdown 报告 |

---

## Task 流水线

```
Phase 1 — 论文获取
┌─────────────────────────────────────────────────────────────────┐
│ acquire_target_paper                                             │
│ agent: plagiarism_detective（复用做加载，非抄袭检测）              │
│ context: []                                                      │
│ 功能：获取元数据 / 提取全文 / 提取图表 / 提取参考文献               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
Phase 2 — 并行专项调查（所有 task 的 context: [acquire_target_paper]）
┌─────────────────────────────────────────────────────────────────┐
│ ┌─ image_forensics_investigation     6 工具取证管线               │
│ ├─ data_integrity_investigation      预比对→统计检验→异常精度      │
│ ├─ citation_network_investigation    引用图→自引率→引用主张       │
│ ├─ peer_review_investigation         审稿文本→模板→审稿人资质     │
│ ├─ methodology_audit                 试剂→伦理→时间线→自洽性     │
│ └─ productivity_anomaly_investigation 发表频率→Salami→雷同检测    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
Phase 3 — 汇总与报告（context: 包含 Phase 1 + 全部 Phase 2 结果）
┌─────────────────────────────────────────────────────────────────┐
│ synthesize_findings（完整模式，6 维度）                            │
│ local_synthesize_findings（本地模式，3 维度）                     │
│ agent: evidence_synthesizer                                      │
│ 功能：交叉验证 → 加权评分 → 预警等级 → 建议 → Markdown 报告       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 加权评分方法

| 维度 | 完整模式权重 | 本地模式权重 | 评分依据 |
|------|:----------:|:----------:|---------|
| 图像取证 | 30% | 30% | 工具检测结果（必须有工具输出支撑，纯 LLM 视觉判定无效） |
| 数据完整性 | 30% | **55%** | 统计检验结果客观可验证，误报率低 |
| ~~抄袭检测~~ | ~~25%~~ | — | 已禁用 |
| 引用操纵 | 15% | — | 引用图分析 + 引用主张验证 |
| 同行评审 | 10% | — | 审稿文本 + 审稿人资质 |
| 方法论一致性 | 10% | 15% | 试剂校验 + 伦理审批 + 时间线 + 自洽性 |
| 产出异常 | 5% | — | 发表频率 + Salami Slicing |

> **风险等级映射**：**0-15** 无风险 → **16-35** 低风险 → **36-55** 中风险 → **56-75** 高风险 → **76-100** 严重风险

### 关键评分原则

- **区分"检测到造假证据"和"数据不足以评估"**：p 值仅以 `*P<0.05` 阈值报告不视为 p-hacking，评低风险（20-40）
- **图像取证发现必须有工具输出引用**：无工具支撑的"视觉判断"一律无效，该维度最多评 15 分
- **跨图数据比对发现精确匹配**：直接评 CRITICAL（90-100），数学上不可能

---

## 工具清单

### 图像取证工具（6 个）

| 工具 | 算法 | 检测目标 |
|------|------|---------|
| `error_level_analysis` | ELA 误差水平分析 | JPEG 压缩差异 → 编辑/拼接痕迹 |
| `clone_detection` | 滑动窗口 + 特征匹配 | 同图内复制粘贴（Western blot 条带复用） |
| `ai_image_detection` | 噪声分析 + 频域分析 | AI 生成/合成图像 |
| `background_consistency_check` | 背景噪声/亮度/纹理比较 | 不同凝胶泳道拼贴到同一张图 |
| `cross_image_duplicate_check` | DCT 感知哈希 + 汉明距离 | 整图旋转/翻转后跨图复用 |
| `feature_based_duplicate_check` | SIFT + FLANN + RANSAC | 面板级局部复用（检出 Fig 1D 在 Fig 4A 中出现） |

### 数据完整性工具（8 个）

| 工具 | 算法 | 检测目标 |
|------|------|---------|
| `bar_chart_extract_values` | 图像处理 + OCR | 从柱状图提取数值数据 |
| `chart_ocr_extract` | pytesseract → easyocr 回退 | 图表文本数值提取 |
| `cross_figure_data_compare` | 共享组精确匹配 | 不同图中"独立实验"有完全相同的数据 |
| `benford_law_test` | 卡方检验 + MAD | 首位数字分布是否符合自然规律 |
| `p_value_distribution_test` | 卡尺比率 + 均匀性检验 | p 值是否恰聚集在 0.05 以下 |
| `grim_test` | GRIM 一致性 | (均值, N) 配对在数学上是否可能 |
| `anomalous_precision_detection` | CV 分析 + 末位偏好 + 近等差 + 高频重复 | 编造数据的"太完美"痕迹 |
| `statistical_consistency_test` | 报告值 vs 计算值 | 论文中报告的值与从数据反算的值是否一致 |

### 引用分析工具（5 个）

| 工具 | 算法 | 检测目标 |
|------|------|---------|
| `citation_graph_analyzer` | NetworkX (SCC, PageRank, 社区检测) | 引用圈、互惠引用对、异常密度 |
| `self_citation_analyzer` | Semantic Scholar API | 作者自引率（>25% 标记） |
| `citation_claim_verifier` | LLM 推理 + 联网搜索 | 引用主张是否真实（防止编造引用） |
| `academic_web_search` | 学术搜索引擎 | 联网验证试剂/引用/论文存在性 |
| `citation_existence_check` | DOI/标题搜索 | 被引论文是否真实存在 |

### 同行评审工具（3 个）

| 工具 | 检测目标 |
|------|---------|
| `review_text_analyzer` | 审稿长度（<100 words 可疑）、技术特异性（<2 处引用可疑）、通用语占比 |
| `review_template_detector` | 句子级比对（SequenceMatcher >0.80），跨审稿模板复用 |
| `reviewer_credential_checker` | 学术发表记录、邮箱域名（免费邮箱 vs 机构邮箱）、领域相关性 |

### 方法论审查工具（4 个）

| 工具 | 检测目标 |
|------|---------|
| `reagent_verification` | 试剂货号 (Cat#) 格式校验 + LLM 知识库交叉验证 |
| `ethics_approval_check` | 伦理审批号格式校验（中国/美国/欧盟） |
| `experimental_timeline_check` | 实验周期 vs 投稿日期的时间冲突 |
| `method_internal_consistency` | n 值一致性、组数一致性、条件描述矛盾、统计方法矛盾 |

### 产出分析工具（3 个）

| 工具 | 检测目标 |
|------|---------|
| `publication_frequency_check` | 年均论文数（>12 篇一作/通讯 = 异常），学科合理范围 |
| `salami_slicing_check` | 样本量相似 + 实验设计雷同 + 对照组数据相同 |
| `text_similarity_check` | 同组多论文 Methods 段落的语义相似度（>0.90 可疑） |

---

## 关键管线详解

### 图像取证管线

```
PDF 加载
  ↓
① 提取嵌入图片 (PyMuPDF)
  ↓
② 拆分复合图 → 独立面板 (投影轮廓分析)
  ↓
③ 面板级 SIFT+RANSAC 比对  ← 检测局部复用（min_inliers=8~15）
  ↓
④ 整图级 pHash 比对        ← 检测旋转/翻转复用（汉明距离 ≤ 阈值）
  ↓
⑤ 单图取证: ELA → 克隆检测 → AI检测 → 背景一致性
```

> **设计意图**: 步骤 ③ 是 P0 修复核心 — 之前只做整图 pHash，遗漏了面板级局部复用。

### 数据完整性预比对管线（启动前自动执行）

```
PDF 加载
  ↓
① bar_chart_extract_values — 从每个柱状图提取数值
  ↓
② cross_figure_data_compare — 跨图比对所有数据集
  ↓
③ 检测"共享组精确匹配" — 数学上不可能（两批独立实验产生完全相同数据的概率 ≈ 0）
  ↓
④ 结果注入 data_integrity_investigation task 的 {cross_figure_precheck}
```

### 图像取证防捏造规则

Image Forensics Analyst 的 backstory 中定义了 6 条铁律：

1. **工具输出是唯一证据来源** — 严禁仅凭 LLM 视觉能力判断
2. **阴性结果必须如实报告** — 全部工具阴性时写"经 N 项工具检测，未发现篡改证据"
3. **每条发现必须引用工具输出** — 格式："工具 X 返回：...。判定：..."
4. **跨图重复必须有精确数值** — Hamming distance + RANSAC inliers + 变换类型
5. **区分"检测到异常"和"无法检测"** — 工具报错 ≠ 检测通过
6. **调用顺序** — 先单图取证（ELA→克隆→AI→背景一致性），再跨图（pHash→SIFT）

---

## 两种运行模式

| | **完整模式 (full)** | **本地 PDF 模式 (local_only)** |
|---|---|---|
| **触发方式** | `python -m ... 10.1038/xxx doi` | `python -m ... --pdf paper.pdf` |
| **需要 API** | ✅ arXiv / CrossRef / Semantic Scholar | ❌ 离线运行 |
| **调查维度** | 6 个 | 3 个（图像 + 数据 + 方法论） |
| **Worker Agents** | 6 个 | 4 个 |
| **Tasks** | 7 个 | 5 个 |
| **报告模板** | `synthesize_findings` | `local_synthesize_findings` |

### 本地模式未执行的维度

| 维度 | 原因 |
|------|------|
| 引用网络分析 | 需要 Semantic Scholar 等引用数据库 API |
| 同行评审审查 | 需要公开的审稿意见元数据（多数期刊不公开） |
| 产出异常分析 | 需要外部文献数据库（PubMed/CrossRef）API |

---

## 报告输出结构

```markdown
# 学术诚信调查报告

## 被调查论文（元数据表格）

## 执行摘要（3-5 条要点）

## 总体风险评估（加权评分表 + 风险等级说明）

## 调查发现
### 一、图像取证分析
### 二、数据完整性分析
    ├── 🔴 跨图数据比对结果（强制执行，不可为空）
    ├── 本福特定律检验
    ├── p 值分布分析
    ├── GRIM 检验
    ├── 异常精度检测（CV + 末位偏好 + 近等差 + 高频重复）
    └── 统计一致性验证
### 三、引用网络分析（完整模式）
### 四、同行评审审查（完整模式）
### 五、方法论一致性审查
    ├── 试剂/设备校验
    ├── 伦理审批号校验
    ├── 实验时间线
    ├── Methods 内部自洽性
    └── 实验设计逻辑检查
### 六、产出异常分析（完整模式）

## 未调查维度（本地模式专属）

## 跨维度关联分析

## 建议（带优先级 + 紧迫度 + 理由）

## 局限性声明

## 证据清单（EVID-001, EVID-002 ...）
```

---

## 项目结构

```
src/academic_fraud_detector/
├── main.py                     # CLI 入口 + run_investigation() + Markdown→HTML 转换
├── crew.py                     # @CrewBase 组装: 初始化工具 + 创建 agents + 组装 crew
├── config/
│   ├── agents.yaml             # 8 个 Agent 的 role/goal/backstory（中文 persona）
│   └── tasks.yaml              # 9 个 Task 的详细调查流程和 expected_output 模板
├── tools/                      # CrewAI BaseTool 子类
│   ├── paper_fetching.py       # arXiv/CrossRef/S2 搜索 + 本地 PDF 加载
│   ├── text_similarity.py      # 语义相似度 (sentence-transformers) + 词法复制检测
│   ├── image_forensics.py      # ELA/克隆/AI检测/跨图pHash/SIFT+RANSAC/背景一致性
│   ├── statistical_analysis.py # Benford/p值/GRIM/异常精度/统计一致性/跨图数据比对
│   ├── citation_analysis.py    # NetworkX 引用图 (SCC, PageRank) + 自引率
│   ├── peer_review_analysis.py # 审稿文本分析/审稿人资质/模板检测
│   ├── methodology_audit.py    # 试剂校验/伦理审批/时间线/Methods 自洽
│   ├── productivity_analysis.py# 发表频率/Salami Slicing
│   └── web_search.py           # 学术联网搜索 + 引用存在性检查
├── utils/
│   ├── chart_ocr.py            # pytesseract → easyocr 回退，提取图中数值
│   ├── figure_splitter.py      # 投影轮廓分析拆分复合图 → 独立面板
│   ├── text_extraction.py      # PyMuPDF 文本/图片/表格提取
│   ├── table_extraction.py     # camelot-py 表格数据提取
│   ├── image_downloader.py     # URL 图片下载与本地缓存
│   ├── api_client.py           # 限流 API 客户端
│   └── cross_figure_pipeline.py# 启动前自动预比对管线
└── models/
    ├── investigation_report.py # 最终报告 Pydantic Schema
    ├── paper.py                # 论文元数据模型
    └── evidence.py             # 证据链追踪模型
```

---

## 环境配置

在项目根目录创建 `.env` 文件：

```bash
# 必填
OPENAI_API_KEY=sk-xxxxx
# 或
ANTHROPIC_API_KEY=sk-ant-xxxxx

# 选填：自定义 API 端点（DeepSeek 等 OpenAI 兼容服务）
OPENAI_API_BASE=https://api.deepseek.com

# 选填：模型选择
MANAGER_MODEL=deepseek-chat       # Manager + Synthesizer 用
AGENT_MODEL=deepseek-chat         # Specialist 用
```

---

## 运行测试

```bash
# 运行所有测试
.venv/Scripts/python -m pytest tests/ -v

# 运行单个测试文件
.venv/Scripts/python -m pytest tests/test_tools/test_statistical_analysis.py -v

# 代码检查
.venv/Scripts/python -m ruff check src/
```
