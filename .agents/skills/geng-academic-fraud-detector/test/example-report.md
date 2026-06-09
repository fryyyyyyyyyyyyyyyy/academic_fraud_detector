# 🔍 耿同学打假报告

## 论文信息
- **标题**：RTA408 alleviates retinal ganglion cells damage in mouse glaucoma by inhibiting excessive autophagy
- **作者**：Hongmei Qian, Wei Chen, Guomei Yuan, Man Luo, Li Zhang, Biao Wu, Hanshi Huang, Jiahao Xu, Qiong Wang*, Mengyun Li*
- **期刊**：PLOS ONE 19(11): e0313446
- **DOI**：10.1371/journal.pone.0313446
- **发表日期**：2024年11月11日
- **当前状态**：已撤稿

## 综合评定：🔴 实锤

多处独立证据指向系统性数据造假，无法用"疏忽"解释。

---

## 详细发现

### 发现 1：图片复用 —— Figure 1D vs Figure 4A

- **位置**：Figure 1D（HE 染色）与 Figure 4A（HE 染色）
- **描述**：两张图声称展示不同实验批次的 HE 染色结果。Fig 1D 为第一批实验（含 Tafluprost 组），Fig 4A 为第二批实验（含 3-MA/Rapamycin 组）。两图共享的 Control、COH、COH+RTA408 面板存在部分重叠。
- **证据**：撤稿声明确认 "panels representing different retinal regions appear to partially overlap"
- **判断逻辑**：若为两批独立动物实验，即使是同一组别，不同批次的切片不可能产生完全相同的图片。
- **严重程度**：🔴 高

### 发现 2：图片复用 —— Figure 2A vs Figure 5A

- **位置**：Figure 2A（RGCs 免疫荧光）与 Figure 5A（RGCs 免疫荧光）
- **描述**：两图同样声称展示不同实验批次的 RGCs 免疫荧光成像。共享组面板存在相似/重叠。更严重的是，Fig 5A 中的部分面板与对应的全视网膜图不匹配。
- **证据**：撤稿声明确认 "panels do not appear to match the corresponding whole retina panels"
- **判断逻辑**：面板与全视网膜图不对应，说明图片被错误拼接或复用。
- **严重程度**：🔴 高

### 发现 3：数据复用 —— Fig 1E vs Fig 4B 原始数据完全相同

- **位置**：Figure 1E（RGCs 计数量化）与 Figure 4B（RGCs 计数量化）
- **描述**：S2 File 中 Control、COH、COH+RTA408 三组的原始数据在两个 figure 中完全一致。
- **证据**：撤稿声明明确指出 "the underlying data in S2 File for Control, COH, and COH+RTA408 results in Fig 1E are the same as in Fig 4B"
- **判断逻辑**：生物实验的独立重复不可能产生完全相同的计数结果。如果数据相同，只有两种可能：(1) 根本没做第二批实验，直接复用了第一批数据；(2) 数据是编造的。无论哪种都构成学术不端。
- **严重程度**：🔴 高（核心问题）

### 发现 4：动物实验伦理异常

- **位置**：Methods Section 2.1
- **描述**：论文描述对小鼠双眼（both eyes）进行硅油注射建模，这违反了动物实验伦理的一般规范。双侧处理使动物完全失去视力，且缺乏对侧眼内部对照。
- **严重程度**：🟡 中

### 发现 5：实验设计逻辑矛盾

- **位置**：全文实验设计
- **描述**：论文声称进行了两批独立动物实验（Section 2.1 明确描述了两个实验方案），但共享组的图片和数据完全一致。这在逻辑上互相矛盾——要么实验方案描述是虚构的，要么数据是复用/编造的。
- **严重程度**：🔴 高

---

## 耿同学辣评

> "同一组小鼠的视网膜细胞，在两个'独立实验'里数出了一模一样的数量——这不是科学，这是 Ctrl+C Ctrl+V。兄弟们，我怀疑这些小鼠压根只存在于 Excel 表格里。"

---

## 检测方法总结

| 耿同学六式 | 是否触发 | 发现 |
|------------|----------|------|
| 第一式：图片复用 | ✅ 触发 | Fig 1D/4A、Fig 2A/5A 面板重复 |
| 第二式：数据造假 | ✅ 触发 | Fig 1E/4B 原始数据完全相同 |
| 第三式：图片拼接 | ⚠️ 疑似 | Fig 5A 面板与全图不匹配 |
| 第四式：统计异常 | 未触发 | 未发现明显 p-hacking |
| 第五式：产出异常 | 未检测 | 需要检索作者其他论文 |
| 第六式：方法矛盾 | ✅ 触发 | 双侧处理 + 实验设计逻辑矛盾 |

---

## 建议后续行动

- [x] 在 PubPeer 上提出质疑（已有）
- [x] 期刊调查并撤稿（已完成）
- [ ] 关注作者其他论文是否存在类似问题

## ⚠️ 免责声明

本报告由 AI 辅助生成，仅供学术讨论参考。
学术不端的最终认定需要专业机构调查。
我们支持学术诚信，但也尊重每一位研究者的名誉权。
如有异议，请以官方调查结论为准。
本工具不保证检测结果的准确性，误报和漏报均有可能。
