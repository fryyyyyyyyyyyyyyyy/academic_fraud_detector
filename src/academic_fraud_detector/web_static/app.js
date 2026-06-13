const modeCopy = {
  doi: {
    label: "DOI",
    placeholder: "例如 10.1038/example",
    help: "输入论文 DOI，系统会使用现有调查流程处理。",
  },
  arxiv_id: {
    label: "arXiv ID",
    placeholder: "例如 2401.00001",
    help: "输入 arXiv ID，系统会拉取论文信息并生成调查报告。",
  },
  title: {
    label: "论文标题",
    placeholder: "输入完整或尽量准确的论文标题",
    help: "标题模式依赖外部检索结果，建议尽量提供完整标题。",
  },
  url: {
    label: "论文 URL",
    placeholder: "例如 https://example.org/paper.pdf",
    help: "输入论文页面或 PDF 地址，系统会按现有流程处理。",
  },
  local_pdf: {
    label: "本机 PDF 路径",
    placeholder: "例如 D:/papers/example.pdf",
    help: "路径位于运行 WebUI 的这台机器上。本地 PDF 模式不执行图像取证。",
  },
  local_case: {
    label: "本机案例目录路径",
    placeholder: "例如 D:/cases/example_case",
    help: "目录需包含至少一个 PDF 和至少一个支持的 Excel 原始数据文件。",
  },
};

const form = document.querySelector("#jobForm");
const identifierInput = document.querySelector("#paperIdentifier");
const outputDirInput = document.querySelector("#outputDir");
const identifierLabel = document.querySelector("#identifierLabel");
const identifierHelp = document.querySelector("#identifierHelp");
const submitButton = document.querySelector("#submitButton");
const resetButton = document.querySelector("#resetButton");
const statusPanel = document.querySelector("#statusPanel");
const statusTitle = document.querySelector("#statusTitle");
const statusBadge = document.querySelector("#statusBadge");
const statusMessage = document.querySelector("#statusMessage");
const statusMeta = document.querySelector("#statusMeta");
const loadingBar = document.querySelector("#loadingBar");
const resultPanel = document.querySelector("#resultPanel");
const resultTitle = document.querySelector("#resultTitle");
const riskSummary = document.querySelector("#riskSummary");
const fileLinks = document.querySelector("#fileLinks");
const markdownPreview = document.querySelector("#markdownPreview");

let pollTimer = null;

function selectedType() {
  return new FormData(form).get("identifier_type");
}

function updateModeCopy() {
  const copy = modeCopy[selectedType()];
  identifierLabel.textContent = copy.label;
  identifierInput.placeholder = copy.placeholder;
  identifierHelp.textContent = copy.help;
}

function stopPolling() {
  if (pollTimer) {
    window.clearInterval(pollTimer);
    pollTimer = null;
  }
}

function setButtonBusy(isBusy) {
  submitButton.disabled = isBusy;
  submitButton.textContent = isBusy ? "调查运行中" : "开始调查";
}

function setStatus(state, title, message, badge) {
  statusPanel.dataset.state = state;
  statusTitle.textContent = title;
  statusBadge.textContent = badge || state;
  statusMessage.textContent = message;
  loadingBar.hidden = state !== "running" && state !== "queued";
}

function setMeta(job) {
  const rows = [];
  if (job.id) rows.push(["任务", job.id]);
  if (job.identifier_type) rows.push(["类型", job.identifier_type]);
  if (job.display_name) rows.push(["对象", job.display_name]);
  if (job.output_dir) rows.push(["输出", job.output_dir]);
  if (job.elapsed !== null && job.elapsed !== undefined) {
    rows.push(["耗时", `${Number(job.elapsed).toFixed(1)} 秒`]);
  }
  statusMeta.replaceChildren();
  rows.forEach(([label, value]) => {
    const row = document.createElement("div");
    row.textContent = `${label}: ${value}`;
    statusMeta.appendChild(row);
  });
  statusMeta.hidden = rows.length === 0;
}

function renderRisk(job) {
  riskSummary.replaceChildren();
  const items = [];
  if (job.risk_level) items.push(["风险等级", job.risk_level]);
  if (job.risk_score) items.push(["风险评分", job.risk_score]);
  if (items.length === 0) {
    riskSummary.hidden = true;
    return;
  }
  items.forEach(([label, value]) => {
    const row = document.createElement("div");
    const strong = document.createElement("strong");
    strong.textContent = label;
    row.append(strong, `: ${value}`);
    riskSummary.appendChild(row);
  });
  riskSummary.hidden = false;
}

function renderFiles(files) {
  fileLinks.replaceChildren();
  const labels = {
    html: "打开 HTML 报告",
    md: "查看 Markdown",
    json: "查看 JSON 备份",
  };
  Object.entries(labels).forEach(([key, label]) => {
    if (!files || !files[key]) return;
    const link = document.createElement("a");
    link.href = files[key];
    link.target = "_blank";
    link.rel = "noreferrer";
    link.textContent = label;
    fileLinks.appendChild(link);
  });
}

function renderJob(job) {
  setMeta(job);

  if (job.status === "queued" || job.status === "running") {
    setStatus("running", "调查正在运行", job.message || "调查正在运行。", job.status);
    resultPanel.hidden = true;
    setButtonBusy(true);
    return;
  }

  if (job.status === "failed") {
    setStatus("failed", "调查失败", job.error || job.message || "调查失败。", "failed");
    resultPanel.hidden = true;
    setButtonBusy(false);
    stopPolling();
    return;
  }

  if (job.status === "succeeded") {
    setStatus("succeeded", "调查完成", job.message || "报告文件已生成。", "succeeded");
    resultPanel.hidden = false;
    resultTitle.textContent = job.title || "调查完成";
    renderRisk(job);
    renderFiles(job.files);
    markdownPreview.textContent = job.markdown_preview || "暂无 Markdown 预览。";
    setButtonBusy(false);
    stopPolling();
  }
}

async function pollJob(jobId) {
  try {
    const response = await fetch(`/api/jobs/${jobId}`);
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "查询任务失败。");
    }
    renderJob(data.job);
  } catch (error) {
    setStatus("failed", "状态查询失败", error.message, "failed");
    resultPanel.hidden = true;
    setButtonBusy(false);
    stopPolling();
  }
}

function startPolling(jobId) {
  stopPolling();
  pollJob(jobId);
  pollTimer = window.setInterval(() => pollJob(jobId), 2200);
}

function showLocalError(message) {
  setStatus("failed", "无法创建任务", message, "failed");
  statusMeta.hidden = true;
  resultPanel.hidden = true;
  setButtonBusy(false);
}

form.addEventListener("change", updateModeCopy);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  stopPolling();

  const payload = {
    identifier_type: selectedType(),
    paper_identifier: identifierInput.value.trim(),
    output_dir: outputDirInput.value.trim(),
  };

  if (!payload.paper_identifier) {
    showLocalError("调查对象不能为空。");
    return;
  }

  setButtonBusy(true);
  setStatus("running", "正在创建任务", "正在校验输入并创建后台调查任务。", "queued");
  statusMeta.hidden = true;
  resultPanel.hidden = true;

  try {
    const response = await fetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      const detailText = data.details?.errors?.join(" ");
      throw new Error(detailText || data.error || "创建任务失败。");
    }
    renderJob(data.job);
    startPolling(data.job_id);
  } catch (error) {
    showLocalError(error.message);
  }
});

resetButton.addEventListener("click", () => {
  stopPolling();
  form.reset();
  updateModeCopy();
  setButtonBusy(false);
  setStatus(
    "idle",
    "尚未开始调查",
    "选择 DOI、论文标题、本机 PDF 或案例目录后启动。报告会保存为 JSON、Markdown 和 HTML。",
    "idle",
  );
  statusMeta.hidden = true;
  resultPanel.hidden = true;
});

updateModeCopy();
