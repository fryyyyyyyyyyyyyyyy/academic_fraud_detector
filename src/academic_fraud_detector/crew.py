"""
Academic Fraud Detection Crew — assembly of all agents, tasks, and tools.

Uses the @CrewBase decorator pattern (CrewAI best practice).
The crew uses Hierarchical process: Lead Investigator manages all specialists,
dynamically delegating tasks and adjusting investigation depth.

Current local investigation dimensions:
1. Raw Data Integrity — XLSX raw data, deterministic statistics, PDF/table alignment
2. Methodology Consistency — reagent verification, ethics check, timeline

Full/API mode still keeps legacy literature, citation, peer-review, productivity,
and image-forensics components available, but local case mode does not run image processing.

Usage:
    crew = AcademicFraudDetectionCrew().crew()
    result = crew.kickoff(inputs={
        "paper_identifier": "10.1038/nature12345",
        "identifier_type": "doi",
    })
"""

# ruff: noqa: E402

import json
import os
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, task, crew, before_kickoff, after_kickoff

# Load .env at module level
load_dotenv()

logger = logging.getLogger(__name__)


def _create_llm(model_env_var: str, default_model: str) -> LLM:
    """
    Create an LLM instance with provider-agnostic configuration.

    Supports OpenAI, DeepSeek, Anthropic, and any OpenAI-compatible API.
    Detects provider from environment variables:
    - OPENAI_API_BASE → custom base_url (e.g., https://api.deepseek.com)
    - ANTHROPIC_API_KEY → uses Anthropic provider
    - Otherwise defaults to OpenAI
    """
    model = os.getenv(model_env_var, default_model)
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")

    if base_url:
        # Custom OpenAI-compatible provider (DeepSeek, Groq, etc.)
        return LLM(model=model, base_url=base_url, api_key=api_key)
    elif os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        # Anthropic-only setup
        return LLM(model=model, api_key=os.getenv("ANTHROPIC_API_KEY"))
    else:
        # Default OpenAI
        return LLM(model=model, api_key=api_key)

from .tools.paper_fetching import (
    ArxivSearchTool,
    CrossrefSearchTool,
    SemanticScholarSearchTool,
    PaperLookupTool,
    LocalPaperLoaderTool,
)
from .tools.text_similarity import SemanticSimilarityTool, lexical_plagiarism_check
from .tools.image_forensics import (
    ELATool, CloneDetectionTool, AIImageDetectionTool,
    CrossImageDuplicateTool, BackgroundConsistencyTool,
    FeatureBasedDuplicateTool,
)
from .tools.statistical_analysis import (
    BenfordLawTool, PValueDistributionTool, GRIMTestTool,
    AnomalousPrecisionTool, StatisticalConsistencyTool,
    CrossFigureDataComparisonTool,
)
from .utils.chart_ocr import ChartOCRTool, BatchChartOCRTool, BarChartExtractionTool
from .tools.citation_analysis import (
    CitationGraphTool,
    SelfCitationTool,
    CitationClaimVerifierTool,
)
from .tools.peer_review_analysis import (
    ReviewTextAnalyzerTool,
    ReviewerCredentialCheckerTool,
    ReviewTemplateDetectorTool,
)
from .tools.methodology_audit import (
    ReagentVerificationTool,
    EthicsApprovalCheckTool,
    ExperimentalTimelineCheckTool,
    MethodInternalConsistencyTool,
)
from .tools.productivity_analysis import (
    PublicationFrequencyTool,
    SalamiSlicingTool,
)
from .tools.web_search import (
    AcademicWebSearchTool,
    CitationExistenceCheckTool,
)
from .utils.case_folder import discover_case_folder
from .utils.raw_data_loader import load_raw_data_files
from .utils.raw_data_precheck import format_raw_data_precheck_for_agent, run_raw_data_precheck

logger = logging.getLogger(__name__)


@CrewBase
class AcademicFraudDetectionCrew:
    """
    Multi-agent academic fraud detection crew.

    Architecture:
    - 1 Manager (Lead Investigator) — coordinates, delegates, decides
    - 5 Specialist agents — each investigates one fraud dimension
    - 1 Synthesizer — aggregates findings into final report

    Process: Hierarchical — enables parallel investigation + adaptive depth.
    """

    agents_config = str(Path(__file__).parent / "config" / "agents.yaml")
    tasks_config = str(Path(__file__).parent / "config" / "tasks.yaml")

    # ═══════════════════════════════════════════════════════════════════
    # Tool Instances (created once, shared across agents)
    # ═══════════════════════════════════════════════════════════════════

    def __init__(self, local_only: bool = False):
        """
        Args:
            local_only: If True, agents get ONLY local tools (no arXiv/CrossRef/
                        Semantic Scholar). Use this for user-uploaded PDFs where
                        external API calls are not desired.
        """
        self._local_only = local_only

        # Paper fetching tools
        self._local_paper_loader = LocalPaperLoaderTool()
        if not local_only:
            self._arxiv_search = ArxivSearchTool()
            self._crossref_search = CrossrefSearchTool()
            self._s2_search = SemanticScholarSearchTool()
            self._paper_lookup = PaperLookupTool()

        # Text similarity tools
        self._semantic_similarity = SemanticSimilarityTool()
        # lexical_plagiarism_check is a @tool-decorated function, passed directly

        # Image forensics tools are kept for full/API mode only. Local case mode is
        # raw-data-first and must not run image extraction, OCR, or panel comparison.
        if not local_only:
            self._ela = ELATool()
            self._clone_detection = CloneDetectionTool()
            self._ai_image = AIImageDetectionTool()
            self._cross_image_duplicate = CrossImageDuplicateTool()
            self._background_consistency = BackgroundConsistencyTool()
            self._feature_duplicate = FeatureBasedDuplicateTool()

        # Statistical tools
        self._benford = BenfordLawTool()
        self._pvalue = PValueDistributionTool()
        self._grim = GRIMTestTool()
        self._anomalous_precision = AnomalousPrecisionTool()
        self._statistical_consistency = StatisticalConsistencyTool()
        self._cross_figure_data = CrossFigureDataComparisonTool()

        # OCR tools are image-derived, so they are disabled in local-only mode.
        if not local_only:
            self._chart_ocr = ChartOCRTool()
            self._batch_chart_ocr = BatchChartOCRTool()
            self._bar_chart_extract = BarChartExtractionTool()

        # Citation tools
        if not local_only:
            self._citation_graph = CitationGraphTool()
            self._self_citation = SelfCitationTool()
            self._citation_claim_verifier = CitationClaimVerifierTool()

        # Peer review tools
        self._review_text = ReviewTextAnalyzerTool()
        self._reviewer_cred = ReviewerCredentialCheckerTool()
        self._template_detect = ReviewTemplateDetectorTool()

        # Methodology audit tools (available in both modes — purely local)
        self._reagent_verify = ReagentVerificationTool()
        self._ethics_check = EthicsApprovalCheckTool()
        self._timeline_check = ExperimentalTimelineCheckTool()
        self._method_consistency = MethodInternalConsistencyTool()

        # Productivity analysis tools (API-dependent, only in full mode)
        if not local_only:
            self._pub_frequency = PublicationFrequencyTool()
            self._salami_slicing = SalamiSlicingTool()
            self._web_search = AcademicWebSearchTool()
            self._citation_check = CitationExistenceCheckTool()

    # ═══════════════════════════════════════════════════════════════════
    # Agent Definitions
    # ═══════════════════════════════════════════════════════════════════

    @agent
    def lead_investigator(self) -> Agent:
        """Manager agent — coordinates the entire investigation."""
        return Agent(
            config=self.agents_config["lead_investigator"],
            llm=_create_llm("MANAGER_MODEL", "deepseek-chat"),
        )

    # ═══════════════════════════════════════════════════════════════════
    # Plagiarism Detective — DISABLED from crew, but kept instantiated
    # so YAML config references still resolve. Tools preserved for reuse.
    # This agent is NOT included in worker_agents list → not used.
    # ═══════════════════════════════════════════════════════════════════
    @agent
    def plagiarism_detective(self) -> Agent:
        """[HIDDEN] Plagiarism detection specialist — not used in current pipeline."""
        if self._local_only:
            tools = [
                self._local_paper_loader,
                self._semantic_similarity,
                lexical_plagiarism_check,
            ]
        else:
            tools = [
                self._local_paper_loader,
                self._arxiv_search,
                self._crossref_search,
                self._s2_search,
                self._paper_lookup,
                self._semantic_similarity,
                lexical_plagiarism_check,
            ]
        return Agent(
            config=self.agents_config["plagiarism_detective"],
            llm=_create_llm("AGENT_MODEL", "deepseek-chat"),
            tools=tools,
        )

    @agent
    def image_forensics_analyst(self) -> Agent:
        """Image manipulation detection specialist.

        CrewBase resolves YAML task-agent references during construction, even when
        local mode later excludes the image task. Keep this method local-safe without
        instantiating or dereferencing image tools in local-only mode.
        """
        if self._local_only:
            tools = [self._local_paper_loader]
        else:
            tools = [
                self._ela,
                self._clone_detection,
                self._ai_image,
                self._cross_image_duplicate,
                self._background_consistency,
                self._feature_duplicate,
            ]
        return Agent(
            config=self.agents_config["image_forensics_analyst"],
            llm=_create_llm("AGENT_MODEL", "deepseek-chat"),
            tools=tools,
        )

    @agent
    def data_integrity_auditor(self) -> Agent:
        """Statistical data fabrication detection specialist."""
        tools = [
            self._benford,
            self._pvalue,
            self._grim,
            self._anomalous_precision,
            self._statistical_consistency,
            self._cross_figure_data,
        ]
        if self._local_only:
            tools.insert(0, self._local_paper_loader)
        else:
            tools.extend([
                self._chart_ocr,
                self._batch_chart_ocr,
                self._bar_chart_extract,
            ])
        return Agent(
            config=self.agents_config["data_integrity_auditor"],
            llm=_create_llm("AGENT_MODEL", "deepseek-chat"),
            tools=tools,
        )

    @agent
    def citation_network_investigator(self) -> Agent:
        """Citation manipulation detection specialist."""
        if self._local_only:
            tools = [self._local_paper_loader]  # Only has access to paper text
        else:
            tools = [
                self._citation_graph,
                self._self_citation,
                self._citation_claim_verifier,
                self._citation_check,
                self._web_search,
                self._s2_search,
            ]
        return Agent(
            config=self.agents_config["citation_network_investigator"],
            llm=_create_llm("AGENT_MODEL", "deepseek-chat"),
            tools=tools,
        )

    @agent
    def peer_review_inspector(self) -> Agent:
        """Peer review fraud detection specialist."""
        if self._local_only:
            tools = [self._local_paper_loader]  # Only has access to paper text
        else:
            tools = [
                self._review_text,
                self._reviewer_cred,
                self._template_detect,
                self._s2_search,
            ]
        return Agent(
            config=self.agents_config["peer_review_inspector"],
            llm=_create_llm("AGENT_MODEL", "deepseek-chat"),
            tools=tools,
        )

    @agent
    def methodology_consistency_reviewer(self) -> Agent:
        """Methodology consistency and experimental design review specialist."""
        tools = [
            self._reagent_verify,
            self._ethics_check,
            self._timeline_check,
            self._method_consistency,
        ]
        if not self._local_only:
            # Web search enables real-time reagent/citation verification
            tools.append(self._web_search)
        if self._local_only:
            tools.insert(0, self._local_paper_loader)
        return Agent(
            config=self.agents_config["methodology_consistency_reviewer"],
            llm=_create_llm("AGENT_MODEL", "deepseek-chat"),
            tools=tools,
        )

    @agent
    def productivity_anomaly_analyst(self) -> Agent:
        """Publication productivity anomaly analysis specialist."""
        if self._local_only:
            tools = [self._local_paper_loader]  # Minimal tools in local mode
        else:
            tools = [
                self._pub_frequency,
                self._salami_slicing,
                self._web_search,
                self._citation_check,
                self._s2_search,
                self._semantic_similarity,
                lexical_plagiarism_check,
            ]
        return Agent(
            config=self.agents_config["productivity_anomaly_analyst"],
            llm=_create_llm("AGENT_MODEL", "deepseek-chat"),
            tools=tools,
        )

    @agent
    def evidence_synthesizer(self) -> Agent:
        """Evidence synthesis and report writing specialist."""
        if self._local_only:
            tools = [self._local_paper_loader]
        else:
            tools = [self._paper_lookup]
        return Agent(
            config=self.agents_config["evidence_synthesizer"],
            llm=_create_llm("MANAGER_MODEL", "deepseek-chat"),
            tools=tools,
        )

    # ═══════════════════════════════════════════════════════════════════
    # Task Definitions
    # ═══════════════════════════════════════════════════════════════════

    @task
    def acquire_target_paper(self) -> Task:
        """Retrieve the target paper's metadata and content."""
        return Task(config=self.tasks_config["acquire_target_paper"])

    # ═══════════════════════════════════════════════════════════════════
    # Plagiarism task — disabled from crew, kept for YAML resolution
    # ═══════════════════════════════════════════════════════════════════
    @task
    def plagiarism_investigation(self) -> Task:
        """[HIDDEN] Investigate the target paper for plagiarism."""
        return Task(config=self.tasks_config["plagiarism_investigation"])

    @task
    def image_forensics_investigation(self) -> Task:
        """Investigate the target paper's figures for manipulation."""
        return Task(config=self.tasks_config["image_forensics_investigation"])

    @task
    def data_integrity_investigation(self) -> Task:
        """Audit the target paper's statistical data for fabrication."""
        return Task(config=self.tasks_config["data_integrity_investigation"])

    @task
    def citation_network_investigation(self) -> Task:
        """Investigate the target paper's citation patterns."""
        return Task(config=self.tasks_config["citation_network_investigation"])

    @task
    def peer_review_investigation(self) -> Task:
        """Investigate the target paper's peer review process."""
        return Task(config=self.tasks_config["peer_review_investigation"])

    @task
    def methodology_audit(self) -> Task:
        """Audit the paper's methodology for internal consistency and logic."""
        return Task(config=self.tasks_config["methodology_audit"])

    @task
    def productivity_anomaly_investigation(self) -> Task:
        """Analyze the authors' publication patterns for anomalies."""
        return Task(config=self.tasks_config["productivity_anomaly_investigation"])

    @task
    def synthesize_findings(self) -> Task:
        """Synthesize all findings into a final investigation report."""
        return Task(config=self.tasks_config["synthesize_findings"])

    @task
    def local_synthesize_findings(self) -> Task:
        """Synthesize image + data + methodology findings (local PDF mode)."""
        return Task(config=self.tasks_config["local_synthesize_findings"])

    # ═══════════════════════════════════════════════════════════════════
    # Crew Assembly
    # ═══════════════════════════════════════════════════════════════════

    @crew
    def crew(self) -> Crew:
        """
        Assemble the investigation crew.

        In local_only mode: focuses on PDF text/tables + XLSX raw-data integrity +
        methodology consistency. Image processing is not executed.

        Uses Hierarchical process:
        - Lead Investigator is the manager (NOT in agents list)
        - Specialist agents execute tasks
        """
        manager = self.lead_investigator()

        if self._local_only:
            # Only tasks that work with local PDF text/tables and raw XLSX data.
            tasks = [
                self.acquire_target_paper(),
                self.data_integrity_investigation(),
                self.methodology_audit(),
                self.local_synthesize_findings(),
            ]
            worker_agents = [
                self.data_integrity_auditor(),
                self.methodology_consistency_reviewer(),
                self.evidence_synthesizer(),
            ]
            logger.info(
                f"Local-only mode: {len(tasks)} tasks, "
                f"{len(worker_agents)} worker agents "
                f"(raw data integrity + methodology audit; no image processing)"
            )
        else:
            # Full mode: all tasks except plagiarism (disabled)
            tasks = [
                t for t in self.tasks
                if t.name != "plagiarism_investigation"
            ]
            worker_agents = [a for a in self.agents if a.role != manager.role]
            logger.info(
                f"Full mode: {len(tasks)} tasks, "
                f"{len(worker_agents)} worker agents "
                f"(6 investigation dimensions)"
            )

        return Crew(
            agents=worker_agents,
            tasks=tasks,
            process=Process.hierarchical,
            manager_agent=manager,
            planning=False,
            memory=False,
            verbose=True,
            max_rpm=10,
        )

    # ═══════════════════════════════════════════════════════════════════
    # Local PDF preloading helpers
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _json_for_task(value: Any) -> str:
        """Serialize injected task context consistently for YAML templates."""
        return json.dumps(value, ensure_ascii=False, default=str)

    @staticmethod
    def _coerce_local_paper_payload(raw_output: Any, pdf_path: str) -> dict[str, Any]:
        """Normalize LocalPaperLoaderTool output into a dict payload."""
        if isinstance(raw_output, dict):
            return raw_output

        if isinstance(raw_output, str):
            try:
                payload = json.loads(raw_output)
                if isinstance(payload, dict):
                    return payload
                return {
                    "source": "local_pdf",
                    "file_path": pdf_path,
                    "raw_output": payload,
                    "warning": "LocalPaperLoaderTool returned JSON that was not an object.",
                }
            except json.JSONDecodeError:
                return {
                    "source": "local_pdf",
                    "file_path": pdf_path,
                    "raw_output": raw_output,
                    "warning": "LocalPaperLoaderTool returned a non-JSON string.",
                }

        return {
            "source": "local_pdf",
            "file_path": pdf_path,
            "raw_output": str(raw_output),
            "warning": (
                "LocalPaperLoaderTool returned unexpected type: "
                f"{type(raw_output).__name__}"
            ),
        }

    @staticmethod
    def _local_paper_stats_payload(payload: dict[str, Any]) -> dict[str, Any]:
        """Build a compact statistics/extraction payload for data-audit tasks."""
        return {
            "pre_extracted_stats": payload.get("pre_extracted_stats", {}),
            "tables": payload.get("tables", []),
            "mineru": payload.get("mineru", {}),
            "page_count": payload.get("page_count", 0),
            "full_text_available": payload.get("full_text_available", False),
            "full_text_length_chars": payload.get("full_text_length_chars", 0),
            "image_count": len(payload.get("images") or []),
            "panel_count": len(payload.get("panels") or []),
            "table_count": len(payload.get("tables") or []),
            "error": payload.get("error"),
        }

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        """Read a non-negative integer environment variable with a safe default."""
        raw = os.getenv(name)
        if raw is None or raw.strip() == "":
            return default
        try:
            return max(0, int(raw))
        except ValueError:
            logger.warning("Ignoring invalid integer value for %s=%r", name, raw)
            return default

    @staticmethod
    def _existing_paths_from_entries(entries: Any) -> list[str]:
        """Extract unique existing local file paths from image/panel payload entries."""
        paths: list[str] = []
        seen = set()
        for entry in entries or []:
            if isinstance(entry, dict):
                raw_path = (
                    entry.get("filepath")
                    or entry.get("path")
                    or entry.get("image_path")
                    or entry.get("filename")
                )
            else:
                raw_path = entry

            if not raw_path:
                continue
            path = str(raw_path)
            if path in seen or not os.path.exists(path):
                continue
            seen.add(path)
            paths.append(path)
        return paths

    @staticmethod
    def _take_with_limit(paths: list[str], limit: int) -> tuple[list[str], int]:
        """Return paths selected for precheck and the number omitted by the cap."""
        if limit <= 0:
            return [], len(paths)
        selected = paths[:limit]
        return selected, max(0, len(paths) - len(selected))

    @staticmethod
    def _parse_tool_json_output(raw_output: Any) -> dict[str, Any]:
        """Normalize a forensic tool output into a JSON-compatible dictionary."""
        if isinstance(raw_output, dict):
            return raw_output
        if isinstance(raw_output, str):
            try:
                parsed = json.loads(raw_output)
                if isinstance(parsed, dict):
                    return parsed
                return {"raw_output": parsed, "flagged": False}
            except json.JSONDecodeError:
                return {
                    "error": "Tool returned a non-JSON string.",
                    "raw_output": raw_output,
                    "flagged": False,
                }
        return {"raw_output": str(raw_output), "flagged": False}

    def _run_forensics_tool(self, tool: Any, **kwargs: Any) -> dict[str, Any]:
        """Run one image forensics tool and capture status plus parsed output."""
        tool_name = getattr(tool, "name", tool.__class__.__name__)
        try:
            output = self._parse_tool_json_output(tool._run(**kwargs))
            matches = output.get("matches")
            if isinstance(matches, list) and len(matches) > 50:
                output = dict(output)
                output.setdefault("match_count", len(matches))
                output["matches"] = matches[:50]
                output["matches_truncated"] = len(matches) - 50
            status = "error" if output.get("error") else "success"
            return {"tool": tool_name, "status": status, "output": output}
        except Exception as e:
            logger.warning("Image forensics tool %s failed: %s", tool_name, e)
            return {
                "tool": tool_name,
                "status": "error",
                "output": {"error": str(e), "flagged": False},
            }

    @staticmethod
    def _record_flagged(record: dict[str, Any]) -> bool:
        """Return whether a captured tool record contains a positive finding."""
        return bool(record.get("output", {}).get("flagged"))

    def _run_image_forensics_precheck(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run deterministic image forensics tools on preloaded local PDF images."""
        image_paths = self._existing_paths_from_entries(payload.get("images", []))
        panel_paths = self._existing_paths_from_entries(payload.get("panels", []))

        max_single_images = self._env_int("IMAGE_FORENSICS_MAX_SINGLE_IMAGES", 12)
        max_cross_images = self._env_int("IMAGE_FORENSICS_MAX_CROSS_IMAGES", 200)
        max_feature_panels = self._env_int("IMAGE_FORENSICS_MAX_FEATURE_PANELS", 80)

        single_paths, single_omitted = self._take_with_limit(image_paths, max_single_images)
        cross_paths, cross_omitted = self._take_with_limit(image_paths, max_cross_images)
        feature_paths, feature_omitted = self._take_with_limit(panel_paths, max_feature_panels)

        result: dict[str, Any] = {
            "status": "success" if image_paths or panel_paths else "no_input",
            "input_counts": {
                "images_total": len(payload.get("images") or []),
                "images_existing": len(image_paths),
                "panels_total": len(payload.get("panels") or []),
                "panels_existing": len(panel_paths),
            },
            "limits": {
                "max_single_images": max_single_images,
                "max_cross_images": max_cross_images,
                "max_feature_panels": max_feature_panels,
            },
            "coverage": {
                "single_image_tools_analyzed": len(single_paths),
                "single_image_tools_omitted": single_omitted,
                "cross_image_paths_analyzed": len(cross_paths),
                "cross_image_paths_omitted": cross_omitted,
                "feature_panel_paths_analyzed": len(feature_paths),
                "feature_panel_paths_omitted": feature_omitted,
            },
            "per_image_results": [],
            "cross_image_duplicate": {
                "tool": getattr(self._cross_image_duplicate, "name", "cross_image_duplicate_check"),
                "status": "skipped",
                "reason": "Need at least 2 existing image paths.",
            },
            "feature_based_duplicate": {
                "tool": getattr(self._feature_duplicate, "name", "feature_based_duplicate_check"),
                "status": "skipped",
                "reason": "Need at least 2 existing panel paths.",
            },
            "tools_attempted": [],
            "flagged_tools": [],
        }

        if not image_paths and not panel_paths:
            result["message"] = "No existing image or panel paths were available for forensic tools."
            return result

        for path in single_paths:
            per_image = {"image_path": path, "tools": {}}
            per_image["tools"]["error_level_analysis"] = self._run_forensics_tool(
                self._ela, image_path_or_url=path
            )
            per_image["tools"]["clone_detection"] = self._run_forensics_tool(
                self._clone_detection, image_path_or_url=path
            )
            per_image["tools"]["ai_image_detection"] = self._run_forensics_tool(
                self._ai_image, image_path_or_url=path
            )
            per_image["tools"]["background_consistency_check"] = self._run_forensics_tool(
                self._background_consistency, image_path_or_url=path
            )
            result["per_image_results"].append(per_image)

        if len(cross_paths) >= 2:
            result["cross_image_duplicate"] = self._run_forensics_tool(
                self._cross_image_duplicate,
                image_paths=json.dumps(cross_paths, ensure_ascii=False),
            )

        if len(feature_paths) >= 2:
            result["feature_based_duplicate"] = self._run_forensics_tool(
                self._feature_duplicate,
                image_paths=json.dumps(feature_paths, ensure_ascii=False),
                min_inliers=8,
                ratio_threshold=0.80,
                sift_contrast_threshold=0.02,
            )

        attempted: list[str] = []
        flagged: list[str] = []
        for per_image in result["per_image_results"]:
            for record in per_image["tools"].values():
                attempted.append(record["tool"])
                if self._record_flagged(record):
                    flagged.append(record["tool"])
        for key in ("cross_image_duplicate", "feature_based_duplicate"):
            record = result[key]
            if record.get("status") != "skipped":
                attempted.append(record["tool"])
                if self._record_flagged(record):
                    flagged.append(record["tool"])

        result["tools_attempted"] = sorted(set(attempted))
        result["flagged_tools"] = sorted(set(flagged))
        result["flagged_check_count"] = len(flagged)
        return result

    @staticmethod
    def _format_image_forensics_precheck(precheck: dict[str, Any]) -> str:
        """Format deterministic image-forensics output for the YAML task prompt."""
        if precheck.get("status") == "no_input":
            return "⚠️ 图像取证预检未执行：没有可读取的本地图片或面板路径。"

        counts = precheck.get("input_counts", {})
        coverage = precheck.get("coverage", {})
        lines = [
            "### 图像取证工具预检（系统已确定性执行）",
            "",
            f"- 可读取图片：{counts.get('images_existing', 0)}/"
            f"{counts.get('images_total', 0)}",
            f"- 可读取面板：{counts.get('panels_existing', 0)}/"
            f"{counts.get('panels_total', 0)}",
            f"- 单图取证覆盖：{coverage.get('single_image_tools_analyzed', 0)} 张"
            f"（因安全上限省略 {coverage.get('single_image_tools_omitted', 0)} 张）",
            f"- 整图 pHash 跨图比对覆盖：{coverage.get('cross_image_paths_analyzed', 0)} 张"
            f"（省略 {coverage.get('cross_image_paths_omitted', 0)} 张）",
            f"- 面板 SIFT/RANSAC 比对覆盖：{coverage.get('feature_panel_paths_analyzed', 0)} 个面板"
            f"（省略 {coverage.get('feature_panel_paths_omitted', 0)} 个）",
            f"- 已尝试工具：{', '.join(precheck.get('tools_attempted', [])) or '无'}",
            f"- 阳性工具：{', '.join(precheck.get('flagged_tools', [])) or '无'}",
        ]

        cross = precheck.get("cross_image_duplicate", {})
        feature = precheck.get("feature_based_duplicate", {})
        lines.extend([
            "",
            "#### 整图跨图重复检测",
            f"- 工具：{cross.get('tool', 'cross_image_duplicate_check')}",
            f"- 状态：{cross.get('status', 'unknown')}",
            f"- flagged：{cross.get('output', {}).get('flagged', False)}",
            f"- match_count：{cross.get('output', {}).get('match_count', 0)}",
            "",
            "#### 面板级特征重复检测",
            f"- 工具：{feature.get('tool', 'feature_based_duplicate_check')}",
            f"- 状态：{feature.get('status', 'unknown')}",
            f"- flagged：{feature.get('output', {}).get('flagged', False)}",
            f"- match_count：{feature.get('output', {}).get('match_count', 0)}",
        ])
        return "\n".join(lines)

    def _inject_image_forensics_precheck(self, inputs: dict) -> None:
        """Run deterministic image-forensics precheck and inject task context."""
        try:
            payload = inputs.get("local_paper_payload") or {}
            precheck = self._run_image_forensics_precheck(payload)
            inputs["image_forensics_precheck"] = self._format_image_forensics_precheck(precheck)
            inputs["image_forensics_precheck_json"] = self._json_for_task(precheck)
        except Exception as e:
            logger.warning("Image forensics precheck failed: %s", e)
            precheck = {"status": "error", "error": str(e)}
            inputs["image_forensics_precheck"] = (
                "⚠️ 图像取证工具预检未能完成（技术错误）。"
                f"错误信息：{e}"
            )
            inputs["image_forensics_precheck_json"] = self._json_for_task(precheck)

    def _inject_local_paper_payload(
        self, inputs: dict, pdf_path: str, extract_images: bool = False
    ) -> None:
        """Preload local PDF deterministically and inject structured task context."""
        logger.info(
            "Preloading local PDF via LocalPaperLoaderTool: %s (extract_images=%s)",
            pdf_path,
            extract_images,
        )
        try:
            raw_output = self._local_paper_loader._run(
                pdf_path,
                extract_images=extract_images,
                extract_tables=True,
            )
            payload = self._coerce_local_paper_payload(raw_output, pdf_path)
            inputs["local_paper_payload"] = payload
            inputs["local_paper_payload_json"] = self._json_for_task(payload)
            inputs["local_paper_load_status"] = "error" if payload.get("error") else "success"
            inputs["local_paper_load_error"] = str(payload.get("error") or "")
            inputs["local_paper_summary"] = payload.get("_summary", "")
            inputs["local_paper_images_json"] = self._json_for_task(payload.get("images", []))
            inputs["local_paper_panels_json"] = self._json_for_task(payload.get("panels", []))
            inputs["local_paper_stats_json"] = self._json_for_task(
                self._local_paper_stats_payload(payload)
            )
            inputs["local_paper_text"] = payload.get("full_text") or ""
            logger.info(
                "Local PDF preload complete: %s chars text, %s tables, images disabled=%s",
                payload.get("full_text_length_chars", 0),
                len(payload.get("tables") or []),
                not extract_images,
            )
        except Exception as e:
            logger.warning("Local PDF preload failed: %s", e)
            payload = {
                "source": "local_pdf",
                "file_path": pdf_path,
                "full_text_available": False,
                "full_text": None,
                "images": [],
                "panels": [],
                "tables": [],
                "pre_extracted_stats": {},
                "mineru": {"used": False},
                "error": str(e),
            }
            inputs["local_paper_payload"] = payload
            inputs["local_paper_payload_json"] = self._json_for_task(payload)
            inputs["local_paper_load_status"] = "error"
            inputs["local_paper_load_error"] = str(e)
            inputs["local_paper_summary"] = ""
            inputs["local_paper_images_json"] = "[]"
            inputs["local_paper_panels_json"] = "[]"
            inputs["local_paper_stats_json"] = self._json_for_task(
                self._local_paper_stats_payload(payload)
            )
            inputs["local_paper_text"] = ""

    def _inject_local_case_payload(self, inputs: dict, case_dir: str) -> None:
        """Discover case folder, load PDF text/tables, parse XLSX, and precompute evidence."""
        manifest = discover_case_folder(case_dir)
        inputs["case_manifest"] = manifest
        inputs["case_manifest_json"] = self._json_for_task(manifest)
        if manifest.get("errors"):
            raise ValueError("; ".join(str(e) for e in manifest["errors"]))

        pdf_path = str(manifest["selected_pdf"])
        raw_paths = [entry["path"] for entry in manifest.get("raw_data_files", [])]
        inputs["case_dir"] = case_dir
        inputs["case_pdf_path"] = pdf_path
        inputs["raw_data_files_json"] = self._json_for_task(manifest.get("raw_data_files", []))

        self._inject_local_paper_payload(inputs, pdf_path, extract_images=False)

        raw_payload = load_raw_data_files(raw_paths)
        inputs["raw_data_payload"] = raw_payload
        inputs["raw_data_datasets_json"] = self._json_for_task(raw_payload.get("datasets", []))
        inputs["raw_data_profile_json"] = self._json_for_task(raw_payload.get("profile", {}))

        precheck = run_raw_data_precheck(raw_payload, inputs.get("local_paper_payload") or {})
        inputs["raw_data_precheck"] = format_raw_data_precheck_for_agent(precheck)
        inputs["raw_data_precheck_json"] = self._json_for_task(precheck)
        inputs["deterministic_evidence_json"] = self._json_for_task(
            precheck.get("deterministic_findings", [])
        )
        inputs["confidence_summary_json"] = self._json_for_task(
            precheck.get("confidence_summary", {})
        )
        inputs["allowed_claims_json"] = self._json_for_task(precheck.get("allowed_claims", []))
        logger.info(
            "Local case preload complete: %s XLSX files, %s datasets, %s evidence items",
            len(raw_paths),
            raw_payload.get("profile", {}).get("dataset_count", 0),
            precheck.get("confidence_summary", {}).get("evidence_count", 0),
        )

    # ═══════════════════════════════════════════════════════════════════
    # Hooks
    # ═══════════════════════════════════════════════════════════════════

    @before_kickoff
    def validate_inputs(self, inputs: dict) -> dict:
        """
        Validate required inputs and inject auto-generated values.

        Required inputs:
        - paper_identifier: DOI, arXiv ID, title, or URL of the target paper.
        - identifier_type: One of 'doi', 'arxiv_id', 'title', 'url'.

        Auto-injected:
        - timestamp: ISO 8601 timestamp for report identification.
        """
        from datetime import datetime

        required = ["paper_identifier", "identifier_type"]
        for key in required:
            if key not in inputs:
                raise ValueError(
                    f"Missing required input: '{key}'. "
                    f"Must provide: {', '.join(required)}."
                )

        valid_types = {
            "doi",
            "arxiv_id",
            "title",
            "url",
            "semantic_scholar_id",
            "local_pdf",
            "local_case",
        }
        if inputs["identifier_type"] not in valid_types:
            raise ValueError(
                f"Invalid identifier_type '{inputs['identifier_type']}'. "
                f"Must be one of: {', '.join(sorted(valid_types))}."
            )

        # Validate local PDF / local case paths before expensive parsing.
        if inputs["identifier_type"] == "local_pdf":
            pdf_path = inputs["paper_identifier"]
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"Local PDF file not found: {pdf_path}")
            if not pdf_path.lower().endswith(".pdf"):
                raise ValueError(f"File must be a PDF: {pdf_path}")
        if inputs["identifier_type"] == "local_case":
            case_dir = inputs.get("case_dir") or inputs["paper_identifier"]
            if not os.path.exists(case_dir):
                raise FileNotFoundError(f"Local case directory not found: {case_dir}")
            if not os.path.isdir(case_dir):
                raise ValueError(f"local_case must be a directory: {case_dir}")

        # Inject auto-generated values needed by task templates
        inputs.setdefault("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
        inputs.setdefault("local_paper_payload", {})
        inputs.setdefault("local_paper_payload_json", "{}")
        inputs.setdefault("local_paper_load_status", "not_applicable")
        inputs.setdefault("local_paper_load_error", "")
        inputs.setdefault("local_paper_summary", "")
        inputs.setdefault("local_paper_images_json", "[]")
        inputs.setdefault("local_paper_panels_json", "[]")
        inputs.setdefault("local_paper_stats_json", "{}")
        inputs.setdefault("local_paper_text", "")
        inputs.setdefault("image_forensics_precheck", "当前模式不执行图像取证工具预检。")
        inputs.setdefault("image_forensics_precheck_json", "{}")
        inputs.setdefault("cross_figure_precheck", "当前模式不执行图像或柱状图 cross-figure 预比对。")
        inputs.setdefault("cross_figure_precheck_json", "{}")
        inputs.setdefault("case_manifest", {})
        inputs.setdefault("case_manifest_json", "{}")
        inputs.setdefault("case_dir", "")
        inputs.setdefault("case_pdf_path", "")
        inputs.setdefault("raw_data_payload", {})
        inputs.setdefault("raw_data_files_json", "[]")
        inputs.setdefault("raw_data_datasets_json", "[]")
        inputs.setdefault("raw_data_profile_json", "{}")
        inputs.setdefault("raw_data_precheck", "未在当前模式下执行 XLSX 原始数据预检。")
        inputs.setdefault("raw_data_precheck_json", "{}")
        inputs.setdefault("deterministic_evidence_json", "[]")
        inputs.setdefault("confidence_summary_json", "{}")
        inputs.setdefault("allowed_claims_json", "[]")

        # ── Local deterministic preload/precheck ───────────────────────
        # 本地模式只做 PDF 文本/表格与 XLSX 原始数据预检，不执行图片、面板、OCR 或图像比对。
        if inputs["identifier_type"] == "local_pdf":
            pdf_path = inputs["paper_identifier"]
            self._inject_local_paper_payload(inputs, pdf_path, extract_images=False)
        elif inputs["identifier_type"] == "local_case":
            case_dir = inputs.get("case_dir") or inputs["paper_identifier"]
            self._inject_local_case_payload(inputs, case_dir)

        logger.info(
            f"Starting investigation of {inputs['paper_identifier']} "
            f"(type: {inputs['identifier_type']})"
        )
        return inputs

    @after_kickoff
    def log_completion(self, result) -> str:
        """Log investigation completion and persist the report."""
        logger.info("Investigation complete.")
        # result can be CrewOutput or str depending on CrewAI version
        try:
            raw = result.raw if hasattr(result, 'raw') else str(result)
            logger.info(f"Result length: {len(raw)} characters.")
        except Exception:
            logger.info(f"Result type: {type(result).__name__}")
        return result
