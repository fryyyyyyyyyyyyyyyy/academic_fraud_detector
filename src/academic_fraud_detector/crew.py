"""
Academic Fraud Detection Crew — assembly of all agents, tasks, and tools.

Uses the @CrewBase decorator pattern (CrewAI best practice).
The crew uses Hierarchical process: Lead Investigator manages all specialists,
dynamically delegating tasks and adjusting investigation depth.

Current investigation dimensions (6 total, plagiarism disabled):
1. Image Forensics (30%) — ELA, clone detection, cross-figure comparison
2. Data Integrity (30%) — Benford, p-value, GRIM, statistical consistency
3. Citation Manipulation (15%) — networkx graph analysis, self-citation
4. Peer Review Fraud (10%) — text analysis, reviewer credentials
5. Methodology Consistency (10%) — reagent verification, ethics check, timeline
6. Productivity Anomaly (5%) — publication frequency, salami slicing

Usage:
    crew = AcademicFraudDetectionCrew().crew()
    result = crew.kickoff(inputs={
        "paper_identifier": "10.1038/nature12345",
        "identifier_type": "doi",
    })
"""

import json
import os
import logging
from pathlib import Path
from typing import Optional

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

        # Image forensics tools
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

        # OCR tools (for extracting numeric data from chart images)
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
        """Image manipulation detection specialist."""
        tools = [
            self._ela,
            self._clone_detection,
            self._ai_image,
            self._cross_image_duplicate,
            self._background_consistency,
            self._feature_duplicate,
        ]
        if self._local_only:
            tools.insert(0, self._local_paper_loader)
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
            self._chart_ocr,
            self._batch_chart_ocr,
            self._bar_chart_extract,
        ]
        if self._local_only:
            tools.insert(0, self._local_paper_loader)
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

        In local_only mode: focuses on data integrity + image forensics only.
        (No plagiarism/citation/peer-review — those require external APIs.)

        Uses Hierarchical process:
        - Lead Investigator is the manager (NOT in agents list)
        - Specialist agents execute tasks
        """
        manager = self.lead_investigator()

        if self._local_only:
            # Only tasks that work with local content alone.
            tasks = [
                self.acquire_target_paper(),
                self.image_forensics_investigation(),
                self.data_integrity_investigation(),
                self.methodology_audit(),
                self.local_synthesize_findings(),
            ]
            skip_roles = {
                "Senior Plagiarism Detection Specialist",
                "Citation Manipulation Investigator",
                "Peer Review Integrity Inspector",
                "Publication Productivity Anomaly Analyst",
            }
            worker_agents = [
                a for a in self.agents
                if a.role != manager.role and a.role not in skip_roles
            ]
            logger.info(
                f"Local-only mode: {len(tasks)} tasks, "
                f"{len(worker_agents)} worker agents "
                f"(image forensics + data integrity + methodology audit)"
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

        valid_types = {"doi", "arxiv_id", "title", "url", "semantic_scholar_id", "local_pdf"}
        if inputs["identifier_type"] not in valid_types:
            raise ValueError(
                f"Invalid identifier_type '{inputs['identifier_type']}'. "
                f"Must be one of: {', '.join(sorted(valid_types))}."
            )

        # Validate local PDF file exists
        if inputs["identifier_type"] == "local_pdf":
            pdf_path = inputs["paper_identifier"]
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(
                    f"Local PDF file not found: {pdf_path}"
                )
            if not pdf_path.lower().endswith(".pdf"):
                raise ValueError(
                    f"File must be a PDF: {pdf_path}"
                )

        # Inject auto-generated values needed by task templates
        inputs.setdefault("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))

        # ── Run cross-figure precheck for local PDF mode ───────────────
        # This is a DETERMINISTIC code-level pipeline that guarantees
        # bar chart extraction + cross-figure comparison ALWAYS runs
        # before the LLM agents start, regardless of agent behavior.
        if inputs["identifier_type"] == "local_pdf":
            pdf_path = inputs["paper_identifier"]
            logger.info(f"Running cross-figure precheck on {pdf_path}...")
            try:
                from .utils.cross_figure_pipeline import (
                    run_cross_figure_pipeline,
                    format_precheck_for_agent,
                )

                precheck_result = run_cross_figure_pipeline(pdf_path)
                formatted = format_precheck_for_agent(precheck_result)

                # Inject both formatted text (for task context) and raw JSON
                # (for programmatic reference by the agent)
                inputs["cross_figure_precheck"] = formatted
                inputs["cross_figure_precheck_json"] = json.dumps(
                    precheck_result, ensure_ascii=False, default=str
                )

                n_datasets = len(precheck_result.get("datasets", []))
                n_matches = len(precheck_result.get("matches", []))
                has_critical = precheck_result.get("has_critical_match", False)

                logger.info(
                    f"Cross-figure precheck complete: {n_datasets} datasets, "
                    f"{n_matches} matches, critical={has_critical}"
                )
            except Exception as e:
                logger.warning(
                    f"Cross-figure precheck failed (investigation will proceed "
                    f"without precheck data): {e}"
                )
                inputs["cross_figure_precheck"] = (
                    "⚠️ 系统预比对未能完成（技术错误）。"
                    "请手动调用 bar_chart_extract_values 和 cross_figure_data_compare。\n"
                    f"错误信息：{e}"
                )
                inputs["cross_figure_precheck_json"] = json.dumps(
                    {"error": str(e), "datasets": [], "matches": []}
                )

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
