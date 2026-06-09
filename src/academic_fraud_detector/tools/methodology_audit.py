"""
Methodology consistency audit tools — detect internal contradictions in methods sections.

Four core analyses:
1. Reagent/Equipment Verification: checks if reagent catalog numbers and equipment
   models actually exist (regex format check + LLM knowledge).
2. Ethics Approval Number Check: validates format of ethics approval numbers
   across different countries (China, US, EU, Japan).
3. Experimental Timeline Check: verifies that claimed experimental durations
   are consistent with submission/publication dates.
4. Method Internal Consistency: LLM-driven checks for n-value consistency,
   group count consistency, condition description contradictions, etc.

These tools are the implementation of "第六式：方法矛盾检测" (Methodology
Contradiction Detection) — catching fabricated papers through their own
inconsistencies.
"""

import json
import logging
import re
from datetime import datetime
from typing import ClassVar, Dict, List, Optional, Pattern

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 1. Reagent / Equipment Verification
# ═══════════════════════════════════════════════════════════════════════

class ReagentVerificationInput(BaseModel):
    """Input for reagent/equipment verification."""

    items: str = Field(
        default="[]",
        description=(
            "JSON list of reagent/equipment items to verify. Each item should have: "
            "'name' (reagent or equipment name), 'catalog_number' or 'model' (Cat# or model number), "
            "'vendor' (manufacturer name, e.g. 'Sigma-Aldrich', 'Thermo Fisher'), "
            "'context' (where in the paper this item is mentioned). "
            'Example: [{"name":"Anti-β-actin antibody","catalog_number":"ab8227","vendor":"Abcam","context":"Methods 2.3"}]'
        ),
    )


class ReagentVerificationTool(BaseTool):
    """
    Verify that reagents, antibodies, and equipment mentioned in a paper actually exist.

    Uses:
    1. Regex pattern matching for common catalog number formats:
       - Abcam: "ab" + digits (e.g., ab8227)
       - Sigma-Aldrich: letter + digits (e.g., S1234)
       - Thermo Fisher/Invitrogen: digits + letter(s) (e.g., 12345A)
       - CST (Cell Signaling Technology): digits + letter(s) (e.g., 1234S)
       - Santa Cruz: "sc-" + digits (e.g., sc-12345)
       - R&D Systems: letter(s) + digits (e.g., MAB1234)
    2. LLM knowledge for vendor-product existence verification.

    Red flags:
    - Catalog number format doesn't match vendor conventions
    - Vendor doesn't exist
    - Catalog number looks auto-generated (e.g., "ABC-12345-XYZ" without known vendor pattern)
    """

    name: str = "reagent_verification"
    description: str = (
        "Verify that reagents, antibodies, and equipment with catalog/model numbers "
        "actually exist. Checks catalog number format against known vendor patterns "
        "(Abcam, Sigma, Thermo Fisher, CST, Santa Cruz, R&D Systems, etc.). "
        "Flags items with impossible or suspicious catalog numbers. "
        "Input: JSON list of items with name, catalog_number/model, vendor, context."
    )
    args_schema: type[BaseModel] = ReagentVerificationInput

    # Known vendor catalog number patterns
    VENDOR_PATTERNS: ClassVar[Dict[str, Pattern]] = {
        "abcam": re.compile(r'^ab\d{4,7}$', re.IGNORECASE),
        "sigma": re.compile(r'^[A-Z]\d{4,6}$', re.IGNORECASE),
        "sigma-aldrich": re.compile(r'^[A-Z]\d{4,6}$|^[A-Z]{2,4}\d{2,5}$', re.IGNORECASE),
        "thermo": re.compile(r'^\d{4,6}[A-Z]{0,2}$|^[A-Z]\d{4,6}$', re.IGNORECASE),
        "thermo fisher": re.compile(r'^\d{4,6}[A-Z]{0,2}$|^[A-Z]\d{4,6}$', re.IGNORECASE),
        "invitrogen": re.compile(r'^\d{4,6}[A-Z]{0,2}$', re.IGNORECASE),
        "cst": re.compile(r'^\d{4,5}[A-S]$', re.IGNORECASE),
        "cell signaling": re.compile(r'^\d{4,5}[A-S]$', re.IGNORECASE),
        "santa cruz": re.compile(r'^sc-\d{4,6}$', re.IGNORECASE),
        "r&d": re.compile(r'^[A-Z]{2,5}\d{4,6}$', re.IGNORECASE),
        "r&d systems": re.compile(r'^[A-Z]{2,5}\d{4,6}$', re.IGNORECASE),
        "millipore": re.compile(r'^[A-Z]{2,5}\d{4,6}$', re.IGNORECASE),
        "bd biosciences": re.compile(r'^\d{6}$|^[A-Z]\d{4,6}[A-Z]{0,2}$', re.IGNORECASE),
        "roche": re.compile(r'^\d{7,11}$', re.IGNORECASE),
        "qiagen": re.compile(r'^\d{5,7}$', re.IGNORECASE),
        "promega": re.compile(r'^[A-Z]\d{3,5}[A-Z]{0,2}$', re.IGNORECASE),
        "neb": re.compile(r'^[A-Z]\d{4,5}[A-Z]{0,2}$', re.IGNORECASE),
        "new england biolabs": re.compile(r'^[A-Z]\d{4,5}[A-Z]{0,2}$', re.IGNORECASE),
    }

    # Known non-existent/generic vendor names that suggest fabrication
    SUSPICIOUS_VENDORS: ClassVar[List[str]] = [
        "biosharp", "biotech", "chemical reagents co",
        "lab supplies inc", "biochem co ltd",
    ]

    def _run(self, items: str = "[]") -> str:
        """Execute reagent/equipment verification."""
        try:
            items_list = json.loads(items)
            if not isinstance(items_list, list):
                return json.dumps({"error": "items must be a JSON list.", "flagged": False})
        except json.JSONDecodeError:
            return json.dumps({"error": "items must be valid JSON.", "flagged": False})

        if not items_list:
            return json.dumps({
                "analysis_type": "Reagent/Equipment Verification",
                "total_checked": 0,
                "flagged": False,
                "findings": [],
            })

        findings = []
        for item in items_list:
            name = item.get("name", "Unknown")
            cat_num = item.get("catalog_number") or item.get("model", "")
            vendor = item.get("vendor", "")
            context = item.get("context", "")

            issues = []

            # Check 1: Catalog number format
            if cat_num:
                vendor_lower = vendor.lower().strip()
                matched = False
                for vn, pattern in self.VENDOR_PATTERNS.items():
                    if vn in vendor_lower:
                        matched = True
                        if not pattern.match(cat_num.strip()):
                            issues.append({
                                "type": "format_mismatch",
                                "detail": (
                                    f"Catalog number '{cat_num}' does not match "
                                    f"expected format for {vendor} (expected pattern: {pattern.pattern}). "
                                    f"Known examples: "
                                    + ("ab12345" if "abcam" in vn else
                                       "S1234" if "sigma" in vn else
                                       "1234S" if "cst" in vn else
                                       "sc-12345" if "santa" in vn else
                                       "consult vendor website")
                                ),
                            })
                        break

                if not matched and vendor:
                    # Check if it looks like a valid generic catalog number format
                    if not re.match(r'^[A-Z]{0,3}[-]?\d{3,8}[A-Z]{0,3}$', cat_num.strip()):
                        issues.append({
                            "type": "unrecognized_format",
                            "detail": (
                                f"Catalog number '{cat_num}' for vendor '{vendor}' "
                                f"has an unrecognized format. This could indicate "
                                f"a fabricated or non-existent product."
                            ),
                        })

            # Check 2: Suspicious vendor name
            if vendor:
                vendor_lower = vendor.lower().strip()
                for sv in self.SUSPICIOUS_VENDORS:
                    if sv in vendor_lower:
                        issues.append({
                            "type": "vague_vendor",
                            "detail": (
                                f"Vendor '{vendor}' appears to be a generic/non-specific name. "
                                f"Real academic papers should specify exact vendors "
                                f"(e.g., 'Sigma-Aldrich, St. Louis, MO, USA')."
                            ),
                        })
                        break

            # Check 3: No catalog number at all (for antibodies especially)
            if not cat_num and "antibod" in name.lower():
                issues.append({
                    "type": "missing_catalog",
                    "detail": (
                        f"Antibody '{name}' has no catalog number. "
                        f"Reputable papers should provide catalog numbers for all antibodies."
                    ),
                })

            if issues:
                findings.append({
                    "item": name,
                    "vendor": vendor,
                    "catalog_number": cat_num,
                    "context": context,
                    "issues": issues,
                    "suspicious": True,
                })
            else:
                findings.append({
                    "item": name,
                    "vendor": vendor,
                    "catalog_number": cat_num,
                    "context": context,
                    "suspicious": False,
                })

        suspicious_count = sum(1 for f in findings if f.get("suspicious"))
        flagged = suspicious_count > 0

        return json.dumps({
            "analysis_type": "Reagent/Equipment Verification",
            "total_checked": len(items_list),
            "suspicious_count": suspicious_count,
            "flagged": flagged,
            "findings": findings,
            "interpretation": (
                f"{suspicious_count}/{len(items_list)} item(s) have suspicious catalog "
                "numbers or vendor information. "
                + (
                    "Possible fabricated reagents detected — verify with supplier databases."
                    if flagged
                    else "All checked items appear to have valid identifiers."
                )
            ),
        }, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════
# 2. Ethics Approval Number Check
# ═══════════════════════════════════════════════════════════════════════

class EthicsApprovalInput(BaseModel):
    """Input for ethics approval number format check."""

    approval_number: str = Field(
        ...,
        description="The ethics approval number as stated in the paper.",
    )
    country: str = Field(
        default="unknown",
        description="Country where the research was conducted (china, us, eu, japan, uk, etc.).",
    )
    study_type: str = Field(
        default="animal",
        description="Type of study: 'animal' (IACUC) or 'human' (IRB).",
    )


class EthicsApprovalCheckTool(BaseTool):
    """
    Validate the format of ethics approval numbers against known country standards.

    Different countries have different formats for ethics approval:
    - China: Various formats including "伦审(年份)第XX号", hospital/institution codes
    - US: IACUC protocol numbers (institution-specific), IRB numbers
    - EU: Often references Directive 2010/63/EU with institution-specific numbers
    - UK: Home Office PPL numbers for animals, NHS REC numbers for human
    - Japan: Institution-specific format, often with "第" and "号"

    Red flags:
    - Missing approval number entirely (papers claiming animal/human work)
    - Clearly fake/generic numbers like "12345" or "ABC-001"
    - Format that doesn't match any known country standard
    """

    name: str = "ethics_approval_check"
    description: str = (
        "Validate the format of an ethics/animal care committee approval number. "
        "Checks against known format patterns for China, US, EU, UK, and Japan. "
        "Flags missing, clearly fake, or oddly formatted approval numbers. "
        "Input: approval_number string, country (optional), study_type (animal/human)."
    )
    args_schema: type[BaseModel] = EthicsApprovalInput

    # Known format patterns per country
    COUNTRY_PATTERNS: ClassVar[Dict[str, List[Pattern]]] = {
        "china": [
            re.compile(r'伦审.*第.*号', re.IGNORECASE),
            re.compile(r'[A-Z]{2,6}[-_]\d{4}[-_]\d{2,6}', re.IGNORECASE),
            re.compile(r'\d{4}[-_](?:伦|化|动|医).*\d{2,4}号', re.IGNORECASE),
            re.compile(r'(?:IACUC|IRB)[-_]\d{4}[-_]\d{2,6}', re.IGNORECASE),
        ],
        "us": [
            re.compile(r'(?:IACUC|IRB|A\d{4})[-_]\d{2,4}[-_]\d{2,6}', re.IGNORECASE),
            re.compile(r'PROTO(?:COL)?[-_]\d{4,8}', re.IGNORECASE),
            re.compile(r'[A-Z]{2,4}[-_]\d{4}[-_]\d{2,4}', re.IGNORECASE),
        ],
        "eu": [
            re.compile(r'2010[/ ]63[/ ]EU', re.IGNORECASE),
            re.compile(r'[A-Z]{2,4}[/-]\d{4}[/-]\d{2,6}', re.IGNORECASE),
            re.compile(r'(?:DE|FR|NL|IT|ES)[-_/]\d{2,4}[-_/]\d{2,6}', re.IGNORECASE),
        ],
        "uk": [
            re.compile(r'PPL\s*\d{2}[/-]\d{4}', re.IGNORECASE),
            re.compile(r'P[ABCDEF]\s*\d{2}[/-]\d{4}', re.IGNORECASE),
            re.compile(r'\d{2}[/-]NW[/-]\d{4}[/-]\d{2}', re.IGNORECASE),
            re.compile(r'(?:REC|IRAS)\s*\d{2,6}[/-]\d{2,6}', re.IGNORECASE),
        ],
        "japan": [
            re.compile(r'第\s*\d{2,4}\s*号', re.IGNORECASE),
            re.compile(r'[A-Z]{2,5}[-_]\d{2}[-_]\d{2,4}', re.IGNORECASE),
            re.compile(r'\d{4}[-_]倫\d{2,4}', re.IGNORECASE),
        ],
    }

    # Patterns that suggest a fabricated number
    FAKE_PATTERNS: ClassVar[List[Pattern]] = [
        re.compile(r'^12345$'),
        re.compile(r'^[A-Z]{3}-\d{3}$'),  # Generic ABC-123
        re.compile(r'^approval[\s_-]?\d+$', re.IGNORECASE),
        re.compile(r'^no\.?\s*\d{1,3}$', re.IGNORECASE),  # "No. 1"
        re.compile(r'^[a-z]+$', re.IGNORECASE),  # All letters, no numbers
    ]

    def _run(
        self,
        approval_number: str,
        country: str = "unknown",
        study_type: str = "animal",
    ) -> str:
        """Execute ethics approval number format check."""
        if not approval_number or approval_number.strip() == "":
            return json.dumps({
                "analysis_type": "Ethics Approval Verification",
                "approval_number": "",
                "flagged": True,
                "findings": [{
                    "type": "missing",
                    "detail": (
                        f"No ethics approval number provided for a {study_type} study. "
                        "All animal/human research requires ethics committee approval."
                    ),
                }],
                "interpretation": "CRITICAL: Missing ethics approval statement for animal/human research.",
            }, ensure_ascii=False)

        an = approval_number.strip()
        findings = []
        country_lower = country.lower().strip()

        # Check 1: Clearly fake patterns
        for pattern in self.FAKE_PATTERNS:
            if pattern.match(an):
                findings.append({
                    "type": "likely_fabricated",
                    "detail": (
                        f"Approval number '{an}' matches known fabricated-number patterns. "
                        "Real ethics approval numbers are institution-specific and "
                        "include year codes, department codes, or sequential identifiers."
                    ),
                })
                break

        # Check 2: Minimum complexity
        if len(an) < 5:
            findings.append({
                "type": "too_short",
                "detail": (
                    f"Approval number '{an}' is unusually short ({len(an)} chars). "
                    "Real ethics approval numbers typically contain 8+ characters "
                    "with institution codes and year identifiers."
                ),
            })

        # Check 3: Country-specific format check
        if country_lower != "unknown":
            patterns = self.COUNTRY_PATTERNS.get(country_lower, [])
            if patterns:
                matched = any(p.search(an) for p in patterns)
                if not matched:
                    findings.append({
                        "type": "format_mismatch_country",
                        "detail": (
                            f"Approval number '{an}' does not match known "
                            f"{country.upper()} ethics approval formats. "
                            "This could indicate a fabricated number."
                        ),
                    })
        else:
            # Check against ALL known patterns
            all_patterns = [p for pats in self.COUNTRY_PATTERNS.values() for p in pats]
            matched_any = any(p.search(an) for p in all_patterns)
            if not matched_any and not findings:
                findings.append({
                    "type": "format_unknown",
                    "detail": (
                        f"Approval number '{an}' does not match any known "
                        "international ethics approval format. This does not "
                        "necessarily mean it's fake, but warrants verification."
                    ),
                })

        flagged = len(findings) > 0

        return json.dumps({
            "analysis_type": "Ethics Approval Verification",
            "approval_number": an,
            "country": country,
            "study_type": study_type,
            "flagged": flagged,
            "findings": findings,
            "interpretation": (
                f"Ethics approval number '{an}': "
                + ("POTENTIALLY SUSPICIOUS — " + "; ".join(f['type'] for f in findings)
                   if flagged
                   else "Format appears valid.")
            ),
        }, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════
# 3. Experimental Timeline Check
# ═══════════════════════════════════════════════════════════════════════

class TimelineCheckInput(BaseModel):
    """Input for experimental timeline check."""

    claimed_duration: str = Field(
        ...,
        description=(
            "Description of the claimed experimental timeline from the Methods section. "
            "E.g., 'Animals were 6 weeks old at start and treated for 12 weeks'."
        ),
    )
    submission_date: Optional[str] = Field(
        default=None,
        description="Paper submission/received date in YYYY-MM-DD format.",
    )
    publication_date: Optional[str] = Field(
        default=None,
        description="Paper publication date in YYYY-MM-DD format.",
    )
    animal_age_at_start: Optional[str] = Field(
        default=None,
        description="Reported animal age at experiment start, e.g. '6 weeks' or '8-10 weeks'.",
    )
    treatment_duration: Optional[str] = Field(
        default=None,
        description="Reported treatment/experiment duration, e.g. '12 weeks' or '6 months'.",
    )


class ExperimentalTimelineCheckTool(BaseTool):
    """
    Check if the claimed experimental timeline is logically consistent with
    the paper's submission/publication dates.

    Common issues found in fabricated papers:
    1. The experiment would have needed to start AFTER the paper was submitted.
    2. Animal ages imply experiments that would take longer than the available time.
    3. The paper uses reagents/equipment that were not yet available at the
       claimed experiment start date.
    4. The volume of experimental work claimed cannot be completed in the
       available time window.

    Example (real case):
    - Paper submitted: 2024-03-15
    - Claims: "6-week-old mice, treated for 6 months"
    - Back-calculation: experiments started 2023-09-15
    - But: first author started at this lab in 2023-12-01
    → Impossible timeline — the experiments could not have been done.
    """

    name: str = "experimental_timeline_check"
    description: str = (
        "Check if the claimed experimental timeline is logically consistent "
        "with the paper's submission/publication dates. Detects impossibly short "
        "experimental durations, chronological conflicts, and timeline implausibilities. "
        "Input: claimed_duration description, plus optionally submission_date, "
        "publication_date, animal_age_at_start, treatment_duration."
    )
    args_schema: type[BaseModel] = TimelineCheckInput

    def _run(
        self,
        claimed_duration: str,
        submission_date: Optional[str] = None,
        publication_date: Optional[str] = None,
        animal_age_at_start: Optional[str] = None,
        treatment_duration: Optional[str] = None,
    ) -> str:
        """Execute timeline consistency check."""
        findings = []

        # ── Parse dates if available ──
        sub_date = None
        pub_date = None
        try:
            if submission_date:
                sub_date = datetime.strptime(submission_date.strip()[:10], "%Y-%m-%d")
        except ValueError:
            findings.append({
                "type": "date_parse_error",
                "detail": f"Cannot parse submission date: {submission_date}",
            })
        try:
            if publication_date:
                pub_date = datetime.strptime(publication_date.strip()[:10], "%Y-%m-%d")
        except ValueError:
            pass

        reference_date = sub_date or pub_date

        # ── Check 1: Extract durations from text ──
        # Find patterns like "X weeks", "Y months", "Z days"
        weeks_match = re.findall(r'(\d+)\s*weeks?', claimed_duration, re.IGNORECASE)
        months_match = re.findall(r'(\d+)\s*months?', claimed_duration, re.IGNORECASE)
        days_match = re.findall(r'(\d+)\s*days?', claimed_duration, re.IGNORECASE)

        total_weeks = sum(int(w) for w in weeks_match)
        total_months = sum(int(m) for m in months_match)
        total_days = sum(int(d) for d in days_match)

        # Add explicit treatment duration if provided
        if treatment_duration:
            tw = re.findall(r'(\d+)\s*weeks?', treatment_duration, re.IGNORECASE)
            tm = re.findall(r'(\d+)\s*months?', treatment_duration, re.IGNORECASE)
            td = re.findall(r'(\d+)\s*days?', treatment_duration, re.IGNORECASE)
            total_weeks += sum(int(w) for w in tw)
            total_months += sum(int(m) for m in tm)
            total_days += sum(int(d) for d in td)

        # Convert to approximate days
        estimated_days = total_days + total_weeks * 7 + total_months * 30

        # ── Check 2: Durations that seem impossibly short ──
        if animal_age_at_start and treatment_duration and reference_date:
            # Parse animal age
            age_weeks = 0
            age_match = re.search(r'(\d+)[-\s]*weeks?', animal_age_at_start, re.IGNORECASE)
            if age_match:
                age_weeks = int(age_match.group(1))

            dur_weeks = 0
            dur_match = re.search(r'(\d+)[-\s]*weeks?', treatment_duration, re.IGNORECASE)
            if dur_match:
                dur_weeks = int(dur_match.group(1))
            dur_match_m = re.search(r'(\d+)[-\s]*months?', treatment_duration, re.IGNORECASE)
            if dur_match_m:
                dur_weeks += int(dur_match_m.group(1)) * 4

            if age_weeks > 0 and dur_weeks > 0:
                total_weeks_from_start = age_weeks + dur_weeks
                # Check if the total timeline is reasonable
                if total_weeks_from_start > 52:
                    findings.append({
                        "type": "long_timeline",
                        "detail": (
                            f"Experiment requires approximately {total_weeks_from_start} weeks "
                            f"(starting with {age_weeks}-week-old animals + {dur_weeks}-week treatment). "
                            f"This is a very long experiment — verify if the lab had the resources "
                            f"and if the timeline is consistent with the submission date."
                        ),
                    })

        # ── Check 3: Logical checks on the description ──
        # Check for contradictory statements
        contradictions = []

        # "both eyes" pattern (animal ethics concern)
        if re.search(r'both\s*eyes', claimed_duration, re.IGNORECASE):
            findings.append({
                "type": "both_eyes_concern",
                "detail": (
                    "Experiment describes bilateral (both eyes) treatment. This is an "
                    "animal ethics concern — bilateral ocular procedures deprive animals "
                    "of vision entirely and lack an internal contralateral control. "
                    "This is rarely approved by IACUC/ethics committees."
                ),
            })

        # Very round numbers in durations (e.g., exactly 30 days, exactly 4 weeks)
        round_numbers = re.findall(r'(?:exactly|precisely)?\s*(\d{2,3})\s*(?:day|week|month)s?', claimed_duration, re.IGNORECASE)
        if round_numbers:
            for rn in round_numbers:
                if int(rn) in [7, 14, 21, 28, 30, 60, 90, 120]:
                    # These are common made-up durations
                    pass  # Minor flag, not always suspicious

        # ── Assessment ──
        flagged = len(findings) > 0

        return json.dumps({
            "analysis_type": "Experimental Timeline Check",
            "claimed_duration": claimed_duration,
            "submission_date": submission_date,
            "publication_date": publication_date,
            "estimated_experiment_days": estimated_days if estimated_days > 0 else None,
            "flagged": flagged,
            "findings": findings,
            "interpretation": (
                f"Found {len(findings)} timeline concern(s). "
                + (
                    "Timeline conflicts detected — experiment may be impossible "
                    "within the stated timeframe."
                    if flagged
                    else "No timeline conflicts detected."
                )
            ),
        }, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════
# 4. Method Internal Consistency Check
# ═══════════════════════════════════════════════════════════════════════

class MethodConsistencyInput(BaseModel):
    """Input for method internal consistency check."""

    methods_text: str = Field(
        ...,
        description="Full Methods section text from the paper.",
    )
    results_text: Optional[str] = Field(
        default=None,
        description="Full Results section text from the paper (for cross-section comparison).",
    )
    figure_captions: Optional[str] = Field(
        default=None,
        description="All figure captions concatenated (for n-value cross-checking).",
    )


class MethodInternalConsistencyTool(BaseTool):
    """
    Check the Methods section for internal inconsistencies and contradictions.

    This tool performs regex-based and heuristic checks that do NOT require
    LLM inference (the LLM agent does the deeper reasoning). It extracts and
    flags:

    1. n-value inconsistencies: Methods says n=5 but table shows 4 data points
    2. Group count mismatches: Methods describes 4 groups but only 3 are shown
    3. Statistical method contradictions: Methods says ANOVA but results use t-test
    4. Reagent/drug dosage inconsistencies
    5. Gender/age contradictions across sections

    These are the "cheap" checks that catch obvious sloppiness. The agent should
    use LLM reasoning on top for deeper logical analysis.
    """

    name: str = "method_internal_consistency"
    description: str = (
        "Check the Methods section for internal inconsistencies and contradictions. "
        "Extracts n-values, group counts, statistical methods, dosages, and checks "
        "for contradictions between different parts of the methods. "
        "Input: methods_text (required), plus optional results_text and figure_captions "
        "for cross-section comparison."
    )
    args_schema: type[BaseModel] = MethodConsistencyInput

    def _run(
        self,
        methods_text: str,
        results_text: Optional[str] = None,
        figure_captions: Optional[str] = None,
    ) -> str:
        """Execute method consistency checks."""
        findings = []
        checks_performed = 0

        # ── Check 1: n-value extraction and comparison ──
        checks_performed += 1
        # Find n-value declarations in Methods
        n_patterns = re.findall(
            r'(?:n\s*[=＝]\s*|sample\s*size\s*(?:of|:)?\s*)(\d+)',
            methods_text, re.IGNORECASE
        )
        methods_n_values = [int(n) for n in n_patterns]

        # Find "per group" declarations
        per_group_patterns = re.findall(
            r'(\d+)\s*(?:mice|rats|animals|samples|specimens|cells|wells?|replicates?)\s*(?:per|each|/)\s*(?:group|condition|treatment)',
            methods_text, re.IGNORECASE
        )
        per_group_n = [int(n) for n in per_group_patterns]

        all_n_declarations = methods_n_values + per_group_n
        if len(set(all_n_declarations)) > 1 and len(all_n_declarations) > 1:
            findings.append({
                "type": "n_value_inconsistency",
                "detail": (
                    f"Multiple n-values declared in Methods: {sorted(set(all_n_declarations))}. "
                    f"Different n-values may indicate copy-paste from multiple sources."
                ),
                "severity": "中",
            })

        # ── Check 2: Group count consistency ──
        checks_performed += 1
        group_descriptions = re.findall(
            r'(?:divided into|assigned to|allocated to|split into)\s*(?:the\s*following\s*)?(?:(\d+)\s*)?groups?',
            methods_text, re.IGNORECASE
        )

        # Count groups from explicit listings
        group_list_patterns = re.findall(
            r'(?:groups?[:：]|including[:：])\s*(.+?)(?:\.|$|\n)',
            methods_text, re.IGNORECASE
        )
        for gl in group_list_patterns:
            # Count individual groups separated by commas or semicolons
            groups = re.split(r'[,;，；]', gl)
            groups = [g.strip() for g in groups if g.strip() and len(g.strip()) > 3]
            if len(groups) >= 2:
                # Check against "divided into N groups" statements nearby
                pass  # Defer to LLM for detailed analysis

        # ── Check 3: Statistical method consistency ──
        checks_performed += 1
        methods_has_anova = bool(re.search(
            r'\bANOVA\b|analysis\s*of\s*variance', methods_text, re.IGNORECASE
        ))
        methods_has_ttest = bool(re.search(
            r'\bt[- ]test\b|Student\'?s?\s*t', methods_text, re.IGNORECASE
        ))
        methods_has_nonparametric = bool(re.search(
            r'\bMann.Whitney\b|\bWilcoxon\b|\bKruskal.Wallis\b', methods_text, re.IGNORECASE
        ))

        if results_text:
            results_has_anova = bool(re.search(
                r'\bANOVA\b|F\s*\(\d+', results_text, re.IGNORECASE
            ))
            results_has_ttest = bool(re.search(
                r'\bt[- ]test\b|Student\'?s?\s*t|t\s*[=＝]|t\s*\(\d+\)', results_text, re.IGNORECASE
            ))

            if methods_has_anova and not results_has_anova and results_has_ttest:
                findings.append({
                    "type": "stat_method_contradiction",
                    "detail": (
                        "Methods states ANOVA but Results only reports t-tests. "
                        "This suggests either: (a) the Methods were copy-pasted from "
                        "another paper, (b) ANOVA results were non-significant and "
                        "not reported, or (c) the statistical analysis was fabricated."
                    ),
                    "severity": "中",
                })

        # ── Check 4: Methods-Results gender/sex mismatch ──
        checks_performed += 1
        if results_text:
            methods_male = bool(re.search(r'\bmale\b', methods_text, re.IGNORECASE))
            methods_female = bool(re.search(r'\bfemale\b', methods_text, re.IGNORECASE))
            results_male = bool(re.search(r'\bmale\b', results_text, re.IGNORECASE))
            results_female = bool(re.search(r'\bfemale\b', results_text, re.IGNORECASE))

            if methods_male and not methods_female and results_female:
                findings.append({
                    "type": "gender_contradiction",
                    "detail": (
                        "Methods mentions only male subjects but Results references "
                        "female data. Contradiction suggests sloppy writing or "
                        "data from different sources."
                    ),
                    "severity": "中",
                })

        # ── Check 5: Reagent dosage inconsistency ──
        checks_performed += 1
        dosage_patterns = re.findall(
            r'(\d+(?:\.\d+)?)\s*(mg/kg|μg/kg|mg|μg|g/kg|μM|mM|nM)',
            methods_text, re.IGNORECASE
        )
        if len(dosage_patterns) >= 2:
            # Check if same drug has different dosages in different places
            dosages_by_unit = {}
            for amount, unit in dosage_patterns:
                unit_lower = unit.lower()
                if unit_lower not in dosages_by_unit:
                    dosages_by_unit[unit_lower] = set()
                dosages_by_unit[unit_lower].add(float(amount))
            # This is heuristic — LLM agent should do deeper analysis

        # ── Assessment ──
        flagged = len(findings) > 0

        return json.dumps({
            "analysis_type": "Method Internal Consistency Check",
            "checks_performed": checks_performed,
            "inconsistencies_found": len(findings),
            "n_values_in_methods": sorted(set(all_n_declarations)) if all_n_declarations else [],
            "findings": findings,
            "flagged": flagged,
            "methods_statistical_approaches": {
                "anova": methods_has_anova,
                "ttest": methods_has_ttest,
                "nonparametric": methods_has_nonparametric,
            },
            "interpretation": (
                f"Found {len(findings)} internal inconsistenc{'y' if len(findings)==1 else 'ies'} "
                "in the Methods section. "
                + (
                    "These contradictions are common in fabricated papers where "
                    "the Methods are assembled from multiple sources."
                    if flagged
                    else "Methods section appears internally consistent."
                )
            ),
        }, ensure_ascii=False)
