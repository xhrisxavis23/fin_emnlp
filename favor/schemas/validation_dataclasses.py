"""
Stage 2: Observation Formula Validation Data Classes

Python dataclasses for formula validation results.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DistributionStats:
    """Distribution statistics per bin."""
    mean: float
    median: float
    std: float
    iqr: float  # Inter-Quartile Range
    skewness: float
    kurtosis: float
    q10: float
    q25: float
    q75: float
    q90: float
    count: int


@dataclass
class LLMJudgment:
    """LLM judgment result (Stage 2: PASS/FAIL only)."""
    verdict: str  # "PASS" or "FAIL" only
    reasoning: str  # rationale (required for both PASS and FAIL)
    # Evidence must be provided as numeric citations
    primary_evidence: List[Dict[str, Any]] = field(default_factory=list)
    raw_response: Optional[str] = None


@dataclass
class FormulaValidationResult:
    """Formula validation result (Stage 2: includes structured evidence)."""
    formula_id: str
    formula_name: str
    obs_id: str

    # Verdict (LLM judgment) - PASS/FAIL only
    verdict: str  # "PASS" or "FAIL"
    reasoning: str  # rationale (required for both PASS and FAIL)

    # Sample counts per quantile/bin
    quantile_counts: Dict[str, int] = field(default_factory=dict)

    # Structured evidence packet (JSON)
    evidence_packet: Dict[str, Any] = field(default_factory=dict)

    # Numeric evidence cited by the LLM
    primary_evidence: List[Dict[str, Any]] = field(default_factory=list)

    # Distribution stats per raw OHLCV element (kept for debugging)
    distribution_by_element: Dict[str, Dict[str, DistributionStats]] = field(default_factory=dict)

    # Improvement hints (for FAIL only) - adjustment hints, not full rewrites
    # improvement_hints: Optional[Dict[str, Any]] = None

    # Human-readable distribution summary
    distribution_summary: str = ""
