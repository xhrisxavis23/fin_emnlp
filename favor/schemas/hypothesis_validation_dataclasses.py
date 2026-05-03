"""
Stage 3: Hypothesis Instance Validation Data Classes

Python dataclasses for hypothesis validation results.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class QuadrantStats:
    """Quadrant analysis stats (S1–S4)."""
    s1_true_positive: int = 0   # signal=1, return>0 (hypothesis correct)
    s2_false_positive: int = 0  # signal=1, return<=0 (false alarm)
    s3_true_negative: int = 0   # signal=0, return<=0 (avoided loss)
    s4_false_negative: int = 0  # signal=0, return>0 (missed opportunity)

    @property
    def precision(self) -> float:
        """S1/(S1+S2): entry precision."""
        total = self.s1_true_positive + self.s2_false_positive
        return self.s1_true_positive / total if total > 0 else 0.0

    @property
    def recall(self) -> float:
        """S1/(S1+S4): opportunity recall."""
        total = self.s1_true_positive + self.s4_false_negative
        return self.s1_true_positive / total if total > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """2 * precision * recall / (precision + recall)"""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        """(S1+S3) / Total"""
        total = self.s1_true_positive + self.s2_false_positive + self.s3_true_negative + self.s4_false_negative
        return (self.s1_true_positive + self.s3_true_negative) / total if total > 0 else 0.0


@dataclass
class StrictnessLevelResult:
    """Evaluation result for a single strictness level."""
    strictness_level: str  # e.g., "loose", "medium", "strict"
    strictness_value: float  # threshold (quantile)

    # Signal characteristics
    signal_count: int = 0
    signal_frequency: float = 0.0

    # Return statistics
    mean_return: float = 0.0
    median_return: float = 0.0
    std_return: float = 0.0
    sharpe_ratio: float = 0.0

    # Win/loss distribution
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0

    # Quadrant analysis
    quadrant_stats: Optional[QuadrantStats] = None


@dataclass
class MonotonicityVerification:
    """Monotonicity verification result (adjacent-level comparison)."""
    metric_name: str
    values: List[float]  # per-strictness values
    is_monotonic: bool
    direction: str  # "improving" or "degrading" or "non_monotonic"
    spearman_corr: float  # monotonicity score (improving_steps / total_steps)
    spearman_pvalue: float  # unused (always 0.0)


@dataclass
class ICMetrics:
    """
    IC (Information Coefficient) metrics.

    NOTE:
    - OOS IC is not used for the validation verdict; it is stored for logging/reference only.
    - The validation verdict uses only IS monotonicity verification results.
    """
    # In-sample IC (logging only; not used for the verdict)
    is_ic_mean: float = 0.0       # IC mean (mean of daily correlations)
    is_ic_std: float = 0.0        # IC standard deviation
    is_icir: float = 0.0          # ICIR = IC_mean / IC_std
    is_ic_positive_ratio: float = 0.0  # fraction of days with IC > 0
    is_n_days: int = 0            # number of days used

    # Out-of-sample IC (logging only; never used for the verdict)
    oos_ic_mean: float = 0.0
    oos_ic_std: float = 0.0
    oos_icir: float = 0.0
    oos_ic_positive_ratio: float = 0.0
    oos_n_days: int = 0


@dataclass
class HypothesisValidationResult:
    """Final hypothesis validation result."""
    hypothesis_id: str

    # Verdict (uses IS monotonicity verification results only)
    overall_verdict: str  # "PASS", "FAIL"
    confidence: float

    # Per-strictness results
    strictness_results: List[StrictnessLevelResult] = field(default_factory=list)

    # Monotonicity verification
    monotonicity_results: List[MonotonicityVerification] = field(default_factory=list)

    # IC/ICIR metrics (AND combo; logging only; not used for the verdict)
    ic_metrics: Optional[ICMetrics] = None

    # Passed formula combinations (forwarded to Stage 4)
    # Each combination is a list of formulas: [[formula1, formula2], [formula3, formula4], ...]
    passed_combinations: List[List[Dict[str, Any]]] = field(default_factory=list)

    # Evaluation results for all combinations (Stage 4 can pick the best)
    # Each result: {"combination": [...], "verdict": "PASS/FAIL", "confidence": float, ...}
    all_combination_results: List[Dict[str, Any]] = field(default_factory=list)

    # Key findings
    key_findings: List[str] = field(default_factory=list)

    # Conclusion
    conclusion: str = ""
