"""
Stage 2: Observation Formula Validation Data Classes

수식 검증 결과를 위한 Python dataclasses 정의
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DistributionStats:
    """구간별 분포 통계량"""
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
    """LLM 판단 결과 (Stage2 개선: PASS/FAIL only)"""
    verdict: str  # "PASS" or "FAIL" only
    reasoning: str  # rationale (PASS/FAIL 모두 필수)
    # 근거는 반드시 숫자 인용 형태로
    primary_evidence: List[Dict[str, Any]] = field(default_factory=list)
    # Per-feature analysis for MAG/DIR/VOL/POS (optional but recommended).
    feature_analysis: Dict[str, str] = field(default_factory=dict)
    raw_response: Optional[str] = None


@dataclass
class FormulaValidationResult:
    """수식 검증 결과 (Stage2 개선: 구조화된 증거 포함)"""
    formula_id: str
    formula_name: str
    obs_id: str

    # 판정 결과 (LLM 판단) - PASS/FAIL only
    verdict: str  # "PASS" or "FAIL"
    reasoning: str  # rationale (PASS/FAIL 모두 포함)

    # 구간별 샘플 수
    quantile_counts: Dict[str, int] = field(default_factory=dict)

    # 구조화된 증거 패킷 (JSON)
    evidence_packet: Dict[str, Any] = field(default_factory=dict)

    # LLM 판단에서 인용된 숫자 근거
    primary_evidence: List[Dict[str, Any]] = field(default_factory=list)

    # 분포 통계 (raw OHLCV 요소별) - 디버깅용 유지
    distribution_by_element: Dict[str, Dict[str, DistributionStats]] = field(default_factory=dict)

    # 개선 힌트 (FAIL인 경우) - 수식 재작성이 아닌 조정 힌트만
    # improvement_hints: Optional[Dict[str, Any]] = None

    # 분포 요약 텍스트 (사람용 보조 설명)
    distribution_summary: str = ""
