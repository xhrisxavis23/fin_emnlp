# util/llm_tracker.py
"""
LLM 토큰 사용량 및 비용 추적 모듈

모든 LLM 호출의 토큰 사용량을 추적하고 비용을 계산합니다.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


# OpenAI 모델별 가격 (2024년 기준, USD per 1M tokens)
PRICING = {
    "gpt-4o": {
        "input": 2.50 / 1_000_000,
        "output": 10.00 / 1_000_000,
    },
    "gpt-4o-mini": {
        "input": 0.150 / 1_000_000,
        "output": 0.600 / 1_000_000,
    },
    "gpt-4o-2024-11-20": {
        "input": 2.50 / 1_000_000,
        "output": 10.00 / 1_000_000,
    },
    "gpt-4o-mini-2024-07-18": {
        "input": 0.150 / 1_000_000,
        "output": 0.600 / 1_000_000,
    },
    "gpt-4-turbo": {
        "input": 10.00 / 1_000_000,
        "output": 30.00 / 1_000_000,
    },
    "gpt-4": {
        "input": 30.00 / 1_000_000,
        "output": 60.00 / 1_000_000,
    },
    "gpt-3.5-turbo": {
        "input": 0.50 / 1_000_000,
        "output": 1.50 / 1_000_000,
    },
}


@dataclass
class LLMCallRecord:
    """단일 LLM 호출 기록"""
    timestamp: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    context: Optional[str] = None  # 호출 컨텍스트 (예: "stage1", "stage2_validation")


@dataclass
class LLMUsageStats:
    """LLM 사용량 통계"""
    total_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    by_model: Dict[str, Dict[str, int | float]] = field(default_factory=dict)
    by_context: Dict[str, Dict[str, int | float]] = field(default_factory=dict)
    calls: List[LLMCallRecord] = field(default_factory=list)


class LLMUsageTracker:
    """
    LLM 토큰 사용량 추적기 (Thread-safe Singleton)

    모든 LLM 호출의 토큰 수와 비용을 추적합니다.
    """

    _instance: Optional[LLMUsageTracker] = None
    _lock = threading.Lock()

    def __init__(self):
        self.stats = LLMUsageStats()
        self._call_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> LLMUsageTracker:
        """Singleton 인스턴스 반환"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """통계 초기화 (주로 테스트용)"""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.stats = LLMUsageStats()

    def track_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        context: Optional[str] = None,
    ):
        """
        LLM 호출 기록

        Args:
            model: 모델명 (예: "gpt-4o-mini")
            prompt_tokens: 입력 토큰 수
            completion_tokens: 출력 토큰 수
            context: 호출 컨텍스트 (예: "stage1", "stage2_validation")
        """
        with self._call_lock:
            total_tokens = prompt_tokens + completion_tokens
            cost = self._calculate_cost(model, prompt_tokens, completion_tokens)

            # 기록 생성
            record = LLMCallRecord(
                timestamp=datetime.now().isoformat(),
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=cost,
                context=context,
            )

            # 전체 통계 업데이트
            self.stats.total_calls += 1
            self.stats.total_prompt_tokens += prompt_tokens
            self.stats.total_completion_tokens += completion_tokens
            self.stats.total_tokens += total_tokens
            self.stats.total_cost_usd += cost
            self.stats.calls.append(record)

            # 모델별 통계
            if model not in self.stats.by_model:
                self.stats.by_model[model] = {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                }
            self.stats.by_model[model]["calls"] += 1
            self.stats.by_model[model]["prompt_tokens"] += prompt_tokens
            self.stats.by_model[model]["completion_tokens"] += completion_tokens
            self.stats.by_model[model]["total_tokens"] += total_tokens
            self.stats.by_model[model]["cost_usd"] += cost

            # 컨텍스트별 통계
            if context:
                if context not in self.stats.by_context:
                    self.stats.by_context[context] = {
                        "calls": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "cost_usd": 0.0,
                    }
                self.stats.by_context[context]["calls"] += 1
                self.stats.by_context[context]["prompt_tokens"] += prompt_tokens
                self.stats.by_context[context]["completion_tokens"] += completion_tokens
                self.stats.by_context[context]["total_tokens"] += total_tokens
                self.stats.by_context[context]["cost_usd"] += cost

    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """토큰 수 기반 비용 계산"""
        if model not in PRICING:
            # 알 수 없는 모델이면 gpt-4o-mini 가격 사용
            pricing = PRICING["gpt-4o-mini"]
        else:
            pricing = PRICING[model]

        cost = (prompt_tokens * pricing["input"]) + (completion_tokens * pricing["output"])
        return cost

    def get_summary(self) -> Dict:
        """사용량 요약 반환"""
        return {
            "total_calls": self.stats.total_calls,
            "total_tokens": self.stats.total_tokens,
            "total_prompt_tokens": self.stats.total_prompt_tokens,
            "total_completion_tokens": self.stats.total_completion_tokens,
            "total_cost_usd": round(self.stats.total_cost_usd, 4),
            "by_model": {
                model: {
                    "calls": int(data["calls"]),
                    "total_tokens": int(data["total_tokens"]),
                    "prompt_tokens": int(data["prompt_tokens"]),
                    "completion_tokens": int(data["completion_tokens"]),
                    "cost_usd": round(data["cost_usd"], 4),
                }
                for model, data in self.stats.by_model.items()
            },
            "by_context": {
                ctx: {
                    "calls": int(data["calls"]),
                    "total_tokens": int(data["total_tokens"]),
                    "prompt_tokens": int(data["prompt_tokens"]),
                    "completion_tokens": int(data["completion_tokens"]),
                    "cost_usd": round(data["cost_usd"], 4),
                }
                for ctx, data in self.stats.by_context.items()
            },
        }

    def get_detailed_records(self, limit: int = 100) -> List[Dict]:
        """상세 호출 기록 반환 (최근 N개)"""
        records = self.stats.calls[-limit:]
        return [
            {
                "timestamp": r.timestamp,
                "model": r.model,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "total_tokens": r.total_tokens,
                "cost_usd": round(r.cost_usd, 6),
                "context": r.context,
            }
            for r in records
        ]

    def print_summary(self):
        """사용량 요약 출력"""
        print("\n" + "=" * 80)
        print("LLM Usage Summary")
        print("=" * 80)
        print(f"Total API Calls: {self.stats.total_calls:,}")
        print(f"Total Tokens: {self.stats.total_tokens:,}")
        print(f"  - Prompt Tokens: {self.stats.total_prompt_tokens:,}")
        print(f"  - Completion Tokens: {self.stats.total_completion_tokens:,}")
        print(f"Total Cost: ${self.stats.total_cost_usd:.4f} USD")

        if self.stats.by_model:
            print("\nBy Model:")
            for model, data in sorted(self.stats.by_model.items()):
                print(f"  {model}:")
                print(f"    Calls: {int(data['calls']):,}")
                print(f"    Tokens: {int(data['total_tokens']):,}")
                print(f"    Cost: ${data['cost_usd']:.4f}")

        if self.stats.by_context:
            print("\nBy Context:")
            for ctx, data in sorted(self.stats.by_context.items()):
                print(f"  {ctx}:")
                print(f"    Calls: {int(data['calls']):,}")
                print(f"    Tokens: {int(data['total_tokens']):,}")
                print(f"    Cost: ${data['cost_usd']:.4f}")

        print("=" * 80)


# 전역 tracker 인스턴스 쉽게 가져오기
def get_tracker() -> LLMUsageTracker:
    """전역 LLM usage tracker 반환"""
    return LLMUsageTracker.get_instance()
