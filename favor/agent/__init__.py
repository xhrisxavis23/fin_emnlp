"""
Agent Package
=============

This package contains all agents for the Hypothesis-Observation-Validation Framework.

Pipeline Stages:
- Stage 1: Formula Generation
  - HypothesisAgent: hypothesis generation
  - ObservationAgent: observation decomposition
  - FormulaAgent: formula generation

- Stage 2: Observation Formula Validation
  - ValidationAgent: validate formula observability (return-agnostic)

- Stage 3: Hypothesis Instance Validation
  - HypothesisValidationAgent: validate hypothesis structure (Strictness–Performance monotonicity)

Code Generation:
  - CoderCodeAgent: generate Polars-based code
  - CoSTEERFullCodeAgent: full CoSTEER pipeline
  - FactorCoderCodeAgent: generate Expression-based code
"""

from agent.base_agent import BaseAgent
from agent.hypothesis_agent import HypothesisAgent
from agent.observation_agent import ObservationAgent
from agent.formula_agent import FormulaAgent
from agent.validation_agent import ValidationAgent
from agent.hypothesis_validation_agent import HypothesisValidationAgent

# Import dataclasses from schemas for public API
from schemas.validation_dataclasses import (
    DistributionStats,
    FormulaValidationResult,
    LLMJudgment,
)
from schemas.hypothesis_validation_dataclasses import (
    HypothesisValidationResult,
    MonotonicityVerification,
    QuadrantStats,
    StrictnessLevelResult,
)

# Code agents
from agent.coder_code_agent import CoderCodeAgent
from agent.costeer_full_code_agent import CoSTEERFullCodeAgent
from agent.factor_coder_code_agent import FactorCoderCodeAgent

# Backward compatibility alias
from agent.diagnostics_agent import DiagnosticsAgent

__all__ = [
    # Base
    "BaseAgent",

    # Stage 1: Formula Generation
    "HypothesisAgent",
    "ObservationAgent",
    "FormulaAgent",

    # Stage 2: Observation Formula Validation
    "ValidationAgent",
    "FormulaValidationResult",
    "DistributionStats",
    "LLMJudgment",

    # Stage 3: Hypothesis Instance Validation
    "HypothesisValidationAgent",
    "HypothesisValidationResult",
    "StrictnessLevelResult",
    "QuadrantStats",
    "MonotonicityVerification",

    # Backward compatibility
    "DiagnosticsAgent",

    # Code Agents
    "CoderCodeAgent",
    "CoSTEERFullCodeAgent",
    "FactorCoderCodeAgent",
]
