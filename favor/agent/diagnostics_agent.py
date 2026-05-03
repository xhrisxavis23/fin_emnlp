"""
================================================================================
DiagnosticsAgent → HypothesisValidationAgent Redirect
================================================================================

This file is kept for backward compatibility with code that imports `DiagnosticsAgent`.
The actual implementation lives in `hypothesis_validation_agent.py`.

[Changes]
- Removed the legacy ReAct-based backtest analysis logic from `DiagnosticsAgent`
- Replaced it with the Stage 3 Hypothesis Instance Validation logic

[Stage 3 Goal]
- Build a hypothesis instance (H_t = obs1_t ∧ obs2_t ∧ ... ∧ obsN_t)
- Evaluate a strictness grid
- Verify monotonicity (Strictness ↑ → Performance ↑)
- Quadrant analysis (S1–S4)

[Migration Guide]
Before:
    from agent.diagnostics_agent import DiagnosticsAgent
    agent = DiagnosticsAgent(model="gpt-4o")
    result = agent.diagnose_experiment(...)

After:
    from agent.hypothesis_validation_agent import HypothesisValidationAgent
    agent = HypothesisValidationAgent(model="gpt-4o")
    result = agent.validate_hypothesis(...)

Or keep using the compatibility alias:
    from agent.diagnostics_agent import DiagnosticsAgent  # auto-redirect
    agent = DiagnosticsAgent(model="gpt-4o")
    result = agent.validate_hypothesis(...)
"""

# Backward compatibility: Import Agent from the new module
from agent.hypothesis_validation_agent import HypothesisValidationAgent

# Import dataclasses from schemas
from schemas.hypothesis_validation_dataclasses import (
    HypothesisValidationResult,
    StrictnessLevelResult,
    QuadrantStats,
    MonotonicityVerification,
)

# Alias for backward compatibility
DiagnosticsAgent = HypothesisValidationAgent

__all__ = [
    "DiagnosticsAgent",
    "HypothesisValidationAgent",
    "HypothesisValidationResult",
    "StrictnessLevelResult",
    "QuadrantStats",
    "MonotonicityVerification",
]
