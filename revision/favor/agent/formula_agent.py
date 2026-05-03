from __future__ import annotations

import logging
import json
import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from importlib_metadata import metadata

from agent.base_agent import BaseAgent
from util.run_context import RunContext
from prompts.formula_agent_prompts import (
    BEHAVIORAL_FORMULA_SELF_CORRECTION_SYSTEM_PROMPT,
    BEHAVIORAL_FORMULA_SELF_CORRECTION_USER_PROMPT_TEMPLATE,
    BEHAVIORAL_FORMULA_SYSTEM_PROMPT,
    BEHAVIORAL_FORMULA_USER_PROMPT_TEMPLATE,
    function_lib_description
)

try:
    from prompts.formula_agent_prompts import (
        BEHAVIORAL_FORMULA_REFINE_SYSTEM_PROMPT,
        BEHAVIORAL_FORMULA_REFINE_USER_PROMPT_TEMPLATE,
    )
except ImportError:
    BEHAVIORAL_FORMULA_REFINE_SYSTEM_PROMPT = BEHAVIORAL_FORMULA_SYSTEM_PROMPT
    BEHAVIORAL_FORMULA_REFINE_USER_PROMPT_TEMPLATE = BEHAVIORAL_FORMULA_USER_PROMPT_TEMPLATE

# Legacy prompts are kept in prompts_khj/ for backward compatibility with older workflows.
try:
    from prompts_khj.factor_agent_prompts import (  # type: ignore
        FACTOR_SYSTEM_PROMPT,
        FACTOR_USER_PROMPT_TEMPLATE,
        FACTOR_REFINE_SYSTEM_PROMPT,
        FACTOR_REFINE_USER_PROMPT_TEMPLATE,
        FACTOR_REFINE_IC_SYSTEM_PROMPT,
        FACTOR_REFINE_IC_USER_PROMPT_TEMPLATE,
    )
except Exception:
    FACTOR_SYSTEM_PROMPT = None
    FACTOR_USER_PROMPT_TEMPLATE = None
    FACTOR_REFINE_SYSTEM_PROMPT = None
    FACTOR_REFINE_USER_PROMPT_TEMPLATE = None
    FACTOR_REFINE_IC_SYSTEM_PROMPT = None
    FACTOR_REFINE_IC_USER_PROMPT_TEMPLATE = None
from schemas.behavioral_formula import BEHAVIORAL_FORMULA_TOOL
from schemas.formula import FORMULA_TOOL
from util.llm_client import call_llm

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _GuardResult:
    ok: bool
    errors: list[str]


# Identifier tokenizer used by the formula guard.
# Exclude tokens that are immediately preceded by a digit to avoid flagging
# scientific notation like "1e-8" / "1E-8" as unknown symbol "e"/"E".
_TOKEN_RE = re.compile(r"(?<![0-9])[A-Za-z_][A-Za-z0-9_]*")


def _normalize_columns(cols: list[str] | None) -> set[str]:
    if not cols:
        return set()
    return {str(c).strip().lower() for c in cols if str(c).strip()}


def _extract_identifier_tokens(expr: str) -> list[str]:
    return _TOKEN_RE.findall(expr or "")


def _check_allowed_columns(definition: str, allowed_columns: set[str], formula_name: str = "", formula_index: int = -1) -> list[str]:
    if not allowed_columns:
        return []

    known_words = {
        "delay",
        "rank",
        "ts_min",
        "ts_max",
        "ts_mean",
        "ts_sum",
        "ts_rank",
        "ts_std",
        "sma",
        "ema",
        "abs",
        "log",
        "sqrt",
        "exp",
        "sign",
        "min",
        "max",
        "nan",
        "inf",
    }

    unknown: list[str] = []
    for tok in _extract_identifier_tokens(definition):
        low = tok.lower()
        if low in allowed_columns or low in known_words:
            continue
        if tok.isupper():  # allow function-like tokens (TS_MEAN etc.)
            continue
        unknown.append(tok)

    if not unknown:
        return []

    seen: set[str] = set()
    uniq = [e for e in unknown if not (e in seen or seen.add(e))]

    # Build descriptive error message
    formula_info = ""
    if formula_index >= 0:
        formula_info = f"formulas[{formula_index}]"
        if formula_name:
            formula_info += f" (name='{formula_name}')"
    elif formula_name:
        formula_info = f"Formula '{formula_name}'"

    if formula_info:
        return [f"{formula_info} references non-existent columns/symbols: {', '.join(uniq)}. Definition: {definition}"]
    else:
        return [f"Formula references non-existent columns/symbols: {', '.join(uniq)}. Definition: {definition}"]


# Formula bundle guard: validates schema/columns/operators and detects duplicate definitions, etc.
def _validate_behavioral_bundle(
    bundle: Dict[str, Any],
    allowed_columns: list[str] | None = None,
    expected_observation_ids: list[str] | None = None,
    min_formulas_per_observation: int = 2,
    max_formulas_per_observation: int = 3,
) -> _GuardResult:
    errors: list[str] = []

    if not isinstance(bundle, dict):
        return _GuardResult(ok=False, errors=["Bundle must be a dict."])

    obs_desc = bundle.get("observation_descriptions")
    formulas = bundle.get("formulas")

    if not isinstance(obs_desc, list) or not obs_desc:
        errors.append("Missing observation_descriptions.")
        obs_desc = []

    if not isinstance(formulas, list) or not formulas:
        errors.append("Missing formulas.")
        formulas = []

    obs_ids: list[str] = []
    for i, obs in enumerate(obs_desc):
        if not isinstance(obs, dict):
            errors.append(f"observation_descriptions[{i}] must be an object.")
            continue
        oid = str(obs.get("observation_id") or "").strip()
        if not oid:
            errors.append(f"observation_descriptions[{i}] missing observation_id.")
            continue
        obs_ids.append(oid)

    expected_ids: list[str] = []
    if expected_observation_ids:
        expected_ids = [str(x).strip() for x in expected_observation_ids if str(x).strip()]
        missing_from_bundle = [oid for oid in expected_ids if oid not in set(obs_ids)]
        if missing_from_bundle:
            errors.append(
                "Bundle observation_descriptions missing observation_id(s) from observation_plan: "
                + ", ".join(missing_from_bundle)
            )

    allowed_cols_set = _normalize_columns(allowed_columns)

    formula_by_obs: dict[str, list[dict[str, Any]]] = {}
    defs_seen: dict[str, str] = {}
    for i, f in enumerate(formulas):
        if not isinstance(f, dict):
            errors.append(f"formulas[{i}] must be an object.")
            continue

        kind = str(f.get("kind") or "").strip()
        if kind and kind != "evidence":
            errors.append(f"formulas[{i}] kind must be 'evidence' (got {kind!r}).")

        name = str(f.get("name") or "").strip()
        if not name:
            errors.append(f"formulas[{i}] missing name.")

        oid = str(f.get("observation_id") or "").strip()
        if not oid:
            errors.append(f"formulas[{i}] missing observation_id.")
        else:
            formula_by_obs.setdefault(oid, []).append(f)

        polarity = str(f.get("polarity") or "").strip()
        if polarity not in {"higher_is_more_true", "lower_is_more_true"}:
            errors.append(
                f"formulas[{i}] polarity must be 'higher_is_more_true' or 'lower_is_more_true' (got {polarity!r})."
            )

        definition = str(f.get("definition") or "").strip()
        if not definition:
            errors.append(f"formulas[{i}] missing definition.")
            continue

        if any(op in definition for op in ["==", "!=", ">=", "<=", ">", "<", "&&", "||"]):
            found_ops = [op for op in ["==", "!=", ">=", "<=", ">", "<", "&&", "||"] if op in definition]
            errors.append(
                f"formulas[{i}] (name='{name}') definition must be continuous numeric (no comparisons/logical ops). "
                f"Found operators: {', '.join(found_ops)}. Definition: {definition}"
            )

        errors.extend(_check_allowed_columns(definition, allowed_cols_set))

        canonical = re.sub(r"\s+", " ", definition).strip()
        existing = defs_seen.get(canonical)
        if existing is not None and existing != name:
            errors.append(f"Duplicate formula definitions detected (e.g., {existing!r} and {name!r}).")
        else:
            defs_seen[canonical] = name

    missing = [oid for oid in obs_ids if oid not in formula_by_obs]
    if missing:
        errors.append(f"Missing evidence formulas for observation_ids: {', '.join(missing)}")

    # NOTE: 1:N mapping is allowed (multiple formulas per observation_id)
    # This enables Formula Pool functionality where each observation can have
    # 2-3 diverse formulas using different mathematical approaches
    base_ids = expected_ids if expected_ids else obs_ids
    if min_formulas_per_observation and min_formulas_per_observation > 1:
        for oid in base_ids:
            n = len(formula_by_obs.get(oid, []))
            if n < min_formulas_per_observation:
                errors.append(
                    f"Insufficient evidence formulas for observation_id {oid!r}: "
                    f"expected {min_formulas_per_observation}–{max_formulas_per_observation}, got {n}."
                )
    if max_formulas_per_observation and max_formulas_per_observation > 0:
        for oid in base_ids:
            n = len(formula_by_obs.get(oid, []))
            if n > max_formulas_per_observation:
                errors.append(
                    f"Too many evidence formulas for observation_id {oid!r}: "
                    f"expected {min_formulas_per_observation}–{max_formulas_per_observation}, got {n}."
                )

    unknown = [oid for oid in formula_by_obs.keys() if oid not in set(obs_ids)]
    if unknown:
        errors.append(f"Formulas reference unknown observation_ids: {', '.join(unknown)}")

    return _GuardResult(ok=(len(errors) == 0), errors=errors)


def _validate_refinement_delta(
    *,
    previous_bundle: Dict[str, Any],
    refined_bundle: Dict[str, Any],
    allowed_columns: list[str] | None,
) -> _GuardResult:
    v = _validate_behavioral_bundle(refined_bundle, allowed_columns=allowed_columns)
    if not v.ok:
        return v

    def _obs_ids(b: Dict[str, Any]) -> set[str]:
        obs = b.get("observation_descriptions")
        if not isinstance(obs, list):
            return set()
        out: set[str] = set()
        for item in obs:
            if isinstance(item, dict):
                oid = str(item.get("observation_id") or "").strip()
                if oid:
                    out.add(oid)
        return out

    prev_ids = _obs_ids(previous_bundle)
    new_ids = _obs_ids(refined_bundle)
    if prev_ids and new_ids != prev_ids:
        return _GuardResult(ok=False, errors=["Refinement changed observation_ids set; rejected."])

    return _GuardResult(ok=True, errors=[])


class FormulaAgent(BaseAgent):
    """
    Formula Agent

    Purpose:
    - Convert a behavioral hypothesis into a set of observable evidence conditions.
    - Generate continuous evidence formulas.
    """

    def __init__(
        self,
        model: str,
        run_ctx: Optional[RunContext] = None,
    ):
        """
        Initialize FormulaAgent.
        
        Args:
            model: LLM model name
            run_ctx: RunContext for logging and artifacts
        """
        super().__init__(model=model, run_ctx=run_ctx)

    def _self_correction_loop(
        self,
        bundle: Dict[str, Any],
        hypothesis: Dict[str, Any],
        rounds: int = 1,
        allowed_columns: list = None,
    ) -> Dict[str, Any]:
        """
        Run a pre-execution self-correction loop to fix logical flaws (Polarity, Overlap, etc.).
        """
        if rounds < 1:
            return bundle

        current_bundle = bundle

        for i in range(rounds):
            logger.info(f"Running Self-Correction Loop {i+1}/{rounds}...")

            bundle_json = json.dumps(current_bundle, ensure_ascii=False, indent=2, default=str)

            user_prompt = BEHAVIORAL_FORMULA_SELF_CORRECTION_USER_PROMPT_TEMPLATE.format(
                bundle_json=bundle_json,
                columns=metadata,
                function_lib_description=function_lib_description,
            )

            resp = call_llm(
                model=self.model,
                system_prompt=BEHAVIORAL_FORMULA_SELF_CORRECTION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                tools=[BEHAVIORAL_FORMULA_TOOL],
                target_tool_name="behavioral_formula_tool",
                temperature=0.7, # Low temp for strict logic checking
                react_log_path=self.run_ctx.root_dir / "logs/agents" if self.run_ctx else None,
                react_agent_name="formula_agent_self_correction",
                context="Stage1: Formula Self-Correction",
            )

            if isinstance(resp, dict) and isinstance(resp.get("formulas"), list):
                # Basic schema check passed
                # Canonicalize common schema variants and ensure names exist before diff/validation.
                resp = self._canonicalize_llm_bundle(resp)
                resp = self._auto_name_formulas(resp)

                # Check if it actually changed anything
                if json.dumps(resp, sort_keys=True) == json.dumps(current_bundle, sort_keys=True):
                    logger.info("Self-Correction: No changes proposed. Stopping early.")
                    break

                # Validate the new bundle
                v = _validate_behavioral_bundle(resp, allowed_columns=allowed_columns)
                if v.ok:
                    current_bundle = resp
                    logger.info("Self-Correction: Bundle updated.")
                else:
                    logger.warning(f"Self-Correction: Proposed bundle invalid ({'; '.join(v.errors)}). Keeping previous. Invalid bundle: {json.dumps(resp, ensure_ascii=False)}")
            else:
                logger.warning(f"Self-Correction: LLM returned invalid response format or empty response. Keeping previous. Raw response: {resp}")

        return current_bundle

    def _canonicalize_llm_bundle(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make LLM outputs more robust by canonicalizing common variants into the expected schema.
        This mutates and returns the same dict.
        """
        if not isinstance(bundle, dict):
            return bundle

        # Drop any violation-kind formulas if present.
        if isinstance(bundle.get("formulas"), list):
            before = len(bundle["formulas"])
            bundle["formulas"] = [
                f for f in bundle["formulas"]
                if isinstance(f, dict) and str(f.get("kind") or "").strip() != "violation"
            ]
            after = len(bundle["formulas"])
            if after != before:
                logger.info(f"Removed violation formulas from bundle: {before} -> {after}")

        return bundle

    def _auto_name_formulas(self, bundle: Dict[str, Any], prefix: str = "formula") -> Dict[str, Any]:
        """
        Ensure all formulas have names. If LLM provided names, keep them.
        If not, auto-generate: formula001, formula002, formula003, etc.

        Args:
            bundle: Formula bundle dict
            prefix: Prefix for formula names (default: "formula")

        Returns:
            Modified bundle with names
        """
        if not isinstance(bundle, dict):
            return bundle

        formulas = bundle.get("formulas", [])
        if not isinstance(formulas, list):
            return bundle

        # Only assign names to formulas that don't have one
        for i, formula in enumerate(formulas, 1):
            if isinstance(formula, dict):
                current_name = formula.get("name", "").strip()
                if not current_name:
                    # No name provided, auto-generate
                    formula["name"] = f"{prefix}{i:03d}"
                # else: keep the LLM-provided name

        return bundle

    def _increment_formula_suffix(self, name: str) -> str:
        """
        Increment the suffix of a formula name.
        formula001 -> formula001_1
        formula001_1 -> formula001_2
        formula001_2 -> formula001_3
        """
        import re
        match = re.match(r'(.+)_(\d+)$', name)
        if match:
            base = match.group(1)
            num = int(match.group(2))
            return f"{base}_{num + 1}"
        else:
            # No suffix, add _1
            return f"{name}_1"

    def _add_version_suffix(self, name: str) -> str:
        """
        Add or increment version suffix: formula001 -> formula001_v1 -> formula001_v2
        """
        import re
        match = re.match(r'(.+)_v(\d+)$', name)
        if match:
            # Already has version suffix, increment it
            base = match.group(1)
            version = int(match.group(2))
            return f"{base}_v{version + 1}"
        else:
            # No version suffix yet, add _v1
            return f"{name}_v1"

    def _auto_name_formulas_refinement(
        self,
        current_bundle: Dict[str, Any],
        refined_bundle: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check if formula definitions changed and add version suffix if needed.
        If formula001 definition changes: formula001 -> formula001_v1 -> formula001_v2

        Args:
            current_bundle: Original formula bundle (with all formulas for context)
            refined_bundle: Refined formula bundle

        Returns:
            Modified refined_bundle with versioned names where definitions changed
        """
        logger.info("═══ [Refinement Naming] Starting _auto_name_formulas_refinement ═══")

        if not isinstance(current_bundle, dict) or not isinstance(refined_bundle, dict):
            logger.warning("[Refinement Naming] Bundle type check failed")
            return refined_bundle

        current_formulas = current_bundle.get("formulas", [])
        refined_formulas = refined_bundle.get("formulas", [])

        logger.info(f"[Refinement Naming] Current bundle has {len(current_formulas)} formulas")
        logger.info(f"[Refinement Naming] Refined bundle has {len(refined_formulas)} formulas")

        if not isinstance(current_formulas, list) or not isinstance(refined_formulas, list):
            logger.warning("[Refinement Naming] Formula list type check failed")
            return refined_bundle

        # Create mapping: name -> definition from current bundle
        name_to_definition = {}
        for formula in current_formulas:
            if isinstance(formula, dict):
                name = formula.get("name", "").strip()
                definition = formula.get("definition", "").strip()
                if name and definition:
                    name_to_definition[name] = definition

        logger.info(f"[Refinement Naming] name_to_definition mapping has {len(name_to_definition)} entries")

        # Check each refined formula and add version if definition changed
        for i, refined_formula in enumerate(refined_formulas):
            if isinstance(refined_formula, dict):
                current_name = refined_formula.get("name", "").strip()
                new_definition = refined_formula.get("definition", "").strip()

                if not current_name:
                    logger.warning(f"[Refinement Naming] Formula {i} has no name - LLM should provide names!")
                    continue

                # Check if definition changed
                if current_name in name_to_definition:
                    old_definition = name_to_definition[current_name]
                    if old_definition != new_definition:
                        # Definition changed - add version suffix
                        new_name = self._add_version_suffix(current_name)
                        logger.info(f"[Refinement Naming] Definition changed: {current_name} -> {new_name}")
                        refined_formula["name"] = new_name
                    else:
                        logger.info(f"[Refinement Naming] Definition unchanged: {current_name}")
                else:
                    # New formula name not in original bundle
                    logger.info(f"[Refinement Naming] New formula: {current_name}")

        logger.info("═══ [Refinement Naming] Finished _auto_name_formulas_refinement ═══")

        return refined_bundle

    def purpose_formula(
        self,
        hypothesis: Dict[str, Any],
        metadata: list = None,
        knowledge: str = "",
        formula_memory: list = None,
        refine_rounds: int = 1,
        observation_plan: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Generates an observation-driven alpha bundle based on a behavioral hypothesis.

        Args:
            hypothesis: The hypotheses response dictionary containing a list of hypotheses.
            metadata: Optional list of metadata.
            knowledge: Retrieved knowledge context.
            formula_memory: List of previously generated formulas.
            refine_rounds: Number of self-correction rounds before returning.
            observation_plan: Optional observation plan from ObservationAgent.

        Returns:
            The generated bundle dictionary (includes `formulas`).
        """
        
        hyp_list = hypothesis.get("hypotheses", []) if isinstance(hypothesis, dict) else []
        hyp_obj = hyp_list[0] if hyp_list and isinstance(hyp_list[0], dict) else (hypothesis if isinstance(hypothesis, dict) else {})

        hypothesis_id = (
            (hyp_obj or {}).get("hypothesis_id")
            or (hyp_obj or {}).get("id")
            or "Unknown"
        )

        observation_plan_json = json.dumps(
            observation_plan or {},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        ) if observation_plan else "None"

        # Format Formula Memory (names/definitions only; no IC/validation metadata injection)
        memory_lines: List[str] = []
        if formula_memory:
            for item in formula_memory:
                # Case A: IC Refinement loop feedback (has 'feedback' field)
                if isinstance(item, dict) and "feedback" in item:
                    memory_lines.append(f"--- FAILED ATTEMPT FEEDBACK ---\n{item['feedback']}\n-------------------------------")
                    continue
                
                # Case B: Standard formula memory (list of formulas)
                for f in item.get("formulas", []):
                    f_name = f.get("name", "Unknown")
                    f_def = f.get("definition", "N/A")
                    memory_lines.append(f"- {f_name}: {f_def}")

        memory_text = "\n".join(memory_lines) if memory_lines else "None"

        user_prompt = BEHAVIORAL_FORMULA_USER_PROMPT_TEMPLATE.format(
            hypothesis_id=hypothesis_id,
            observation_plan_json=observation_plan_json,
            columns=metadata,
            function_lib_description=function_lib_description,
            knowledge=knowledge,
            formula_memory=memory_text
        )

        response_dict: Dict[str, Any] | None = None
        last_errors: List[str] = []
        for attempt in range(3):  # Increased from 2 to 3 attempts
            attempt_prompt = user_prompt
            if attempt > 0 and last_errors:
                # Format errors more explicitly for LLM understanding
                error_feedback = "\n\n" + "=" * 60 + "\n"
                error_feedback += "CRITICAL: Your previous bundle failed validation. YOU MUST FIX THESE ERRORS:\n"
                error_feedback += "=" * 60 + "\n"
                for err in last_errors[:12]:
                    if "Missing evidence formulas for observation_ids" in err:
                        # Extract missing IDs and emphasize
                        error_feedback += f"\n** REQUIRED ACTION: {err}\n"
                        error_feedback += "   -> Create 2–3 evidence formulas (kind='evidence') for EACH listed observation_id.\n"
                    elif "Insufficient evidence formulas for observation_id" in err or "Too many evidence formulas for observation_id" in err:
                        error_feedback += f"\n** REQUIRED ACTION: {err}\n"
                        error_feedback += "   -> For EACH observation_id, create 2–3 evidence formulas (no more, no less).\n"
                    # elif "1:1 mapping violation" in err:
                    #     error_feedback += f"\n** MAPPING ERROR: {err}\n"
                    #     error_feedback += "   -> Ensure EXACTLY 1 evidence formula per observation_description.\n"
                    elif "Duplicate formula definitions" in err:
                        error_feedback += f"\n** DUPLICATE FORMULA ERROR: {err}\n"
                        error_feedback += "   -> EACH formula MUST have a UNIQUE mathematical formula.\n"
                        error_feedback += "   -> Different observations measure DIFFERENT phenomena - use DIFFERENT formulas.\n"
                        error_feedback += "   -> Example fixes:\n"
                        error_feedback += "      * 'no_fundamental_info': ABS(close - TS_MEAN(close, 5)) / TS_STD(close, 5)  (price stability)\n"
                        error_feedback += "      * 'volatility_spike': TS_STD(close, 5) / TS_STD(close, 20)  (short vs long vol ratio)\n"
                    elif "references non-existent columns" in err:
                        error_feedback += f"\n** INVALID COLUMN ERROR: {err}\n"
                        error_feedback += "   -> You can ONLY use columns from the Allowed Columns list.\n"
                        error_feedback += "   -> Rewrite the formula using ONLY available columns.\n"
                        error_feedback += "   -> If you need bid-ask spread, approximate it using: (high - low) / close\n"
                        error_feedback += "   -> If you need other unavailable data, find an alternative proxy from allowed columns.\n"
                    else:
                        error_feedback += f"\n- {err}\n"
                error_feedback += "\nDo not change the hypothesis or strategy. Only fix the bundle structure.\n"
                attempt_prompt = attempt_prompt + error_feedback

            response_dict = call_llm(
                model=self.model,
                system_prompt=BEHAVIORAL_FORMULA_SYSTEM_PROMPT,
                user_prompt=attempt_prompt,
                tools=[BEHAVIORAL_FORMULA_TOOL],
                target_tool_name="behavioral_formula_tool",
                temperature=0.7,
                react_log_path=self.run_ctx.root_dir / "logs/agents" if self.run_ctx else None,
                react_agent_name="formula_agent_generation",
                context="Stage1: Formula Generation",
            )

            if isinstance(response_dict, dict) and isinstance(response_dict.get("formulas"), list):
                response_dict.setdefault("hypothesis_id", hypothesis_id)
                # Normalize common LLM mistakes (&& / 0.9 quantile levels) before validation
                response_dict = self._canonicalize_llm_bundle(response_dict)
                # Auto-generate formula names (formula001, formula002, etc.)
                response_dict = self._auto_name_formulas(response_dict)
                expected_obs_ids = []
                if observation_plan and isinstance(observation_plan, dict):
                    obs_list = observation_plan.get("observations", [])
                    if isinstance(obs_list, list):
                        expected_obs_ids = [
                            str(o.get("observation_id") or "").strip()
                            for o in obs_list
                            if isinstance(o, dict) and str(o.get("observation_id") or "").strip()
                        ]

                v = _validate_behavioral_bundle(
                    response_dict,
                    allowed_columns=metadata,
                    expected_observation_ids=expected_obs_ids,
                    min_formulas_per_observation=2,
                    max_formulas_per_observation=3,
                )
                if v.ok:
                    # If valid, run self-correction: repeat "bundle fix suggestion → guard re-validation" for `refine_rounds`.
                    final_bundle = self._self_correction_loop(response_dict, hypothesis, rounds=refine_rounds, allowed_columns=metadata)
                    return final_bundle
                last_errors = v.errors
                # Log formula definitions for debugging duplicate detection
                if "Duplicate formula definitions" in "; ".join(v.errors):
                    formulas_debug = response_dict.get("formulas", [])
                    logger.error("behavioral bundle validation failed (attempt=%s): %s", attempt + 1, "; ".join(v.errors))
                    logger.error("Generated formulas for debugging:")
                    for f in formulas_debug:
                        logger.error("  - %s: %s", f.get("name"), f.get("definition"))
                else:
                    logger.error("behavioral bundle validation failed (attempt=%s): %s", attempt + 1, "; ".join(v.errors))
                continue
            logger.error("Failed to parse behavioral formula bundle response (attempt=%s): %r", attempt + 1, response_dict)
            last_errors = ["Failed to parse behavioral bundle tool output."]
            continue

        # Fail closed: return with validation_errors for caller to stop early (prevents silent mining/duplication).
        return {
            "hypothesis_id": hypothesis_id,
            "formulas": [],
            "validation_errors": last_errors or ["behavioral bundle validation failed"],
        }

    def refine_behavioral_bundle(
        self,
        *,
        hypothesis: Dict[str, Any],
        current_bundle: Dict[str, Any],
        diagnostics: Dict[str, Any],
        metadata: list = None,
        knowledge: str = "",
        focus: str = "Improve hypothesis-observation alignment; reduce invalid exposure; improve failure attribution.",
        refine_rounds: int = 1,
        observation_plan: Dict[str, Any] = None,
        original_bundle: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Refine an observation-driven alpha bundle using alignment diagnostics (NOT IC/sharpe optimization).

        This is intended for the research-safe refinement loop:
        - local DSL edits, thresholds/persistence tweaks, violation/regime improvements
        - no strategy changes, no rank/quantile/weight optimization, no IC sign flip
        """
        hyp_list = hypothesis.get("hypotheses", []) if isinstance(hypothesis, dict) else []
        hyp_obj = hyp_list[0] if hyp_list and isinstance(hyp_list[0], dict) else (hypothesis if isinstance(hypothesis, dict) else {})
        hypothesis_id = (
            (hyp_obj or {}).get("hypothesis_id")
            or (hyp_obj or {}).get("id")
            or current_bundle.get("hypothesis_id")
            or "Unknown"
        )

        if not knowledge:
            knowledge = "None"

        observation_plan_json = json.dumps(
            observation_plan or {},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        ) if observation_plan else "None"

        current_bundle_json = json.dumps(
            current_bundle or {},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
        diagnostics_json = json.dumps(
            diagnostics or {},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )

        user_prompt = BEHAVIORAL_FORMULA_REFINE_USER_PROMPT_TEMPLATE.format(
            hypothesis_id=hypothesis_id,
            observation_plan_json=observation_plan_json,
            current_bundle_json=current_bundle_json,
            columns=metadata,
            function_lib_description=function_lib_description,
            diagnostics_json=diagnostics_json,
            focus=focus or "Improve alignment.",
        )

        resp = call_llm(
            model=self.model,
            system_prompt=BEHAVIORAL_FORMULA_REFINE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            tools=[BEHAVIORAL_FORMULA_TOOL],
            target_tool_name="behavioral_formula_tool",
            temperature=0.7,
            react_log_path=self.run_ctx.root_dir / "logs/agents" if self.run_ctx else None,
            react_agent_name="formula_agent_refine",
            context="Stage1: Formula Refinement",
        )

        if isinstance(resp, dict) and isinstance(resp.get("formulas"), list):
            resp.setdefault("hypothesis_id", hypothesis_id)
            # Auto-generate formula names with refinement suffix
            # Use original_bundle if provided, otherwise fall back to current_bundle
            bundle_for_naming = original_bundle if original_bundle is not None else current_bundle
            resp = self._auto_name_formulas_refinement(bundle_for_naming, resp)
            v = _validate_refinement_delta(
                previous_bundle=current_bundle,
                refined_bundle=resp,
                allowed_columns=metadata,
            )
            if not v.ok:
                logger.error("refined bundle rejected by guards: %s", "; ".join(v.errors))
                return current_bundle
            
            # Apply Self-Correction Loop (Refinement Phase)
            final_bundle = self._self_correction_loop(resp, hypothesis, rounds=refine_rounds, allowed_columns=metadata)
            return final_bundle

        logger.error("Failed to parse refined behavioral bundle: %r", resp)
        return current_bundle

    def refine_formula(self, hypothesis: Dict[str, Any], formula: Dict[str, Any], feedback: str) -> Dict[str, Any]:
        """
        Refines a formula based on feedback.

        Args:
            hypothesis: The hypothesis dictionary.
            formula: The original formula dictionary.
            feedback: Feedback string explaining why it is misaligned.

        Returns:
            The refined formula dictionary.
        """
        if not FACTOR_REFINE_SYSTEM_PROMPT or not FACTOR_REFINE_USER_PROMPT_TEMPLATE:
            raise RuntimeError(
                "Legacy refine_factor prompts are unavailable. "
                "Expected prompts in `prompts_khj/factor_agent_prompts.py`."
            )
        
        # Backward-compatible hypothesis formatting (hypothesis payload is typically {'hypotheses':[...]} in this repo)
        hyp_obj = None
        if isinstance(hypothesis, dict):
            if "hypotheses" in hypothesis and hypothesis.get("hypotheses"):
                hyp_obj = hypothesis.get("hypotheses", [None])[0]
            else:
                hyp_obj = hypothesis

        hyp_title = (hyp_obj or {}).get("title", "")
        hyp_desc = (hyp_obj or {}).get("hypothesis", "") or (hyp_obj or {}).get("description", "")
        hyp_rat = (hyp_obj or {}).get("reason", "")
        hypothesis_text = f"Title: {hyp_title}\nDescription: {hyp_desc}\nRationale: {hyp_rat}"
        
        fac_name = formula.get("name", "")
        fac_def = formula.get("definition", "")
        fac_desc = formula.get("description", "")
        
        # NOTE: This legacy refine prompt template is not currently used by the main loop.
        # Keep it functional for potential future use.
        user_prompt = FACTOR_REFINE_USER_PROMPT_TEMPLATE.format(
            hypothesis_id=(hyp_obj or {}).get("id", "Unknown"),
            hypothesis=hypothesis_text,
            columns="N/A",
            factor_name=fac_name,
            factor_definition=fac_def,
            factor_description=fac_desc,
            feedback=feedback,
            state_id=None,
        )
        
        response_dict = call_llm(
            model=self.model,
            system_prompt=FACTOR_REFINE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            tools=[FORMULA_TOOL],
            target_tool_name="formula_tool",
            temperature=0.7, # Slightly lower temperature for focused refinement
            react_log_path=self.run_ctx.root_dir / "logs/agents" if self.run_ctx else None,
            react_agent_name="formula_agent_refine_factor",
            context="Stage1: Factor Refinement (Legacy)",
        )
        
        if isinstance(response_dict, dict) and "formulas" in response_dict:
            # Return the first formula (refinement should produce one)
            formulas = response_dict.get("formulas", [])
            if formulas:
                return formulas[0]
        
        logger.error(f"Failed to parse refined formula response: {response_dict}")
        return {}

    def refine_formula_with_ic(
        self,
        hypothesis: Dict[str, Any],
        formula: Dict[str, Any],
        ic_feedback: Dict[str, Any],
        metadata: list = None,
    ) -> Dict[str, Any]:
        """
        Refine a formula using ONLY quantitative IC feedback (numbers).

        Args:
            hypothesis: Hypothesis payload (usually {'hypotheses':[...]}).
            formula: Formula dict with 'name'/'definition'/'description'.
            ic_feedback: Dict containing only IC metrics (numbers/None).
            metadata: Allowed columns list.

        Returns:
            A single refined formula dict (name/definition/description), or {}.
        """
        if not FACTOR_REFINE_IC_SYSTEM_PROMPT or not FACTOR_REFINE_IC_USER_PROMPT_TEMPLATE:
            raise RuntimeError(
                "Legacy refine_factor_with_ic prompts are unavailable. "
                "Expected prompts in `prompts_khj/factor_agent_prompts.py`."
            )
        hyp_obj = None
        if isinstance(hypothesis, dict):
            if "hypotheses" in hypothesis and hypothesis.get("hypotheses"):
                hyp_obj = hypothesis.get("hypotheses", [None])[0]
            else:
                hyp_obj = hypothesis

        hypothesis_id = (hyp_obj or {}).get("id", "Unknown")

        fac_name = formula.get("name", "")
        fac_def = formula.get("definition", "")
        target_horizon_days = (hyp_obj or {}).get("target_horizon_days")

        ic_feedback_json = json.dumps(ic_feedback, ensure_ascii=False, sort_keys=True)

        user_prompt = FACTOR_REFINE_IC_USER_PROMPT_TEMPLATE.format(
            hypothesis_id=hypothesis_id,
            columns=metadata,
            factor_name=fac_name,
            factor_definition=fac_def,
            ic_feedback_json=ic_feedback_json,
            target_horizon_days=target_horizon_days,
            state_id=None,
        )

        response_dict = call_llm(
            model=self.model,
            system_prompt=FACTOR_REFINE_IC_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            tools=[FORMULA_TOOL],
            target_tool_name="formula_tool",
            temperature=0.7,
            react_log_path=self.run_ctx.root_dir / "logs/agents" if self.run_ctx else None,
            react_agent_name="formula_agent_refine_ic",
            context="Stage1: IC-based Formula Refinement",
        )

        if isinstance(response_dict, dict) and "formulas" in response_dict:
            formulas = response_dict.get("formulas", []) or []
            if formulas and isinstance(formulas[0], dict):
                # Ensure name stability (overwrite if the model changed it)
                formulas[0]["name"] = fac_name
                return formulas[0]

        logger.error(f"Failed to parse IC-refined formula response: {response_dict}")
        return {}
