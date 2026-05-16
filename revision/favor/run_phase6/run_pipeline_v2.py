#!/usr/bin/env python
"""
Phase 6 pipeline entrypoint — Per-Combo parallel runner with Stage 4 objective branching.

Identical to `run_pipeline_parallel_per_combo_parallel.py` except for one added line:
after the existing stage4 monkey-patch, we call `run_phase6.objective_patch.apply()`
to rebind `_create_objective` in all three stage4 modules. The behaviour at runtime
is controlled by env vars:

    FAVOR_STAGE4_OBJECTIVE  ∈ {"ir", "calmar", "ir_minus_mdd"}    default "ir"
    FAVOR_STAGE4_MDD_LAMBDA float (only when mode=="ir_minus_mdd")  default 2.0

Env vars (inherited from original runner):
    STAGE4_COMBO_WORKERS, STAGE4_OPTUNA_N_JOBS

Usage:
    python run_phase6/run_pipeline_v2.py "<concept text>" --combo-workers 12 --optuna-jobs 1
"""

import os
import sys
from datetime import datetime


def _ensure_conda_lib_in_ld_library_path() -> None:
    if os.environ.get("_FINAGENT_LD_LIBRARY_PATH_REEXEC") == "1":
        return

    env_prefix = os.environ.get("CONDA_PREFIX") or os.environ.get("VIRTUAL_ENV") or sys.prefix
    if not env_prefix:
        return

    conda_lib = os.path.join(env_prefix, "lib")
    if not os.path.isdir(conda_lib):
        return

    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in current_ld_path.split(":") if p] if current_ld_path else []
    if conda_lib in parts:
        return

    os.environ["LD_LIBRARY_PATH"] = f"{conda_lib}:{current_ld_path}" if current_ld_path else conda_lib
    os.environ["_FINAGENT_LD_LIBRARY_PATH_REEXEC"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)


_ensure_conda_lib_in_ld_library_path()

# project_root = revision/revision/favor/  (parent of run_phase6/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# ============================================================================
# Monkey-patch (1): stage4 → stage4_parallel_per_combo  (original runner step)
# ============================================================================
from run.pipeline import stage4_parallel_per_combo  # noqa: E402
import run.main as main_module  # noqa: E402

main_module.run_stage4 = stage4_parallel_per_combo.run_stage4_parallel_per_combo

# ============================================================================
# Monkey-patch (2): Phase 6 — replace _create_objective in stage4 + parallel wrappers
# ============================================================================
from run_phase6.objective_patch import apply as _apply_phase6_objective  # noqa: E402

_phase6_patch_info = _apply_phase6_objective()
_phase6_mode = os.getenv("FAVOR_STAGE4_OBJECTIVE", "ir").lower()

print("=" * 60)
print("🔄 Stage4 Per-Combo Parallel Mode Enabled")
print("   - Per-combination optimization + backtest in parallel workers")
print("   - IS/OOS backtest loops: per-combo (not merged)")
print(f"🎯 Phase 6 Stage 4 objective: {_phase6_mode}")
if _phase6_mode == "ir_minus_mdd":
    print(f"   λ (FAVOR_STAGE4_MDD_LAMBDA) = {os.getenv('FAVOR_STAGE4_MDD_LAMBDA', '2.0')}")
print("=" * 60)

# ============================================================================
# 이후 기존 run_pipeline.py와 동일
# ============================================================================
from run.main import run_pipeline, run_outer_loop  # noqa: E402
from util.llm_tracker import get_tracker  # noqa: E402


def _append_to_run_log(*, run_id: str, lines: list[str]) -> None:
    try:
        log_path = os.path.join(project_root, "runs", run_id, "logs", "run.log")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as f:
            for line in lines:
                f.write(f"[{ts}] {line}\n")
    except Exception:
        pass


def _pop_arg(args: list[str], flag: str) -> str | None:
    if flag not in args:
        return None
    idx = args.index(flag)
    if idx + 1 >= len(args):
        return None
    val = args[idx + 1]
    del args[idx : idx + 2]
    return val


def main() -> None:
    args = sys.argv[1:]

    concept = "Mean Reversion after Panic Selling"
    use_outer_loop = False
    max_outer_iterations = None

    if args and not args[0].startswith("--"):
        concept = args[0]
        args = args[1:]

    combo_workers_raw = _pop_arg(args, "--combo-workers")
    if combo_workers_raw is not None:
        try:
            os.environ["STAGE4_COMBO_WORKERS"] = str(int(combo_workers_raw))
        except Exception:
            pass

    optuna_jobs_raw = _pop_arg(args, "--optuna-jobs")
    if optuna_jobs_raw is not None:
        try:
            os.environ["STAGE4_OPTUNA_N_JOBS"] = str(int(optuna_jobs_raw))
        except Exception:
            pass

    if "--outer-loop" in args:
        use_outer_loop = True
        idx = args.index("--outer-loop")
        if idx + 1 < len(args) and args[idx + 1].isdigit():
            max_outer_iterations = int(args[idx + 1])

    combo_workers_effective = os.environ.get("STAGE4_COMBO_WORKERS", "1")
    optuna_jobs_effective = os.environ.get("STAGE4_OPTUNA_N_JOBS", "(auto)")

    print("=" * 60)
    print("🚀 Hypothesis-Observation-Validation Framework (Phase 6, Per-Combo Parallel)")
    print("=" * 60)
    print(f"📝 Concept: {concept}")
    print(f"📁 Project root: {project_root}")
    print(f"⚙️  STAGE4_COMBO_WORKERS: {combo_workers_effective}")
    print(f"⚙️  STAGE4_OPTUNA_N_JOBS: {optuna_jobs_effective}")
    print(f"🎯 FAVOR_STAGE4_OBJECTIVE: {_phase6_mode}")
    if use_outer_loop:
        iterations_text = f"{max_outer_iterations} iterations" if max_outer_iterations else "Enabled (default: 5)"
        print(f"🔄 Outer Loop: {iterations_text}")
    else:
        print("🔄 Outer Loop: Disabled")
    print("=" * 60)

    try:
        if use_outer_loop:
            result = run_outer_loop(
                concept=concept,
                max_outer_iterations=max_outer_iterations,
            )
        else:
            result = run_pipeline(concept=concept)

        print("\n" + "=" * 60)
        print("✅ Pipeline completed successfully!")
        print("=" * 60)
        print(f"   Run ID: {result['run_id']}")

        tracker = get_tracker()
        summary = tracker.get_summary()
        print("\n💰 LLM Usage:")
        print(f"   Total API Calls: {summary['total_calls']:,}")
        print(f"   Total Tokens: {summary['total_tokens']:,}")
        print(f"   Total Cost: ${summary['total_cost_usd']:.4f} USD")

        if use_outer_loop:
            saved_lines = [
                f"📂 Results saved to: runs/{result['run_id']}/",
                "   - specs/outer_loop_history.json",
                "   - specs/*.json",
                "   - data/*_iter_N.parquet",
                "   - specs/llm_usage.json",
            ]
            print("\n" + "\n".join(saved_lines))
            _append_to_run_log(run_id=result["run_id"], lines=saved_lines)
        else:
            stage2_summary = (result.get("stage2") or {}).get("summary") or {}
            stage3_result = (result.get("stage3") or {}).get("result") or {}
            print(f"   Stage2 verdict: {stage2_summary.get('overall_verdict', 'N/A')}")
            print(f"   Stage3 verdict: {stage3_result.get('overall_verdict', 'N/A')}")
            print(f"   Stage2 n_tickers: {(result.get('stage2') or {}).get('n_tickers', 'N/A')}")
            print(f"   Stage3 n_tickers: {(result.get('stage3') or {}).get('n_tickers', 'N/A')}")
            saved_lines = [
                f"📂 Results saved to: runs/{result['run_id']}/",
                "   - specs/llm_usage.json",
            ]
            print("\n" + "\n".join(saved_lines))
            _append_to_run_log(run_id=result["run_id"], lines=saved_lines)
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ Pipeline failed: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    main()
