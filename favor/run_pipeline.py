#!/usr/bin/env python
"""
Pipeline runner script.

Usage:
    # Run a single pipeline
    python run_pipeline.py "Mean Reversion after Panic Selling"
    python run_pipeline.py  # Uses the default concept

    # Run the outer loop
    python run_pipeline.py "Short-term mean reversion after downward price moves." --outer-loop 5
    python run_pipeline.py --outer-loop 10
"""
import sys
import os

def _ensure_conda_lib_in_ld_library_path() -> None:
    """
    Ensure the active env's `lib/` directory is present in LD_LIBRARY_PATH.

    Some native dependencies (e.g., torch / faiss) may require this when launched via non-interactive
    shells. This function re-execs the current Python process at most once.
    """

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

# Add project root to PYTHONPATH.
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from run.main import run_pipeline, run_outer_loop
from util.llm_tracker import get_tracker


def main():
    # Simple CLI parsing.
    args = sys.argv[1:]

    # Defaults.
    concept = "Mean Reversion after Panic Selling"
    use_outer_loop = False
    max_outer_iterations = None

    # If the first arg is not a flag, treat it as the concept string.
    if args and not args[0].startswith("--"):
        concept = args[0]
        args = args[1:]  # Remaining args

    # Parse --outer-loop.
    if "--outer-loop" in args:
        use_outer_loop = True
        idx = args.index("--outer-loop")
        # If a number is provided after --outer-loop, use it; otherwise, fall back to config defaults.
        if idx + 1 < len(args) and args[idx + 1].isdigit():
            max_outer_iterations = int(args[idx + 1])

    print("=" * 60)
    print("🚀 Hypothesis-Observation-Validation Framework")
    print("=" * 60)
    print(f"📝 Concept: {concept}")
    print(f"📁 Project root: {project_root}")
    if use_outer_loop:
        iterations_text = f"{max_outer_iterations} iterations" if max_outer_iterations else "Enabled (config default)"
        print(f"🔄 Outer Loop: {iterations_text}")
    else:
        print(f"🔄 Outer Loop: Disabled")
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

        # Print LLM usage summary.
        tracker = get_tracker()
        summary = tracker.get_summary()
        print(f"\n💰 LLM Usage:")
        print(f"   Total API Calls: {summary['total_calls']:,}")
        print(f"   Total Tokens: {summary['total_tokens']:,}")
        print(f"   Total Cost: ${summary['total_cost_usd']:.4f} USD")

        if use_outer_loop:
            # Outer loop outputs.
            print(f"\n📂 Results saved to: runs/{result['run_id']}/")
            print(f"   - specs/outer_loop_history.json (outer-iteration summary)")
            print(f"   - specs/*.json (each file contains outer_iter_1..N fields)")
            print(f"   - data/*_iter_N.parquet (per-iteration parquet)")
            print(f"   - specs/llm_usage.json (LLM token usage summary)")
        else:
            # Single-pipeline outputs.
            stage2_summary = (result.get("stage2") or {}).get("summary") or {}
            stage3_result = (result.get("stage3") or {}).get("result") or {}
            print(f"   Stage2 verdict: {stage2_summary.get('overall_verdict', 'N/A')}")
            print(f"   Stage3 verdict: {stage3_result.get('overall_verdict', 'N/A')}")
            print(f"   Stage2 n_tickers: {(result.get('stage2') or {}).get('n_tickers', 'N/A')}")
            print(f"   Stage3 n_tickers: {(result.get('stage3') or {}).get('n_tickers', 'N/A')}")
            print(f"\n📂 Results saved to: runs/{result['run_id']}/")
            print(f"   - specs/llm_usage.json (LLM token usage summary)")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
