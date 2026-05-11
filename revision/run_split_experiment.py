"""Wrapper for revision/favor that injects custom data_split.

Usage:
    python run_split_experiment.py "concept text" \\
        --train-start 2022-01-01 --train-end 2023-12-31 \\
        --val-start   2024-01-01 --val-end   2024-12-31 \\
        --test-start  2025-01-01 --test-end  2025-12-31 \\
        --outer-loop 3

Overrides cfg.data_split AFTER load_rd_config() — no favor codebase mod.
Run dir is created under revision/favor/runs/ as usual.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

# 1) Setup environment + import path BEFORE importing favor modules
FAVOR_DIR = Path("/home/dgu_wj92/fin_emnlp/revision/favor")
os.environ.setdefault("MARKET", "cn")
os.chdir(FAVOR_DIR)
sys.path.insert(0, str(FAVOR_DIR))

# 2) LD_LIBRARY_PATH boost (mirror run_pipeline.py self-exec logic)
def _ensure_conda_lib():
    conda = os.environ.get("CONDA_PREFIX") or os.environ.get("VIRTUAL_ENV")
    if not conda:
        return
    libdir = os.path.join(conda, "lib")
    if not os.path.isdir(libdir):
        return
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if libdir in cur.split(":"):
        return
    new_ld = f"{libdir}:{cur}" if cur else libdir
    if os.environ.get("_FINAGENT_LD_LIBRARY_PATH_REEXEC") == "1":
        return
    os.environ["LD_LIBRARY_PATH"] = new_ld
    os.environ["_FINAGENT_LD_LIBRARY_PATH_REEXEC"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
_ensure_conda_lib()

# 3) CLI parse
ap = argparse.ArgumentParser()
ap.add_argument("concept", nargs="?", default="Mean Reversion after Panic Selling")
ap.add_argument("--train-start", required=True)
ap.add_argument("--train-end", required=True)
ap.add_argument("--val-start", required=True)
ap.add_argument("--val-end", required=True)
ap.add_argument("--test-start", required=True)
ap.add_argument("--test-end", default="2025-12-31")
ap.add_argument("--outer-loop", type=int, default=3)
ap.add_argument("--tag", default="", help="optional tag to print, traceability")
args = ap.parse_args()

# 4) Import favor modules and override cfg
from run.config import load_rd_config
from run.main import run_outer_loop
from util.llm_tracker import get_tracker

cfg = load_rd_config()
cfg.data_split.train_start = args.train_start
cfg.data_split.train_end = args.train_end
cfg.data_split.val_start = args.val_start
cfg.data_split.val_end = args.val_end
cfg.data_split.test_start = args.test_start
cfg.data_split.test_end = args.test_end

print("=" * 60)
print("🔬 Custom Split Experiment")
print("=" * 60)
print(f"  Concept    : {args.concept}")
print(f"  Tag        : {args.tag or '(none)'}")
print(f"  Train      : {cfg.data_split.train_start} ~ {cfg.data_split.train_end}")
print(f"  Validation : {cfg.data_split.val_start} ~ {cfg.data_split.val_end}")
print(f"  Test       : {cfg.data_split.test_start} ~ {cfg.data_split.test_end}")
print(f"  Outer-loop : {args.outer_loop}")
print(f"  LLM model  : {cfg.llm.model_name}")
print(f"  Provider   : {cfg.qlib.provider_uri}")
print("=" * 60)

# 5) Run
result = run_outer_loop(
    concept=args.concept,
    cfg=cfg,
    max_outer_iterations=args.outer_loop,
)

print("\n" + "=" * 60)
print("✅ Pipeline completed successfully!")
print("=" * 60)
print(f"   Run ID: {result['run_id']}")

tracker = get_tracker()
summary = tracker.get_summary()
print(f"\n💰 LLM Usage:")
print(f"   Total API Calls: {summary['total_calls']:,}")
print(f"   Total Tokens   : {summary['total_tokens']:,}")
print(f"   Total Cost     : ${summary['total_cost_usd']:.4f} USD")
