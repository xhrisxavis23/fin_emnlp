#!/usr/bin/env python3
"""
alpha101_packager: Alpha101 -> AlphaAgent-compatible factor workspace

Creates a new factor workspace directory under `git_ignore_folder/RD-Agent_workspace/<uuid>`:
  - factor.py: computes a selected alpha101 factor from daily_pv.h5 and writes result.h5
  - daily_pv.h5: symlink to a source daily_pv.h5
  - alpha101_spec.json: metadata (alpha id, name, created time, mapping notes)

Then you can run Stage2 directly:
  python stage2.py --factor-ws git_ignore_folder/RD-Agent_workspace/<uuid>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Alpha101Spec:
    alpha: str
    factor_name: str
    created_at: str
    daily_pv_source: str
    notes: list[str]


def _normalize_alpha_id(alpha: str) -> str:
    s = alpha.strip().lower().replace("#", "")
    if s.startswith("alpha"):
        s = s[5:]
    s = s.strip()
    if not s.isdigit():
        raise ValueError(f"Invalid alpha id: {alpha!r} (expected e.g. 101 or alpha101)")
    n = int(s)
    if n < 1 or n > 101:
        raise ValueError(f"Invalid alpha number: {n} (expected 1..101)")
    return f"alpha{n:03d}"


def _safe_name(name: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", name or "").strip("_")
    return out or "Alpha101Factor"


def _ensure_symlink(link_path: Path, target: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()
    except FileNotFoundError:
        pass
    link_path.symlink_to(target)


def _render_factor_py(*, alpha_id: str, factor_name: str) -> str:
    """
    Writes result.h5 with key 'data' as a (datetime,instrument) MultiIndex Series.
    """
    return f'''\
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _repo_root() -> Path:
    # This workspace can live under different roots (e.g. git_ignore_folder/... or results/...).
    # Find the repo root by walking upwards until we see stage2.py + alpha101.py.
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "stage2.py").exists() and (parent / "alpha101.py").exists():
            return parent
    # Fallback: keep the historical assumption.
    return p.parents[3]


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(f"Missing required column. Tried: {{candidates}}. Found: {{sorted(list(cols))[:30]}} ...")


def _wide(df: pd.DataFrame, col: str) -> pd.DataFrame:
    s = df[col].astype(float)
    wide = s.unstack(level="instrument").sort_index()
    wide.index.name = "datetime"
    return wide


def calculate_alpha101(alpha_id: str) -> pd.Series:
    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root))
    import alpha101  # noqa: E402

    raw = pd.read_hdf("./daily_pv.h5", key="data")
    if not isinstance(raw.index, pd.MultiIndex) or not {{"datetime", "instrument"}} <= set(raw.index.names):
        # Best-effort: align to expected names
        if isinstance(raw.index, pd.MultiIndex) and len(raw.index.names) >= 2:
            names = list(raw.index.names)
            names[0] = names[0] or "datetime"
            names[1] = names[1] or "instrument"
            raw.index = raw.index.set_names(names)
        else:
            raise ValueError("daily_pv.h5 must have a MultiIndex with datetime/instrument.")

    open_c = _pick_col(raw, ["$open", "open", "OPEN"])
    high_c = _pick_col(raw, ["$high", "high", "HIGH"])
    low_c = _pick_col(raw, ["$low", "low", "LOW"])
    close_c = _pick_col(raw, ["$close", "close", "CLOSE"])
    vol_c = _pick_col(raw, ["$volume", "volume", "VOL", "VOLUME"])
    amount_c = None
    for c in ["$amount", "amount", "$money", "money", "$turnover", "turnover"]:
        if c in raw.columns:
            amount_c = c
            break

    open_w = _wide(raw, open_c)
    high_w = _wide(raw, high_c)
    low_w = _wide(raw, low_c)
    close_w = _wide(raw, close_c)
    vol_w = _wide(raw, vol_c)

    # Returns: percent change * 100 (approx to S_DQ_PCTCHANGE style)
    ret_w = close_w.pct_change().replace([np.inf, -np.inf], np.nan) * 100.0

    # VWAP proxy:
    # - if amount exists, use amount/volume
    # - else fall back to close (still allows many alphas to run, though vwap-based ones degrade)
    if amount_c is not None:
        amt_w = _wide(raw, amount_c)
        vwap_w = (amt_w / (vol_w + 1e-12)).replace([np.inf, -np.inf], np.nan)
    else:
        vwap_w = close_w.copy()

    df_data = {{
        "S_DQ_OPEN": open_w,
        "S_DQ_HIGH": high_w,
        "S_DQ_LOW": low_w,
        "S_DQ_CLOSE": close_w,
        "S_DQ_VOLUME": vol_w,
        "S_DQ_PCTCHANGE": ret_w,
        "S_DQ_AMOUNT": (vwap_w * vol_w),
    }}

    stock = alpha101.Alphas(df_data)
    fn = getattr(stock, alpha_id)
    out = fn()

    if isinstance(out, pd.DataFrame):
        s = out.stack(dropna=False)
    else:
        s = out  # assume Series

    if not isinstance(s.index, pd.MultiIndex):
        # If it's a 1-D series indexed by datetime only, we can't use it as a factor.
        raise ValueError(f"alpha output must be wide (date x instrument) or MultiIndex Series. Got index={{type(s.index)}}")

    s.index = s.index.set_names(["datetime", "instrument"])
    s.name = {json.dumps(factor_name)}
    s = s.astype(float)
    s = s.sort_index()
    return s


if __name__ == "__main__":
    alpha_id = {json.dumps(alpha_id)}
    name = {json.dumps(factor_name)}
    result = calculate_alpha101(alpha_id)
    if os.path.exists("result.h5"):
        os.remove("result.h5")
    result.to_hdf("result.h5", key="data")
    print(f"[alpha101 factor] wrote result.h5 for {{alpha_id}} as {{name}}")
'''


def create_workspace(*, alpha_id: str, factor_name: str, workspace_root: Path, daily_pv_path: Path) -> Path:
    ws_id = uuid.uuid4().hex
    ws_dir = workspace_root / ws_id
    ws_dir.mkdir(parents=True, exist_ok=False)

    notes = [
        "Maps Qlib-style daily_pv.h5 ($open/$high/$low/$close/$volume[/amount]) to alpha101 expected fields.",
        "If amount is unavailable, VWAP is approximated as close (vwap-based alphas may degrade).",
        "Output is saved as result.h5 (key=data) MultiIndex Series (datetime,instrument).",
    ]
    spec = Alpha101Spec(
        alpha=alpha_id,
        factor_name=factor_name,
        created_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        daily_pv_source=str(daily_pv_path.resolve()),
        notes=notes,
    )
    (ws_dir / "alpha101_spec.json").write_text(json.dumps(asdict(spec), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ws_dir / "factor.py").write_text(_render_factor_py(alpha_id=alpha_id, factor_name=factor_name), encoding="utf-8")
    _ensure_symlink(ws_dir / "daily_pv.h5", daily_pv_path.resolve())
    return ws_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", required=True, help="Alpha id to package (e.g. 101 or alpha101)")
    ap.add_argument("--name", default="", help="Factor name override (default: Alpha101_<alpha>)")
    ap.add_argument(
        "--workspace-root",
        default="git_ignore_folder/RD-Agent_workspace",
        help="Where to create factor workspace dirs",
    )
    ap.add_argument(
        "--daily-pv",
        default="git_ignore_folder/factor_implementation_source_data/daily_pv.h5",
        help="Path to daily_pv.h5 to symlink into the new workspace",
    )
    ap.add_argument("--run", action="store_true", help="Run factor.py after writing workspace (produces result.h5)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    alpha_id = _normalize_alpha_id(args.alpha)
    factor_name = _safe_name(args.name or f"Alpha101_{alpha_id}")

    ws_root = Path(args.workspace_root)
    if not ws_root.is_absolute():
        ws_root = repo_root / ws_root
    daily_pv = Path(args.daily_pv)
    if not daily_pv.is_absolute():
        daily_pv = repo_root / daily_pv
    if not daily_pv.exists():
        raise SystemExit(f"daily_pv.h5 not found: {daily_pv}")

    ws_dir = create_workspace(alpha_id=alpha_id, factor_name=factor_name, workspace_root=ws_root, daily_pv_path=daily_pv)
    print(f"[alpha101_packager] wrote workspace: {ws_dir}")
    print(f"[alpha101_packager] alpha_id={alpha_id}")
    print(f"[alpha101_packager] factor_name={factor_name}")

    if args.run:
        import subprocess

        subprocess.check_call([sys.executable, "factor.py"], cwd=str(ws_dir))
        print(f"[alpha101_packager] wrote: {ws_dir / 'result.h5'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
