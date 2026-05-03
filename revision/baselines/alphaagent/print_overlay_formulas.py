#!/usr/bin/env python3
"""
Print the factor/formula expressions for the sources that are overlaid in compare plots.

This is a small helper to answer:
  "When I run run_stage2_4ways.sh --compare-plots, what formulas are being overlaid?"

It inspects Stage2 outputs and the formula packager index produced by run_stage2_4ways.sh.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class SourceFormula:
    source: str
    name: str
    expr: str
    stage2_dir: str
    factor_ws: str
    note: str = ""


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_stage2_summary(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    best: Optional[Path] = None
    best_mtime = -1.0
    for p in root.rglob("stage2_summary.json"):
        try:
            mt = p.stat().st_mtime
        except Exception:
            continue
        if mt > best_mtime:
            best = p
            best_mtime = mt
    return best


def _from_stage2_dir(source: str, stage2_dir: Path) -> Optional[SourceFormula]:
    summ = stage2_dir / "stage2_summary.json"
    if not summ.exists():
        return None
    obj = _read_json(summ)
    name = str(obj.get("factor_name") or stage2_dir.name)
    expr = str(obj.get("factor_expr") or "")
    factor_ws = str(obj.get("factor_ws") or "")
    return SourceFormula(
        source=source,
        name=name,
        expr=expr,
        stage2_dir=str(stage2_dir),
        factor_ws=factor_ws,
    )


def _resolve_alphaagent(results: Path, alphaagent_factor_name: str) -> Optional[SourceFormula]:
    # run_stage2_4ways.sh writes AlphaAgent Stage2 under results/alphaagent/<factor_name>/.
    p = results / "alphaagent" / alphaagent_factor_name
    if (p / "stage2_summary.json").exists():
        return _from_stage2_dir("alphaagent", p)
    # Fallback: pick latest under results/alphaagent.
    latest = _latest_stage2_summary(results / "alphaagent")
    return _from_stage2_dir("alphaagent", latest.parent) if latest else None


def _resolve_gpt(results: Path, gpt_stage2_dir: str) -> Optional[SourceFormula]:
    # The user may pass either a Stage2 output dir (results/gpt/<uuid>) or a factor workspace.
    p = Path(gpt_stage2_dir)
    if not p.is_absolute():
        p = (results / p).resolve() if (results / p).exists() else p

    # If it's already a Stage2 output dir.
    if (p / "stage2_summary.json").exists():
        return _from_stage2_dir("gpt", p)

    # Otherwise, pick latest Stage2 output under results/gpt.
    latest = _latest_stage2_summary(results / "gpt")
    return _from_stage2_dir("gpt", latest.parent) if latest else None


def _resolve_alpha101(results: Path, alpha_id: int = 2) -> Optional[SourceFormula]:
    tag = f"alpha{alpha_id:03d}"
    root = results / "alpha101" / tag
    latest = _latest_stage2_summary(root) or _latest_stage2_summary(results / "alpha101")
    item = _from_stage2_dir("alpha101", latest.parent) if latest else None
    if item is None:
        return None
    # Alpha101 workspaces are implemented as Python functions and typically don't have a symbolic expr.
    note = f"Alpha101 is code-defined (see alpha101.py:{tag}); factor_ws/factor.py runs alpha101.{tag}()."
    return SourceFormula(
        source=item.source,
        name=item.name or f"Alpha101_{tag}",
        expr=item.expr or f"(python) alpha101.{tag}()",
        stage2_dir=item.stage2_dir,
        factor_ws=item.factor_ws,
        note=note,
    )


def _resolve_ours_formulas(results: Path, ours_stage2_dir: str) -> List[SourceFormula]:
    """
    Prefer the packager index written by run_stage2_4ways.sh:
      results/workspaces/formulas_index.json
    """
    out: List[SourceFormula] = []
    if ours_stage2_dir:
        p = Path(ours_stage2_dir)
        if not p.is_absolute():
            p = (results / p).resolve() if (results / p).exists() else p
        item = _from_stage2_dir("ours", p)
        return [item] if item else []

    idx = results / "workspaces" / "formulas_index.json"
    if idx.exists():
        rows = json.loads(idx.read_text(encoding="utf-8"))
        if isinstance(rows, list):
            for r in rows:
                if not isinstance(r, dict):
                    continue
                fid = str(r.get("formula_id") or "").strip() or str(r.get("formula_name") or "").strip() or "unknown"
                expr = str(r.get("expr") or "").strip()
                ws = str(r.get("workspace") or "").strip()
                out.append(
                    SourceFormula(
                        source="ours",
                        name=fid,
                        expr=expr,
                        stage2_dir="(varies per formula output under results/formulas/...)",
                        factor_ws=ws,
                    )
                )
            return out

    # Fallback: enumerate existing Stage2 outputs under results/formulas/**.
    for summ in sorted((results / "formulas").rglob("stage2_summary.json"), key=lambda p: str(p)):
        obj = _read_json(summ)
        name = str(obj.get("factor_name") or summ.parent.name)
        expr = str(obj.get("factor_expr") or "")
        out.append(
            SourceFormula(
                source="ours",
                name=name,
                expr=expr,
                stage2_dir=str(summ.parent),
                factor_ws=str(obj.get("factor_ws") or ""),
            )
        )
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results", help="Results root (default: results)")
    ap.add_argument(
        "--alphaagent-factor-name",
        default=os.environ.get("ALPHAAGENT_FACTOR_NAME", "Return_Volatility_Interaction_Mean_Reversion_15D"),
        help="AlphaAgent factor_name (default matches run_stage2_4ways.sh default).",
    )
    ap.add_argument(
        "--gpt-stage2-dir",
        default="",
        help="GPT Stage2 output dir (e.g. results/gpt/<uuid>). If omitted, picks latest under results/gpt.",
    )
    ap.add_argument("--alpha101-id", type=int, default=2, help="Alpha101 id fixed in the script (default: 2).")
    ap.add_argument(
        "--ours-stage2-dir",
        default="",
        help="If set, print only this OURS Stage2 output dir (results/formulas/<tag>/<uuid>). Otherwise prints formulas_index.json if available.",
    )
    ap.add_argument("--format", choices=["json", "text"], default="text")
    args = ap.parse_args(argv)

    results = Path(args.results)
    items: List[SourceFormula] = []

    aa = _resolve_alphaagent(results, args.alphaagent_factor_name)
    if aa:
        items.append(aa)

    gpt = _resolve_gpt(results, args.gpt_stage2_dir) if args.gpt_stage2_dir else _resolve_gpt(results, "results/gpt")
    if gpt:
        items.append(gpt)

    a101 = _resolve_alpha101(results, alpha_id=args.alpha101_id)
    if a101:
        items.append(a101)

    items.extend(_resolve_ours_formulas(results, args.ours_stage2_dir))

    if args.format == "json":
        print(json.dumps([item.__dict__ for item in items], ensure_ascii=False, indent=2))
        return 0

    # text
    for item in items:
        print(f"[{item.source}] {item.name}")
        if item.expr:
            print(f"  expr: {item.expr}")
        if item.factor_ws:
            print(f"  factor_ws: {item.factor_ws}")
        if item.stage2_dir:
            print(f"  stage2_dir: {item.stage2_dir}")
        if item.note:
            print(f"  note: {item.note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
