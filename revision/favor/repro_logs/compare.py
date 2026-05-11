"""Compare 3 reproduction runs against the original CSI500 FaVOR run (20260207_051736).

Produces a Markdown summary of each run's best combo (by IS excess IR) and its OOS metrics,
plus the cross-run dispersion that quantifies LLM-randomness sensitivity.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from statistics import mean, stdev

REVISION_RUNS_DIR = Path("/home/dgu/fin/revision/revision/favor/runs")
ORIGINAL_PATH = Path("/home/dgu/fin/01_15_new_qlib/runs/20260207_051736/specs/stage4_summary.json")

PAPER_TABLE1 = {  # CSI 500 FaVOR row from FaVOR_paper.pdf
    "AR": 0.1397, "IR": 0.6470, "MDD": -0.2224, "CR": 0.7104,
}


def load_outer_iter1(p: Path) -> dict | None:
    if not p.exists():
        return None
    j = json.load(open(p))
    if "outer_iter_1" not in j:
        return None
    return j["outer_iter_1"]


def best_combo(outer: dict) -> dict | None:
    """Pick the combo with the largest IS excess_return_with_cost.information_ratio
    — this is what the original pipeline ranks on (per aggregated_stage4_results.csv)."""
    best, best_ir = None, -1e9
    for c in outer.get("all_combinations", []):
        ir = (c.get("insample", {})
                .get("excess_return_with_cost", {})
                .get("information_ratio"))
        if ir is None: continue
        if ir > best_ir:
            best_ir, best = ir, c
    return best


def metrics_of(combo: dict, sample: str = "outsample") -> dict:
    m = combo[sample]["excess_return_with_cost"]
    return dict(
        AR=m["annualized_return"], IR=m["information_ratio"],
        MDD=m["max_drawdown"], CR_geom=m["net_return"],
    )


def fmt(d: dict) -> str:
    return f"AR={d['AR']:+.4f}  IR={d['IR']:+.4f}  MDD={d['MDD']:+.4f}  CR(geom)={d['CR_geom']:+.4f}"


def hypothesis_id(specs_dir: Path) -> str:
    h = specs_dir / "hypothesis.json"
    if not h.exists(): return "?"
    j = json.load(open(h))
    o = j.get("outer_iter_1") or j
    if isinstance(o, dict):
        if "hypotheses" in o and o["hypotheses"]:
            return o["hypotheses"][0].get("hypothesis_id", "?")
        return o.get("hypothesis_id", "?")
    return "?"


def main():
    rows = []
    # Original (frozen)
    o0 = load_outer_iter1(ORIGINAL_PATH)
    if o0:
        target = next((c for c in o0["all_combinations"] if c["combo_idx"] == 51), None)
        rows.append({
            "label": "Original (paper Table 1, combo_51)",
            "hyp": o0["hypothesis_id"],
            "n_combos": len(o0["all_combinations"]),
            "best_oos": metrics_of(target, "outsample") if target else None,
            "best_is":  metrics_of(target, "insample")  if target else None,
            "combo_idx": 51,
            "formulas": target["formula_names"] if target else [],
            "thresholds": target["optimal_thresholds"] if target else {},
        })

    # Reproductions
    if len(sys.argv) > 1:
        run_dirs = [Path(p) for p in sys.argv[1:]]
    else:
        run_dirs = sorted(REVISION_RUNS_DIR.glob("20260510_*"))
    for d in run_dirs:
        sp = d / "specs" / "stage4_summary.json"
        outer = load_outer_iter1(sp)
        if not outer:
            rows.append({"label": d.name, "hyp": "(stage4 not yet finished)", "n_combos": 0,
                         "best_oos": None, "best_is": None, "combo_idx": None,
                         "formulas": [], "thresholds": {}})
            continue
        bc = best_combo(outer)
        rows.append({
            "label": d.name,
            "hyp": outer["hypothesis_id"],
            "n_combos": len(outer["all_combinations"]),
            "best_oos": metrics_of(bc, "outsample") if bc else None,
            "best_is":  metrics_of(bc, "insample")  if bc else None,
            "combo_idx": bc["combo_idx"] if bc else None,
            "formulas": bc["formula_names"] if bc else [],
            "thresholds": bc["optimal_thresholds"] if bc else {},
        })

    print("# CSI 500 FaVOR reproduction comparison\n")
    print(f"Paper Table 1: AR={PAPER_TABLE1['AR']} IR={PAPER_TABLE1['IR']} MDD={PAPER_TABLE1['MDD']} CR={PAPER_TABLE1['CR']}\n")
    for r in rows:
        print(f"## {r['label']}")
        print(f"- hypothesis_id : `{r['hyp']}`")
        print(f"- n_combinations: {r['n_combos']}    chosen combo_idx: {r['combo_idx']}")
        print(f"- formulas      : {r['formulas']}")
        print(f"- thresholds    : {r['thresholds']}")
        if r["best_is"]:
            print(f"- IS  (excess, with cost): {fmt(r['best_is'])}")
        if r["best_oos"]:
            print(f"- OOS (excess, with cost): {fmt(r['best_oos'])}")
        print()

    # Dispersion across reproductions only
    repros = [r for r in rows if r["label"] != "Original (paper Table 1, combo_51)" and r["best_oos"]]
    if len(repros) >= 2:
        print("## Cross-run dispersion (OOS excess_with_cost)")
        for k in ("AR", "IR", "MDD", "CR_geom"):
            vals = [r["best_oos"][k] for r in repros]
            print(f"- {k:7s}: vals={['%.4f'%v for v in vals]}  mean={mean(vals):+.4f}  std={stdev(vals):+.4f}")

if __name__ == "__main__":
    main()
