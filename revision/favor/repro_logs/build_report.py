"""Generate a single-file HTML report from all FaVOR runs under revision/favor/runs/.

Re-run anytime; the script is idempotent and tolerates incomplete (in-progress) runs.

Output: revision/favor/repro_logs/report.html
"""
from __future__ import annotations

import base64
import io
import json
import pickle
import sys
from datetime import datetime
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]      # revision/favor/
RUNS_DIR = ROOT / "runs"
OUT = ROOT / "favor_dashboard.html"              # one level up from repro_logs/

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

PAPER = {
    "label": "Paper Table 1 (CSI 500, OOS-oracle combo_51)",
    "hypothesis_id": "BH_Continuation_Breakout_5D_v1",
    "n_combos": 84,
    "is_best_oos_ir": -0.5621,   # honest IS-best (combo 15) → catastrophic OOS
    "is_best_oos_ar": -0.1406,
    "oracle_oos_ir": 0.6470,     # what the paper actually printed
    "oracle_oos_ar": 0.1397,
    "oracle_oos_mdd": -0.2224,
    "oracle_oos_cr": 0.7104,
    "is_ir": 0.4388,             # combo_51's IS metrics
    "concept": (
        "After a breakout to a new high, a pullback toward the 20-day moving average "
        "often serves as support, increasing the probability of price revisiting the "
        "breakout level or exceeding it."
    ),
    "horizon": 5,
    "stop_loss": -0.10,
    "n_trials": 50,
    "model": "gpt-4o (paper)",
    "is_baseline": True,
}


def safe_json(p: Path):
    if not p.exists():
        return None
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None


def _load_report(pkl: Path):
    if not pkl.exists():
        return None
    try:
        with open(pkl, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _cumret(df, fee: bool = True):
    """qlib report_normal_1day → cumulative return Series."""
    if df is None or "return" not in df.columns:
        return None
    net = df["return"] - df["cost"] if (fee and "cost" in df.columns) else df["return"]
    return (1.0 + net).cumprod() - 1.0


def cumret_chart_b64(run_dir: Path, is_best_idx, oracle_idx,
                      is_best_iter: str = "iter_1", oracle_iter: str = "iter_1") -> str | None:
    """Render IS-best + oracle portfolio cum-ret + benchmark, return base64 PNG (or None).

    is_best_iter/oracle_iter let us read combos from different outer-loop iters."""
    if not _HAS_MPL or is_best_idx is None:
        return None
    art_root = run_dir / "qlib_artifacts"
    if not art_root.exists():
        return None

    def _norm_iter(name: str) -> str:
        # stage4_summary.json uses "outer_iter_N"; qlib_artifacts/ uses "iter_N"
        if name and name.startswith("outer_"):
            return name[len("outer_"):]
        return name or "iter_1"

    def _series(combo_idx, split, iter_name):
        if combo_idx is None:
            return None
        rep = _load_report(art_root / _norm_iter(iter_name) / f"combo_{combo_idx}" / split / "report_normal_1day.pkl")
        return _cumret(rep)

    def _bench(combo_idx, split, iter_name):
        if combo_idx is None:
            return None
        rep = _load_report(art_root / _norm_iter(iter_name) / f"combo_{combo_idx}" / split / "report_normal_1day.pkl")
        if rep is None or "bench" not in rep.columns:
            return None
        return (1.0 + rep["bench"]).cumprod() - 1.0

    is_best_is  = _series(is_best_idx, "is",  is_best_iter)
    is_best_oos = _series(is_best_idx, "oos", is_best_iter)
    same = (oracle_idx == is_best_idx) and (oracle_iter == is_best_iter)
    oracle_oos  = _series(oracle_idx, "oos", oracle_iter) if (oracle_idx and not same) else None
    bench_is    = _bench(is_best_idx, "is",  is_best_iter)
    bench_oos   = _bench(is_best_idx, "oos", is_best_iter)

    if is_best_oos is None and is_best_is is None:
        return None

    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=100)
    fig.patch.set_facecolor("#0b1220")
    ax.set_facecolor("#0b1220")

    if bench_is is not None:
        ax.plot(bench_is.index,  bench_is.values,  color="#64748b", lw=1.0, alpha=0.7, label="benchmark (IS)")
    if bench_oos is not None:
        ax.plot(bench_oos.index, bench_oos.values, color="#94a3b8", lw=1.0, alpha=0.9, label="benchmark (OOS)")

    _ib_lbl = f"IS-best #{is_best_idx} @{_norm_iter(is_best_iter)}"
    _or_lbl = f"oracle #{oracle_idx} @{_norm_iter(oracle_iter)}"
    if is_best_is is not None:
        ax.plot(is_best_is.index,  is_best_is.values,  color="#f59e0b", lw=1.4, alpha=0.6, label=f"{_ib_lbl} (IS)")
    if is_best_oos is not None:
        ax.plot(is_best_oos.index, is_best_oos.values, color="#f97316", lw=2.0,            label=f"{_ib_lbl} (OOS)")
    if oracle_oos is not None:
        ax.plot(oracle_oos.index,  oracle_oos.values,  color="#a78bfa", lw=1.6, ls="--",   label=f"{_or_lbl} (OOS)")

    # IS/OOS boundary
    if is_best_oos is not None and len(is_best_oos) > 0:
        ax.axvline(is_best_oos.index[0], color="#475569", lw=0.8, ls=":", alpha=0.7)
        ax.text(is_best_oos.index[0], ax.get_ylim()[1], "  OOS →",
                color="#94a3b8", fontsize=9, va="top")

    ax.axhline(0, color="#475569", lw=0.6, ls="-", alpha=0.6)
    ax.set_title("Cumulative excess return (net of cost)", color="#e2e8f0", fontsize=11, loc="left")
    ax.set_ylabel("cumret", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, color="#1e293b", lw=0.5)
    ax.legend(loc="upper left", fontsize=8, facecolor="#1e293b", edgecolor="#334155", labelcolor="#cbd5e1", framealpha=0.85)
    fig.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=100)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def metrics(combo: dict, sample: str) -> dict | None:
    if not combo:
        return None
    s = combo.get(sample) or {}
    m = s.get("excess_return_with_cost") or {}
    if not m:
        return None
    return {
        "AR": m.get("annualized_return"),
        "IR": m.get("information_ratio"),
        "MDD": m.get("max_drawdown"),
        "CR": m.get("net_return"),
        "mean": m.get("mean"),
        "std": m.get("std"),
    }


def _iter_keys_sorted(d: dict) -> list[str]:
    """Return outer_iter_* keys sorted by N."""
    ks = [k for k in d.keys() if k.startswith("outer_iter_")]
    def _n(k):
        try:
            return int(k.split("_")[-1])
        except Exception:
            return 0
    return sorted(ks, key=_n)


def _combo_is_ir(c):
    return (c.get("insample", {}).get("excess_return_with_cost", {}) or {}).get("information_ratio", -1e9)


def _combo_oos_ir(c):
    return (c.get("outsample", {}).get("excess_return_with_cost", {}) or {}).get("information_ratio", -1e9)


def _iter_summary(combos: list[dict]) -> dict:
    """Compute per-iter aggregate stats."""
    if not combos:
        return {"n_combos": 0}
    bis = max(combos, key=_combo_is_ir)
    bos = max(combos, key=_combo_oos_ir)
    oos_irs = [_combo_oos_ir(c) for c in combos if _combo_oos_ir(c) > -1e8]
    out = {
        "n_combos": len(combos),
        "is_best_combo": bis.get("combo_idx"),
        "oracle_combo":  bos.get("combo_idx"),
        "is_best_formulas":   bis.get("formula_names"),
        "is_best_thresholds": bis.get("optimal_thresholds"),
        "oracle_formulas":    bos.get("formula_names"),
        "oracle_thresholds":  bos.get("optimal_thresholds"),
        "is_best_is":  metrics(bis, "insample"),
        "is_best_oos": metrics(bis, "outsample"),
        "oracle_is":   metrics(bos, "insample"),
        "oracle_oos":  metrics(bos, "outsample"),
    }
    if oos_irs:
        out.update({
            "oos_ir_min":    min(oos_irs),
            "oos_ir_max":    max(oos_irs),
            "oos_ir_median": sorted(oos_irs)[len(oos_irs) // 2],
            "oos_ir_pos":    sum(1 for x in oos_irs if x > 0),
            "oos_ir_total":  len(oos_irs),
            "oos_pos_frac":  sum(1 for x in oos_irs if x > 0) / len(oos_irs),
        })
    # Top 10 by OOS IR
    ranked = sorted(combos, key=_combo_oos_ir, reverse=True)
    out["top_combos"] = [{
        "combo_idx":  c.get("combo_idx"),
        "formulas":   c.get("formula_names"),
        "thresholds": c.get("optimal_thresholds"),
        "is":  metrics(c, "insample"),
        "oos": metrics(c, "outsample"),
    } for c in ranked[:10]]
    return out


# ────────────────────────────────────────────────────────────────────
# Sweep / Concept classification helpers
# ────────────────────────────────────────────────────────────────────

# Concept text → short label mapping (substring match, first hit wins)
_CONCEPT_SIGS = [
    ("paper",      "After a breakout to a new high"),
    ("uptrend",    "In a strong uptrend, when price pulls back"),
    ("panic",      "After a sharp sell-off"),
    ("compressed", "Following a prior price decline"),
    ("volcomp",    "compressed volatility"),
]

def classify_concept(concept_text: str | None) -> str | None:
    if not concept_text:
        return None
    for short, sig in _CONCEPT_SIGS:
        if sig.lower() in concept_text.lower():
            return short
    return None


# label prefix → phase mapping
def classify_phase(label: str) -> str | None:
    if not label:
        return None
    head = label.split("_")[0]
    if head.startswith("A") and head[1:].isdigit():
        return "Phase 0 (5/10 mini × S5 × ol=1)"
    if head.startswith("B0"):  # B01..B08
        return "Phase 0 (5/10 mini × S5 × ol=1)"
    if head in ("B1", "B2", "B3", "B4"):
        return "Phase 5 (5/12 mini × S1 × Stage 3/4 Cartesian)"
    if head.startswith("M"):
        return "Phase 4 (5/12 mini × S1 × n_trials=20)"
    if head.startswith("N"):
        return "Phase ? (N-series, 별도 sweep)"
    # multi-split S1/S2/S5 in label
    if "_S1" in label or "_S2" in label or "_S5" in label:
        if "gpt4o" in label.lower():
            return "Phase 3 (5/12 4o × S1 × ol=3)"
        return "Phase 2 (5/11 mini × S1/S2/S5 × ol=3)"
    return None


def parse_phase5_cell(label: str) -> dict:
    """Parse Phase 5 label like 'B1_paper_h10_s05_e0_pr04_t55' into cell metadata."""
    out = {}
    tokens = label.split("_")
    if not tokens or tokens[0] not in ("B1", "B2", "B3", "B4"):
        return out
    out["base"] = "_".join(tokens[:3]) if len(tokens) >= 3 else tokens[0]
    for tok in tokens[3:]:
        if tok == "s05":   out["cell_stop"] = "−0.05"
        elif tok == "sN":  out["cell_stop"] = "None (손절 없음)"
        elif tok == "e0":  out["cell_entry"] = "none"
        elif tok == "e1":  out["cell_entry"] = "up_day_and_close_pos"
        elif tok in ("pr04", "pr05", "pr06"):
            out["cell_pass_rate"] = f"0.{tok[3]}"   # pr04 → 0.4, pr06 → 0.6
        elif tok == "t55":
            out["cell_thr_min"] = "0.55"
        elif tok == "t70":
            out["cell_thr_min"] = "0.70"
        elif tok.startswith("t") and tok[1:].isdigit():
            out["cell_thr_min"] = f"0.{tok[1:]}"   # fallback for other t-values
    return out


def collect_run(d: Path) -> dict:
    info = {"run_id": d.name, "status": "incomplete", "path": str(d.relative_to(ROOT.parent.parent))}

    # Extract label from run_id (format: YYYYMMDD_HHMMSS_<label>)
    rid_parts = d.name.split("_", 2)
    label = rid_parts[2] if len(rid_parts) >= 3 else None
    info["label"]        = label
    info["phase"]        = classify_phase(label) if label else None
    if label and label.split("_")[0] in ("B1", "B2", "B3", "B4"):
        info.update(parse_phase5_cell(label))

    cfg = safe_json(d / "run_config.json")
    if cfg:
        info["concept"] = cfg.get("concept", "")
        info["concept_short"] = classify_concept(info["concept"])
        c = cfg.get("config", {})
        s4 = c.get("stage4", {})
        s3 = c.get("stage3", {})
        llm = c.get("llm", {})
        ds = cfg.get("data_split", {})
        info.update({
            "model": llm.get("model_name"),
            "temperature": llm.get("temperature"),
            "horizon": s4.get("horizon_days"),
            "stop_loss": s4.get("stop_loss_threshold"),
            "n_trials": s4.get("n_trials"),
            "threshold_min": s4.get("threshold_min"),
            "threshold_max": s4.get("threshold_max"),
            "entry_confirm": s4.get("entry_confirm_rule"),
            "native_strategy": s4.get("native_strategy"),
            "combo_pass_rate": s3.get("combination_pass_rate_threshold"),
            "train_period": f"{ds.get('train_start')} ~ {ds.get('train_end')}" if ds.get("train_start") else None,
            "val_period":   f"{ds.get('val_start')} ~ {ds.get('val_end')}"     if ds.get("val_start")   else None,
            "test_period":  f"{ds.get('test_start')} ~ {ds.get('test_end')}"   if ds.get("test_start")  else None,
        })

    hyp_all  = safe_json(d / "specs" / "hypothesis.json") or {}
    fb_all   = safe_json(d / "specs" / "formula_bundle.json") or {}
    s4s_all  = safe_json(d / "specs" / "stage4_summary.json") or {}

    iter_keys = _iter_keys_sorted(s4s_all)
    if not iter_keys:
        iter_keys = _iter_keys_sorted(hyp_all)  # incomplete run; fall back to hypothesis iters
    info["iters"] = []
    for ik in iter_keys:
        s4_iter = s4s_all.get(ik, {}) if isinstance(s4s_all.get(ik), dict) else {}
        combos = s4_iter.get("all_combinations", []) if s4_iter else []
        iter_entry = {
            "iter": ik,
            "iter_num": int(ik.split("_")[-1]) if ik.split("_")[-1].isdigit() else None,
            "hypothesis_id": s4_iter.get("hypothesis_id"),
        }
        # hypothesis metadata for this iter
        hyp_iter = hyp_all.get(ik, {}) if isinstance(hyp_all.get(ik), dict) else {}
        if hyp_iter.get("hypotheses"):
            h = hyp_iter["hypotheses"][0]
            iter_entry["hypothesis_id"]   = iter_entry["hypothesis_id"] or h.get("hypothesis_id")
            iter_entry["hypothesis_name"] = h.get("hypothesis_name")
            iter_entry["behavioral"]      = h.get("behavioral_description")
        # formulas for this iter
        fb_iter = fb_all.get(ik, {}) if isinstance(fb_all.get(ik), dict) else {}
        if fb_iter:
            iter_entry["formulas"]     = fb_iter.get("formulas", [])
            iter_entry["observations"] = fb_iter.get("observation_descriptions", [])
        # stage4 aggregate
        iter_entry.update(_iter_summary(combos))
        info["iters"].append(iter_entry)

    # ─── select "best iter": iter whose IS-best combo has the highest IS-IR ──
    best_iter_entry = None
    best_score = -1e18
    for it in info["iters"]:
        ib_is = (it.get("is_best_is") or {}).get("IR")
        if ib_is is not None and ib_is > best_score:
            best_score = ib_is
            best_iter_entry = it
    if best_iter_entry is None and info["iters"]:
        # all iters lack IS metrics (e.g. b200-style skip) — fall back to oracle OOS IR
        for it in info["iters"]:
            or_oos = (it.get("oracle_oos") or {}).get("IR")
            if or_oos is not None and or_oos > best_score:
                best_score = or_oos
                best_iter_entry = it

    if best_iter_entry:
        info["best_iter"]   = best_iter_entry["iter"]
        info["status"]      = "complete"
        info["n_combos"]    = best_iter_entry.get("n_combos")
        info["hypothesis_id"]   = best_iter_entry.get("hypothesis_id")
        info["hypothesis_name"] = best_iter_entry.get("hypothesis_name")
        info["behavioral"]      = best_iter_entry.get("behavioral")
        info["formulas"]        = best_iter_entry.get("formulas", [])
        info["observations"]    = best_iter_entry.get("observations", [])
        for k in ("is_best_combo", "oracle_combo",
                  "is_best_formulas", "is_best_thresholds",
                  "oracle_formulas", "oracle_thresholds",
                  "is_best_is", "is_best_oos", "oracle_is", "oracle_oos",
                  "oos_ir_min", "oos_ir_max", "oos_ir_median",
                  "oos_ir_pos", "oos_ir_total", "oos_pos_frac",
                  "top_combos"):
            if k in best_iter_entry:
                info[k] = best_iter_entry[k]
    elif info["iters"]:
        info["status"] = "stage4_empty"
    # else: status remains "incomplete"

    # Verdict
    info["verdict"] = compute_verdict(info)
    return info


def compute_verdict(info: dict) -> dict:
    """Returns dict with class, label, sub_label."""
    if info.get("status") == "incomplete":
        return {"cls": "v-incomplete", "label": "INCOMPLETE", "sub": "no stage4_summary yet"}
    if info.get("status") == "stage4_empty":
        return {"cls": "v-incomplete", "label": "NO COMBOS", "sub": "Stage 2/3 produced no qualifying combos"}

    oracle_ir = (info.get("oracle_oos") or {}).get("IR")
    is_best_oos_ir = (info.get("is_best_oos") or {}).get("IR")

    if oracle_ir is None:
        return {"cls": "v-incomplete", "label": "INCOMPLETE", "sub": "no metrics"}

    paper_ir = PAPER["oracle_oos_ir"]

    if oracle_ir >= paper_ir:
        cls, label = "v-beats", "BEATS PAPER"
    elif oracle_ir >= 0.5:
        cls, label = "v-strong", "STRONG"
    elif oracle_ir >= 0.2:
        cls, label = "v-moderate", "MODERATE OOS"
    elif oracle_ir > 0:
        cls, label = "v-weak", "WEAK +OOS"
    else:
        cls, label = "v-bad", "OOS NEGATIVE"

    sub_parts = [f"oracle IR={oracle_ir:+.3f}"]
    if is_best_oos_ir is not None:
        if is_best_oos_ir > 0:
            sub_parts.append(f"IS-best OOS IR={is_best_oos_ir:+.3f} (ROBUST!)")
        else:
            sub_parts.append(f"IS-best OOS IR={is_best_oos_ir:+.3f}")
    return {"cls": cls, "label": label, "sub": " · ".join(sub_parts)}


def fmt_num(v, fmt="{:+.4f}"):
    if v is None:
        return "—"
    try:
        return fmt.format(v)
    except Exception:
        return str(v)


HTML_HEAD = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>FaVOR dashboard</title>
<style>
:root {
  --bg: #0f172a; --bg2: #1e293b; --card: #1e293b; --fg: #e2e8f0; --muted: #94a3b8;
  --accent: #60a5fa; --good: #22c55e; --warn: #eab308; --bad: #ef4444; --paper: #a78bfa;
  --border: #334155;
}
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans KR", sans-serif;
       margin: 0; background: var(--bg); color: var(--fg); padding: 24px; }
h1 { margin: 0 0 4px; }
header { margin-bottom: 24px; }
header .meta { color: var(--muted); font-size: 13px; }
.paper-baseline {
  background: linear-gradient(135deg, #4c1d95, #312e81); padding: 14px 18px;
  border-radius: 8px; margin-top: 12px; border-left: 4px solid var(--paper);
}
.paper-baseline strong { color: #ddd6fe; }
.paper-baseline .nums { font-family: ui-monospace, monospace; font-size: 14px; margin-top: 6px; }
.paper-baseline .caveat { color: #c4b5fd; font-size: 12px; margin-top: 6px; }

.controls { display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; align-items: center; }
.controls input, .controls select {
  background: var(--bg2); color: var(--fg); border: 1px solid var(--border);
  padding: 8px 12px; border-radius: 6px; font-size: 14px;
}
.controls .summary { color: var(--muted); font-size: 13px; }

table { width: 100%; border-collapse: collapse; background: var(--card); border-radius: 8px; overflow: hidden; }
thead { background: #0f172a; position: sticky; top: 0; }
th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); font-size: 13px; }
th { font-weight: 600; cursor: pointer; user-select: none; }
th:hover { background: #334155; }
th .sort-ind { color: var(--muted); margin-left: 4px; font-size: 10px; }
tbody tr { cursor: pointer; }
tbody tr:hover { background: #334155; }
tbody tr.expanded { background: #1e3a5f; }

.grp { background: #0b1220; }
.grp-is     { border-left: 1px solid #1e293b; }
.grp-is.first { border-left: 2px solid #475569; }
.grp-oracle { border-left: 2px solid #6d28d9; }
thead .grp { text-align: center; color: var(--accent); font-size: 11px; letter-spacing: 0.5px; }
thead .grp-oracle { color: #c4b5fd; }
.verdict { display: inline-block; padding: 3px 10px; border-radius: 4px; font-size: 11px; font-weight: 700; letter-spacing: 0.3px; white-space: nowrap; }
.v-beats     { background: #14532d; color: #86efac; }
.v-strong    { background: #422006; color: #fde68a; }
.v-moderate  { background: #1e3a8a; color: #93c5fd; }
.v-weak      { background: #1e293b; color: #cbd5e1; border: 1px solid #475569; }
.v-bad       { background: #7f1d1d; color: #fca5a5; }
.v-incomplete{ background: #312e81; color: #c7d2fe; }
.v-paper     { background: #4c1d95; color: #ddd6fe; }

.num { font-family: ui-monospace, monospace; }
.num.pos { color: var(--good); }
.num.neg { color: var(--bad); }
.muted { color: var(--muted); }
.tag { display: inline-block; background: var(--bg2); border: 1px solid var(--border); padding: 1px 6px; border-radius: 3px; font-size: 11px; margin-right: 4px; font-family: ui-monospace, monospace; }

.detail-row { background: #0b1220 !important; cursor: default !important; }
.detail-row td { padding: 16px 24px; }
.detail-grid { display: grid; grid-template-columns: 280px 1fr; gap: 12px 20px; font-size: 13px; }
.detail-grid > .k { color: var(--muted); }
.detail-grid > .v { font-family: ui-monospace, monospace; word-break: break-word; }
.detail-section { margin-top: 18px; padding-top: 14px; border-top: 1px solid var(--border); }
.detail-section h3 { margin: 0 0 10px; font-size: 14px; color: var(--accent); }
.combo-table { font-size: 11px; }
.combo-table th, .combo-table td { padding: 4px 8px; }
.formula-list li { margin-bottom: 4px; font-family: ui-monospace, monospace; font-size: 12px; }
.behavioral { font-style: italic; color: var(--muted); padding: 8px 12px; background: var(--bg2); border-left: 3px solid var(--accent); border-radius: 4px; max-width: 800px; }
</style>
</head>
<body>
"""

HTML_TAIL = """
<script>
const data = window.__RUNS__;
const tbody = document.querySelector('#runs-table tbody');
const filterInput = document.querySelector('#filter');
const sortSel = document.querySelector('#sort-by');
const summary = document.querySelector('#summary');

let sortKey = 'oracle_oos_ir';
let sortDir = -1;  // -1 desc, 1 asc

function num(v) { return v === null || v === undefined ? null : Number(v); }
function fmt(v, digits=4, sign=true) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  const s = (sign && v >= 0) ? '+' : '';
  return s + Number(v).toFixed(digits);
}
function cls(v) {
  if (v === null || v === undefined) return '';
  return v >= 0 ? 'pos' : 'neg';
}

function expand(idx) {
  const tr = tbody.children[idx*2];
  const dr = tbody.children[idx*2 + 1];
  if (!dr) return;
  const isHidden = dr.style.display === 'none';
  dr.style.display = isHidden ? '' : 'none';
  tr.classList.toggle('expanded', isHidden);
}

function render() {
  // filter
  const f = filterInput.value.toLowerCase().trim();
  let rows = data.filter(r => {
    if (!f) return true;
    return JSON.stringify(r).toLowerCase().includes(f);
  });
  // sort
  rows.sort((a,b) => {
    const av = num(a[sortKey]); const bv = num(b[sortKey]);
    if (av === null && bv === null) return 0;
    if (av === null) return 1;
    if (bv === null) return -1;
    return (av - bv) * sortDir;
  });
  summary.textContent = `${rows.length} runs · sorted by ${sortKey} ${sortDir<0?'↓':'↑'}`;
  tbody.innerHTML = '';
  rows.forEach((r, i) => {
    const tr = document.createElement('tr');
    tr.className = r.is_baseline ? 'paper-row' : '';
    tr.onclick = () => expand(i);
    tr.innerHTML = `
      <td><strong>${escape(r.label || r.run_id || '')}</strong>
          <div class="muted" style="font-size:11px">${escape(r.hypothesis_id || '')}</div></td>
      <td>${r.n_combos ?? '—'}</td>
      <td class="grp grp-is first"><span class="num ${cls(r.is_best_oos_ir)}">${fmt(r.is_best_oos_ir,4)}</span></td>
      <td class="grp grp-is"><span class="num ${cls(r.is_best_oos_ar)}">${fmt(r.is_best_oos_ar,4)}</span></td>
      <td class="grp grp-is"><span class="num">${fmt(r.is_best_oos_mdd,4,false)}</span></td>
      <td class="grp grp-is"><span class="num ${cls(r.is_best_oos_cr)}">${fmt(r.is_best_oos_cr,4)}</span></td>
      <td class="grp grp-oracle"><span class="num ${cls(r.oracle_oos_ir)}">${fmt(r.oracle_oos_ir,4)}</span></td>
      <td class="grp grp-oracle"><span class="num ${cls(r.oracle_oos_ar)}">${fmt(r.oracle_oos_ar,4)}</span></td>
      <td class="grp grp-oracle"><span class="num">${fmt(r.oracle_oos_mdd,4,false)}</span></td>
      <td class="grp grp-oracle"><span class="num ${cls(r.oracle_oos_cr)}">${fmt(r.oracle_oos_cr,4)}</span></td>
      <td>${r.oos_pos_frac !== null && r.oos_pos_frac !== undefined ? (r.oos_pos_frac*100).toFixed(0)+'%' : '—'}</td>
      <td><span class="verdict ${r.is_baseline ? 'v-paper' : (r.verdict_cls||'v-incomplete')}">${escape(r.is_baseline ? 'PAPER REF' : (r.verdict_label||'?'))}</span></td>
      <td><span class="muted" style="font-family:ui-monospace,monospace;font-size:11px">${escape(r.run_id||'')}</span></td>
    `;
    tbody.appendChild(tr);
    const dr = document.createElement('tr');
    dr.className = 'detail-row';
    dr.style.display = 'none';
    dr.innerHTML = `<td colspan="13">${r.detail_html || '<span class="muted">(no details)</span>'}</td>`;
    tbody.appendChild(dr);
  });
}

function escape(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

document.querySelectorAll('th[data-key]').forEach(th => {
  th.onclick = () => {
    const k = th.dataset.key;
    if (sortKey === k) sortDir *= -1;
    else { sortKey = k; sortDir = -1; }
    document.querySelectorAll('.sort-ind').forEach(s => s.textContent = '');
    th.querySelector('.sort-ind').textContent = sortDir < 0 ? '↓' : '↑';
    render();
  };
});
filterInput.oninput = render;

render();
</script>
</body>
</html>
"""


def render_run_detail(r: dict) -> str:
    """Returns inner HTML for the expanded detail row."""
    sections = []

    # ─── Sweep 분류 (이번 run 의 phase / base / cell 변수) ──────────
    classify_items = [
        ("Label", r.get("label")),
        ("Phase / Sweep", r.get("phase")),
        ("Concept (short)", r.get("concept_short")),
        ("Base setting", r.get("base")),
        ("Cell — stop_loss", r.get("cell_stop")),
        ("Cell — entry_confirm", r.get("cell_entry")),
        ("Cell — combo_pass_rate", r.get("cell_pass_rate")),
        ("Cell — threshold_min", r.get("cell_thr_min")),
    ]
    if any(v is not None for _, v in classify_items):
        classify_html = "<div class='detail-grid'>" + "".join(
            f"<div class='k'>{escape(str(k))}</div><div class='v'>{escape(str(v if v is not None else '—'))}</div>"
            for k, v in classify_items
        ) + "</div>"
        sections.append(("Sweep 분류 (label 해석)", classify_html))

    # ─── 기본 (model, temperature, dates) ──────────────────────────
    basic_items = [
        ("LLM model", r.get("model")),
        ("temperature (config; actual hardcoded)", r.get("temperature")),
        ("Train (IS, Stage 2/3)", r.get("train_period")),
        ("Val (Optuna)",          r.get("val_period")),
        ("Test (OOS)",            r.get("test_period")),
        ("Run ID", r.get("run_id")),
    ]
    basic_html = "<div class='detail-grid'>" + "".join(
        f"<div class='k'>{escape(str(k))}</div><div class='v'>{escape(str(v if v is not None else '—'))}</div>"
        for k, v in basic_items
    ) + "</div>"
    sections.append(("기본 세팅 (model / 기간)", basic_html))

    # ─── Stage 1 (concept; hypothesis horizon은 LLM 결정이지만 env override 시 효과) ─
    stage1_items = [
        ("Concept short", r.get("concept_short")),
        ("Concept text (CLI 입력)", (r.get("concept") or "")[:200] + ("..." if len(r.get("concept") or "") > 200 else "")),
        ("horizon_days (Stage 1 hypothesis + Stage 4)", r.get("horizon")),
    ]
    stage1_html = "<div class='detail-grid'>" + "".join(
        f"<div class='k'>{escape(str(k))}</div><div class='v'>{escape(str(v if v is not None else '—'))}</div>"
        for k, v in stage1_items
    ) + "</div>"
    sections.append(("Stage 1 세팅 (LLM hypothesis 입력)", stage1_html))

    # ─── Stage 3 (combo filter) ────────────────────────────────────
    stage3_items = [
        ("combination_pass_rate_threshold", r.get("combo_pass_rate")),
    ]
    stage3_html = "<div class='detail-grid'>" + "".join(
        f"<div class='k'>{escape(str(k))}</div><div class='v'>{escape(str(v if v is not None else '—'))}</div>"
        for k, v in stage3_items
    ) + "</div>"
    sections.append(("Stage 3 세팅 (combo filter)", stage3_html))

    # ─── Stage 4 (Optuna + backtest) ───────────────────────────────
    stage4_items = [
        ("n_trials (Optuna)", r.get("n_trials")),
        ("threshold range", f"[{r.get('threshold_min')}, {r.get('threshold_max')}]" if r.get("threshold_min") is not None else None),
        ("stop_loss_threshold", r.get("stop_loss")),
        ("entry_confirm_rule", r.get("entry_confirm")),
        ("native_strategy", r.get("native_strategy")),
    ]
    stage4_html = "<div class='detail-grid'>" + "".join(
        f"<div class='k'>{escape(str(k))}</div><div class='v'>{escape(str(v if v is not None else '—'))}</div>"
        for k, v in stage4_items
    ) + "</div>"
    sections.append(("Stage 4 세팅 (Optuna + 백테스트)", stage4_html))

    # Best-iter notice
    if r.get("best_iter") and len(r.get("iters", [])) > 1:
        sections.append((
            f"Outer-loop: best = {r['best_iter']} (out of {len(r['iters'])} iters)",
            "<div class='muted' style='font-size:12px;'>"
            "메인 row 의 metric 들은 outer-loop 의 여러 iter 중 IS-IR 가 가장 높은 iter 의 결과. "
            "아래 'Outer-loop iter breakdown' 섹션에서 모든 iter 의 결과를 비교할 수 있어."
            "</div>"
        ))

    # Cumulative return chart (IS-best + oracle + benchmark)
    if r.get("cumret_b64"):
        sections.append((
            "Cumulative return — IS-best vs oracle vs benchmark",
            f"<div><img src='data:image/png;base64,{r['cumret_b64']}' "
            "style='width:100%; max-width:980px; border-radius:6px; border:1px solid #334155;'/></div>"
            "<div class='muted' style='font-size:11px; margin-top:6px;'>"
            "IS 구간 (옅은 색) + OOS 구간 (진한 색). 점선=paper 의 oracle combo (OOS IR 1위) cum-ret. "
            "Y축은 net excess return (수익률 − 거래비용). 회색=benchmark."
            "</div>"
        ))

    # Concept
    if r.get("concept"):
        sections.append(("Concept (CLI 입력)", f"<div class='behavioral'>{escape(r['concept'])}</div>"))

    # Hypothesis
    if r.get("behavioral"):
        sections.append(
            ("LLM 생성 가설",
             f"<div><strong>{escape(r.get('hypothesis_id', '?'))}</strong> · <em>{escape(r.get('hypothesis_name',''))}</em></div>"
             f"<div class='behavioral'>{escape(r['behavioral'])}</div>")
        )

    # Observations + Formulas
    if r.get("observations"):
        obs_html = "<ul class='formula-list'>" + "".join(
            f"<li><strong>{escape(o.get('observation_id',''))}</strong>: {escape(o.get('description',''))}</li>"
            for o in r["observations"]
        ) + "</ul>"
        sections.append(("관측 (Stage 1)", obs_html))

    if r.get("formulas"):
        f_html = "<ul class='formula-list'>" + "".join(
            f"<li><span class='tag'>{escape(f.get('name',''))}</span> "
            f"<span class='muted'>[{escape(f.get('observation_id',''))}, {escape(f.get('polarity',''))}]</span> "
            f"<code>{escape(f.get('definition',''))}</code></li>"
            for f in r["formulas"]
        ) + "</ul>"
        sections.append(("Formula 정의 (Stage 1)", f_html))

    # Top combos table
    if r.get("top_combos"):
        rows = []
        for c in r["top_combos"]:
            isd = c.get("is") or {}
            oos = c.get("oos") or {}
            rows.append(
                "<tr>"
                f"<td>{c.get('combo_idx','?')}</td>"
                f"<td>{escape(', '.join(c.get('formulas') or []))}</td>"
                f"<td class='num'>{fmt_num(isd.get('IR'))}</td>"
                f"<td class='num {('pos' if (oos.get('IR') or 0)>=0 else 'neg')}'>{fmt_num(oos.get('IR'))}</td>"
                f"<td class='num {('pos' if (oos.get('AR') or 0)>=0 else 'neg')}'>{fmt_num(oos.get('AR'))}</td>"
                f"<td class='num'>{fmt_num(oos.get('MDD'))}</td>"
                f"<td class='num'>{fmt_num(oos.get('CR'))}</td>"
                "</tr>"
            )
        combos_html = (
            "<table class='combo-table'><thead><tr>"
            "<th>combo_idx</th><th>formulas</th><th>IS IR</th><th>OOS IR</th><th>OOS AR</th><th>OOS MDD</th><th>OOS CR</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        )
        sections.append(("Top 10 combos by OOS IR (paper의 selection 방식)", combos_html))

    # OOS distribution
    if r.get("oos_ir_total"):
        dist_html = (
            f"<div class='detail-grid'>"
            f"<div class='k'>min OOS IR</div><div class='v num {('pos' if (r.get('oos_ir_min') or 0)>=0 else 'neg')}'>{fmt_num(r.get('oos_ir_min'))}</div>"
            f"<div class='k'>median OOS IR</div><div class='v num {('pos' if (r.get('oos_ir_median') or 0)>=0 else 'neg')}'>{fmt_num(r.get('oos_ir_median'))}</div>"
            f"<div class='k'>max OOS IR (oracle)</div><div class='v num {('pos' if (r.get('oos_ir_max') or 0)>=0 else 'neg')}'>{fmt_num(r.get('oos_ir_max'))}</div>"
            f"<div class='k'>positive OOS combos</div><div class='v'>{r.get('oos_ir_pos')}/{r.get('oos_ir_total')} ({r.get('oos_pos_frac',0)*100:.0f}%)</div>"
            f"</div>"
        )
        sections.append(("OOS IR 분포 (전체 combo 기준)", dist_html))

    # Comparison vs paper
    paper_oracle = PAPER["oracle_oos_ir"]
    cmp_lines = []
    if r.get("oracle_oos") and r["oracle_oos"].get("IR") is not None:
        diff = r["oracle_oos"]["IR"] - paper_oracle
        cls_ = "pos" if diff >= 0 else "neg"
        cmp_lines.append(f"<li>OOS-oracle IR <strong class='num {cls_}'>{fmt_num(r['oracle_oos']['IR'])}</strong> vs paper +0.6470 → diff <span class='num {cls_}'>{fmt_num(diff)}</span></li>")
    if r.get("is_best_oos") and r["is_best_oos"].get("IR") is not None:
        cls_ = "pos" if r["is_best_oos"]["IR"] >= 0 else "neg"
        cmp_lines.append(
            f"<li>IS-best (honest selection) OOS IR <strong class='num {cls_}'>{fmt_num(r['is_best_oos']['IR'])}</strong> "
            f"<span class='muted'>(paper의 IS-best는 -0.5621로 OOS 붕괴)</span></li>"
        )
    if cmp_lines:
        sections.append(("vs Paper Table 1", "<ul>" + "".join(cmp_lines) + "</ul>"))

    # ─── Outer-loop iter breakdown ────────────────────────────────────────
    iters = r.get("iters", []) or []
    if len(iters) >= 1:
        # Summary table: one row per iter
        head = (
            "<tr><th>iter</th><th>★</th><th>hypothesis</th><th>#combos</th>"
            "<th class='grp grp-is'>IS-best OOS IR</th>"
            "<th class='grp grp-is'>OOS AR</th>"
            "<th class='grp grp-oracle'>oracle OOS IR</th>"
            "<th class='grp grp-oracle'>OOS AR</th>"
            "<th>+OOS</th></tr>"
        )
        body = []
        for it in iters:
            is_best = it.get("is_best_oos") or {}
            ora     = it.get("oracle_oos") or {}
            star = "★" if it.get("iter") == r.get("best_iter") else ""
            pos_frac = it.get("oos_pos_frac")
            pos_str = f"{pos_frac*100:.0f}%" if pos_frac is not None else "—"
            body.append(
                f"<tr><td><strong>{escape(it.get('iter','?'))}</strong></td>"
                f"<td>{star}</td>"
                f"<td><code>{escape(it.get('hypothesis_id') or '—')}</code></td>"
                f"<td>{it.get('n_combos','—')}</td>"
                f"<td class='num {('pos' if (is_best.get('IR') or 0)>=0 else 'neg')}'>{fmt_num(is_best.get('IR'))}</td>"
                f"<td class='num {('pos' if (is_best.get('AR') or 0)>=0 else 'neg')}'>{fmt_num(is_best.get('AR'))}</td>"
                f"<td class='num {('pos' if (ora.get('IR') or 0)>=0 else 'neg')}'>{fmt_num(ora.get('IR'))}</td>"
                f"<td class='num {('pos' if (ora.get('AR') or 0)>=0 else 'neg')}'>{fmt_num(ora.get('AR'))}</td>"
                f"<td>{pos_str}</td></tr>"
            )
        iter_table_html = (
            "<table class='combo-table'><thead>" + head + "</thead><tbody>"
            + "".join(body) + "</tbody></table>"
            "<div class='muted' style='font-size:11px; margin-top:6px;'>"
            "★ 마크가 메인 row 에 반영된 iter (IS-best 의 IS-IR 최고). 다른 iter 는 LLM refinement 의 중간 단계."
            "</div>"
        )

        # Per-iter expanded details (cumret chart + hypothesis + formulas + top combos)
        iter_details = []
        cumret_dict = r.get("iter_cumret_b64") or {}
        for it in iters:
            blocks = []
            if cumret_dict.get(it.get("iter")):
                blocks.append(
                    f"<div><img src='data:image/png;base64,{cumret_dict[it['iter']]}' "
                    "style='width:100%; max-width:900px; border-radius:6px; border:1px solid #334155;'/></div>"
                )
            if it.get("behavioral"):
                blocks.append(
                    f"<div class='behavioral' style='margin-top:8px;'><strong>{escape(it.get('hypothesis_id') or '?')}</strong>"
                    f" — {escape(it.get('hypothesis_name') or '')}<br>{escape(it['behavioral'])}</div>"
                )
            if it.get("observations"):
                blocks.append(
                    "<details style='margin-top:6px;'><summary class='muted' style='cursor:pointer;'>observations</summary>"
                    "<ul class='formula-list'>" + "".join(
                        f"<li><strong>{escape(o.get('observation_id',''))}</strong>: {escape(o.get('description',''))}</li>"
                        for o in it["observations"]
                    ) + "</ul></details>"
                )
            if it.get("formulas"):
                blocks.append(
                    "<details style='margin-top:6px;'><summary class='muted' style='cursor:pointer;'>formulas</summary>"
                    "<ul class='formula-list'>" + "".join(
                        f"<li><span class='tag'>{escape(f.get('name',''))}</span> "
                        f"<span class='muted'>[{escape(f.get('observation_id',''))}, {escape(f.get('polarity',''))}]</span> "
                        f"<code>{escape(f.get('definition',''))}</code></li>"
                        for f in it["formulas"]
                    ) + "</ul></details>"
                )
            if it.get("top_combos"):
                rows_h = []
                for c in it["top_combos"][:5]:
                    isd = c.get("is") or {}
                    oos = c.get("oos") or {}
                    rows_h.append(
                        "<tr>"
                        f"<td>{c.get('combo_idx','?')}</td>"
                        f"<td>{escape(', '.join(c.get('formulas') or []))}</td>"
                        f"<td class='num'>{fmt_num(isd.get('IR'))}</td>"
                        f"<td class='num {('pos' if (oos.get('IR') or 0)>=0 else 'neg')}'>{fmt_num(oos.get('IR'))}</td>"
                        f"<td class='num'>{fmt_num(oos.get('AR'))}</td>"
                        "</tr>"
                    )
                blocks.append(
                    "<details style='margin-top:6px;'><summary class='muted' style='cursor:pointer;'>top 5 combos</summary>"
                    "<table class='combo-table'><thead><tr>"
                    "<th>combo</th><th>formulas</th><th>IS IR</th><th>OOS IR</th><th>OOS AR</th>"
                    "</tr></thead><tbody>" + "".join(rows_h) + "</tbody></table></details>"
                )
            star = " ★ best" if it.get("iter") == r.get("best_iter") else ""
            iter_details.append(
                f"<details style='margin-top:10px;'><summary style='cursor:pointer; font-weight:600;'>"
                f"{escape(it.get('iter','?'))}{star} — "
                f"<code>{escape(it.get('hypothesis_id') or '—')}</code></summary>"
                f"<div style='margin-top:8px; padding-left:8px; border-left:2px solid #334155;'>"
                + "".join(blocks) + "</div></details>"
            )

        sections.append((
            f"Outer-loop iter breakdown ({len(iters)} iter{'s' if len(iters) != 1 else ''})",
            iter_table_html + "<div style='margin-top:12px;'>" + "".join(iter_details) + "</div>"
        ))

    # Path hint
    sections.append(("산출물 경로", f"<code>runs/{escape(r.get('run_id',''))}/</code>"))

    out = ""
    for title, content in sections:
        out += f"<div class='detail-section'><h3>{escape(title)}</h3>{content}</div>"
    return out


def main() -> None:
    rows = []

    # Paper baseline first
    paper_row = {
        "is_baseline": True,
        "label": "Paper Table 1",
        "hypothesis_id": PAPER["hypothesis_id"],
        "n_combos": PAPER["n_combos"],
        "is_best_oos_ir": PAPER["is_best_oos_ir"],
        "is_best_oos_ar": PAPER["is_best_oos_ar"],
        "is_best_oos_mdd": None,  # paper run의 combo_15에 대한 MDD/CR은 stage4_summary에서 직접 조회해야 함
        "is_best_oos_cr": None,
        "oracle_oos_ir": PAPER["oracle_oos_ir"],
        "oracle_oos_ar": PAPER["oracle_oos_ar"],
        "oracle_oos_mdd": PAPER["oracle_oos_mdd"],
        "oracle_oos_cr": PAPER["oracle_oos_cr"],
        "oos_pos_frac": 18 / 84,
        "verdict_cls": "v-paper",
        "verdict_label": "PAPER REF",
        "run_id": "20260207_051736 (frozen)",
    }
    # Try to enrich paper row with IS-best MDD/CR from frozen stage4_summary if available
    frozen_p = Path("/home/dgu/fin/01_15_new_qlib/runs/20260207_051736/specs/stage4_summary.json")
    if frozen_p.exists():
        try:
            jp = json.load(open(frozen_p))
            op = jp.get("outer_iter_1") or {}
            combos_p = op.get("all_combinations", [])
            if combos_p:
                bis_p = max(combos_p, key=lambda c: (c.get("insample", {}).get("excess_return_with_cost", {}) or {}).get("information_ratio", -1e9))
                m_p = bis_p.get("outsample", {}).get("excess_return_with_cost", {}) or {}
                paper_row["is_best_oos_mdd"] = m_p.get("max_drawdown")
                paper_row["is_best_oos_cr"]  = m_p.get("net_return")
        except Exception:
            pass
    paper_detail = (
        "<div class='detail-section'><h3>Paper Table 1 베이스라인</h3>"
        "<div class='detail-grid'>"
        f"<div class='k'>concept</div><div class='v'>{escape(PAPER['concept'])}</div>"
        f"<div class='k'>hypothesis_id</div><div class='v'>{escape(PAPER['hypothesis_id'])}</div>"
        f"<div class='k'>n_combos</div><div class='v'>{PAPER['n_combos']}</div>"
        f"<div class='k'>OOS-oracle (paper Table 1)</div>"
        f"<div class='v'>IR={fmt_num(PAPER['oracle_oos_ir'])} AR={fmt_num(PAPER['oracle_oos_ar'])} MDD={fmt_num(PAPER['oracle_oos_mdd'])} CR={fmt_num(PAPER['oracle_oos_cr'])}</div>"
        f"<div class='k'>IS-best (honest)</div>"
        f"<div class='v'>OOS IR={fmt_num(PAPER['is_best_oos_ir'])} (catastrophic — 같은 paper가 IS로 골랐다면 -0.56)</div>"
        f"<div class='k'>horizon</div><div class='v'>{PAPER['horizon']}</div>"
        f"<div class='k'>stop_loss</div><div class='v'>{PAPER['stop_loss']}</div>"
        f"<div class='k'>n_trials</div><div class='v'>{PAPER['n_trials']}</div>"
        f"<div class='k'>model</div><div class='v'>{PAPER['model']}</div>"
        "</div>"
        "<div style='margin-top:14px; padding:10px; background:#1e293b; border-radius:6px; font-size:12px; color:#cbd5e1;'>"
        "⚠️ paper의 +0.65 OOS IR은 <code>analysis/0203 copy 3.ipynb</code>에서 84 combo 중 OOS IR 1위만 채택한 결과 (data snooping). "
        "IS-best로 정직하게 골랐다면 같은 paper run에서도 OOS IR=-0.5621."
        "</div></div>"
    )
    paper_row["detail_html"] = paper_detail
    rows.append(paper_row)

    # Scan all run dirs
    if RUNS_DIR.exists():
        for d in sorted(RUNS_DIR.iterdir()):
            if not d.is_dir():
                continue
            info = collect_run(d)

            # Try to derive a "label" from run_id (sweep_runner appends label after timestamp)
            rid = info["run_id"]
            label = rid
            # Pattern: YYYYMMDD_HHMMSS_<label>  (sweep) or YYYYMMDD_HHMMSS or YYYYMMDD_HHMMSS_microsec
            parts = rid.split("_")
            if len(parts) >= 3:
                # If 3rd part is non-numeric, treat as label
                if not parts[2].isdigit():
                    label = "_".join(parts[2:])
            info["label"] = label

            # Flatten metric fields used by JS
            ibo = info.get("is_best_oos") or {}
            oco = info.get("oracle_oos") or {}
            info["is_best_oos_ir"]  = ibo.get("IR")
            info["is_best_oos_ar"]  = ibo.get("AR")
            info["is_best_oos_mdd"] = ibo.get("MDD")
            info["is_best_oos_cr"]  = ibo.get("CR")
            info["oracle_oos_ir"]   = oco.get("IR")
            info["oracle_oos_ar"]   = oco.get("AR")
            info["oracle_oos_mdd"]  = oco.get("MDD")
            info["oracle_oos_cr"]   = oco.get("CR")
            info["verdict_cls"] = info["verdict"]["cls"]
            info["verdict_label"] = info["verdict"]["label"]

            # Cumulative return chart (base64-embedded PNG), keyed to best iter
            _best_iter_name = info.get("best_iter", "iter_1")
            try:
                info["cumret_b64"] = cumret_chart_b64(
                    d, info.get("is_best_combo"), info.get("oracle_combo"),
                    is_best_iter=_best_iter_name, oracle_iter=_best_iter_name,
                )
            except Exception as e:
                print(f"  [cumret skip] {d.name}: {e}", file=sys.stderr)
                info["cumret_b64"] = None

            # Per-iter cumret charts (for iter breakdown expand)
            info["iter_cumret_b64"] = {}
            for it in info.get("iters", []):
                ib = it.get("is_best_combo")
                if ib is None:
                    continue
                try:
                    info["iter_cumret_b64"][it["iter"]] = cumret_chart_b64(
                        d, ib, it.get("oracle_combo"),
                        is_best_iter=it["iter"], oracle_iter=it["iter"],
                    )
                except Exception:
                    pass

            info["detail_html"] = render_run_detail(info)
            info.pop("cumret_b64", None)  # avoid duplicate PNG in JSON
            info.pop("iter_cumret_b64", None)
            rows.append(info)

    # Build HTML
    n_runs = len(rows) - 1  # excluding paper baseline
    n_complete = sum(1 for r in rows if r.get("status") == "complete")

    html = HTML_HEAD
    html += f"""<header>
  <h1>FaVOR sweep results</h1>
  <p class="meta">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · {n_runs} run(s) discovered · {n_complete} complete</p>
  <div class="paper-baseline">
    <strong>Paper Table 1 (CSI 500 FaVOR) 기준선</strong>
    <div class="nums">AR={fmt_num(PAPER['oracle_oos_ar'])} · IR={fmt_num(PAPER['oracle_oos_ir'])} · MDD={fmt_num(PAPER['oracle_oos_mdd'])} · CR={fmt_num(PAPER['oracle_oos_cr'])}</div>
    <div class="caveat">※ 이 수치는 84 combo 중 OOS IR 1위만 채택한 OOS-oracle 값. 정직하게 IS-best로 골랐다면 같은 run에서도 OOS IR=-0.5621.</div>
  </div>
</header>

<div class="controls">
  <input id="filter" placeholder="🔍 라벨/가설/concept으로 필터…" style="flex:1; min-width:200px;">
  <span id="summary" class="summary"></span>
</div>

<table id="runs-table">
  <thead>
    <tr>
      <th rowspan="2" data-key="label">Label / Hypothesis<span class="sort-ind"></span></th>
      <th rowspan="2" data-key="n_combos">#combos<span class="sort-ind"></span></th>
      <th colspan="4" class="grp grp-is">IS-best combo (honest selection) — OOS metrics</th>
      <th colspan="4" class="grp grp-oracle">Oracle combo (paper's selection) — OOS metrics</th>
      <th rowspan="2" data-key="oos_pos_frac">+OOS<br/>combos<span class="sort-ind"></span></th>
      <th rowspan="2">Verdict</th>
      <th rowspan="2">run_id</th>
    </tr>
    <tr>
      <th data-key="is_best_oos_ir"  class="grp grp-is">IR<span class="sort-ind"></span></th>
      <th data-key="is_best_oos_ar"  class="grp grp-is">AR<span class="sort-ind"></span></th>
      <th data-key="is_best_oos_mdd" class="grp grp-is">MDD<span class="sort-ind"></span></th>
      <th data-key="is_best_oos_cr"  class="grp grp-is">CR<span class="sort-ind"></span></th>
      <th data-key="oracle_oos_ir"   class="grp grp-oracle">IR<span class="sort-ind">↓</span></th>
      <th data-key="oracle_oos_ar"   class="grp grp-oracle">AR<span class="sort-ind"></span></th>
      <th data-key="oracle_oos_mdd"  class="grp grp-oracle">MDD<span class="sort-ind"></span></th>
      <th data-key="oracle_oos_cr"   class="grp grp-oracle">CR<span class="sort-ind"></span></th>
    </tr>
  </thead>
  <tbody></tbody>
</table>

<script>
window.__RUNS__ = {json.dumps(rows, ensure_ascii=False)};
</script>
"""
    html += HTML_TAIL

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} — {n_runs} runs ({n_complete} complete)")


if __name__ == "__main__":
    main()
