#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


RAW_FEATURES = ("mag", "dir", "vol", "pos")


def _iter_factor_dirs(root: Path) -> Iterable[Path]:
    root = root.expanduser()
    if not root.exists():
        return
    for p in root.rglob("stage2_summary.json"):
        yield p.parent


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _rankdata(values: Sequence[float]) -> List[float]:
    # Average ranks for ties (1..n).
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    n = len(x)
    if n != len(y) or n < 2:
        return None
    mx = sum(x) / n
    my = sum(y) / n
    num = 0.0
    vx = 0.0
    vy = 0.0
    for xi, yi in zip(x, y):
        dx = xi - mx
        dy = yi - my
        num += dx * dy
        vx += dx * dx
        vy += dy * dy
    den = math.sqrt(vx * vy)
    if den <= 0:
        return None
    return float(num / den)


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 2:
        return None
    rx = _rankdata(x)
    ry = _rankdata(y)
    return _pearson(rx, ry)


def step_consistency(values: Sequence[Optional[float]]) -> Optional[float]:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if len(finite) < 3:
        return None
    # Determine overall direction from first/last finite values.
    first = next((v for v in values if v is not None and math.isfinite(v)), None)
    last = next((v for v in reversed(values) if v is not None and math.isfinite(v)), None)
    if first is None or last is None or first == last:
        return 0.0
    sgn = 1.0 if (last - first) > 0 else -1.0

    diffs: List[float] = []
    for a, b in zip(values[:-1], values[1:]):
        if a is None or b is None or (not math.isfinite(a)) or (not math.isfinite(b)):
            continue
        d = b - a
        if d != 0:
            diffs.append(d)
    if not diffs:
        return 0.0
    agree = sum(1 for d in diffs if (1.0 if d > 0 else -1.0) == sgn)
    return float(agree / len(diffs))


def _read_stage2_distributions_csv(path: Path) -> Tuple[List[int], Dict[str, Dict[str, List[Optional[float]]]]]:
    """
    Returns:
      buckets: [1..n]
      data[feature][stat] -> list aligned with buckets
    """
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    buckets = []
    for r in rows:
        b = _safe_float(r.get("bucket"))
        if b is None:
            continue
        buckets.append(int(b))

    data: Dict[str, Dict[str, List[Optional[float]]]] = {feat: {} for feat in RAW_FEATURES}
    # Collect columns present
    if not rows:
        return buckets, data
    cols = rows[0].keys()
    for feat in RAW_FEATURES:
        for col in cols:
            if not col.startswith(feat + "_"):
                continue
            stat = col[len(feat) + 1 :]
            data[feat][stat] = []

    for r in rows:
        for feat in RAW_FEATURES:
            for stat in list(data[feat].keys()):
                data[feat][stat].append(_safe_float(r.get(f"{feat}_{stat}")))
    return buckets, data


@dataclass(frozen=True)
class FactorRow:
    group: str
    factor_dir: str
    factor_name: str
    passed: bool
    pass_source: str
    n_quantiles: int
    plot_path: str
    plot_all_stats_path: str
    rho_mean: Dict[str, Optional[float]]
    sc_mean: Dict[str, Optional[float]]
    tail_range: Dict[str, Optional[float]]
    fail_mode: str


def _pick_plot_path(factor_dir: Path, summary: Dict[str, Any]) -> Tuple[str, str]:
    plot_all_stats = factor_dir / "stage2_all_features_all_stats.png"
    if plot_all_stats.exists():
        return str(plot_all_stats), str(plot_all_stats)
    # Fallback to the first plot path if present.
    plot_paths = summary.get("plot_paths") or []
    if isinstance(plot_paths, list) and plot_paths:
        return str(plot_paths[0]), str(plot_all_stats)
    return str(plot_all_stats), str(plot_all_stats)


def _classify_fail_mode(rho: Dict[str, Optional[float]], sc: Dict[str, Optional[float]]) -> str:
    # Heuristic buckets for writing analysis text.
    abs_rho = {k: (abs(v) if v is not None else 0.0) for k, v in rho.items()}
    best_feat = max(abs_rho, key=lambda k: abs_rho[k]) if abs_rho else ""
    best_val = abs_rho.get(best_feat, 0.0)

    if best_val < 0.2:
        return "flat/no-structure"

    if abs_rho.get("vol", 0.0) >= 0.6 and max(abs_rho.get("mag", 0.0), abs_rho.get("dir", 0.0), abs_rho.get("pos", 0.0)) < 0.25:
        return "vol-only"

    low_sc = [f for f, v in sc.items() if v is not None and v < 0.55]
    if low_sc:
        return "non-monotonic/unstable"

    if best_feat in ("dir", "pos", "mag"):
        return f"weak-but-{best_feat}-driven"
    return "other"


def _overall_pass_like_score(rho: Dict[str, Optional[float]], tail_range: Dict[str, Optional[float]]) -> float:
    # Favor DIR/POS/MAG monotonicity, then tail evidence.
    core = 0.0
    for f, w in (("dir", 1.0), ("pos", 0.9), ("mag", 0.7), ("vol", 0.2)):
        v = rho.get(f)
        if v is None:
            continue
        core += w * abs(v)
    tail = 0.0
    for f, w in (("dir", 1.0), ("pos", 0.9), ("mag", 0.7)):
        tr = tail_range.get(f)
        if tr is None or not math.isfinite(tr):
            continue
        tail += w * tr
    return float(core + 0.15 * tail)


def _summ(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0.0}
    return {
        "n": float(len(values)),
        "mean": float(sum(values) / len(values)),
        "median": float(statistics.median(values)),
        "p25": float(statistics.quantiles(values, n=4)[0]) if len(values) >= 4 else float(min(values)),
        "p75": float(statistics.quantiles(values, n=4)[2]) if len(values) >= 4 else float(max(values)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--group",
        action="append",
        default=[],
        help="Group spec NAME=PATH (repeatable). Example: --group ours=results/formulas",
    )
    ap.add_argument(
        "--out-json",
        default="mk_images/validation_stage_analysis.json",
        help="Where to write the analysis JSON.",
    )
    ap.add_argument(
        "--out-md",
        default="mk_images/validation_stage_analysis.md",
        help="Where to write a human-readable summary.",
    )
    ap.add_argument(
        "--max-per-group",
        type=int,
        default=0,
        help="Optional cap on number of factors per group (0 = no cap).",
    )
    ap.add_argument(
        "--target-n-quantiles",
        default="auto",
        help="Target n_quantiles for exemplar selection. Use an int (e.g., 20) or 'auto' to pick the most available value across groups.",
    )
    args = ap.parse_args()

    if not args.group:
        raise SystemExit("Provide at least one --group NAME=PATH")

    groups: List[Tuple[str, Path]] = []
    for item in args.group:
        if "=" not in item:
            raise SystemExit(f"Invalid --group '{item}'. Expected NAME=PATH.")
        name, path = item.split("=", 1)
        groups.append((name.strip(), Path(path.strip())))

    rows: List[FactorRow] = []
    for group_name, root in groups:
        n = 0
        for factor_dir in _iter_factor_dirs(root):
            if args.max_per_group and n >= args.max_per_group:
                break
            summary_path = factor_dir / "stage2_summary.json"
            dist_path = factor_dir / "stage2_distributions.csv"
            if not summary_path.exists() or not dist_path.exists():
                continue

            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            factor_name = str(summary.get("factor_name") or factor_dir.name)
            passed = bool(summary.get("passed"))
            pass_source = str(summary.get("pass_source") or "")
            n_quantiles = int(summary.get("n_quantiles") or 0)

            try:
                buckets, data = _read_stage2_distributions_csv(dist_path)
            except Exception:
                continue
            if not buckets:
                continue
            n_quantiles = n_quantiles or max(buckets)
            x = [float(b) for b in buckets]

            rho_mean: Dict[str, Optional[float]] = {}
            sc_mean: Dict[str, Optional[float]] = {}
            tail_range: Dict[str, Optional[float]] = {}
            for feat in RAW_FEATURES:
                mean_vals = data.get(feat, {}).get("mean", [])
                # align lengths; tolerate missing
                if len(mean_vals) == len(x) and any(v is not None for v in mean_vals):
                    y = [float(v) for v in mean_vals if v is not None]
                    xx = [x[i] for i, v in enumerate(mean_vals) if v is not None]
                    rho_mean[feat] = spearman_rho(xx, y) if len(xx) >= 2 else None
                    sc_mean[feat] = step_consistency(mean_vals)
                else:
                    rho_mean[feat] = None
                    sc_mean[feat] = None

                q10 = data.get(feat, {}).get("q10", [])
                q90 = data.get(feat, {}).get("q90", [])
                if len(q10) == len(q90) == len(x) and any(a is not None and b is not None for a, b in zip(q10, q90)):
                    spreads: List[float] = []
                    for a, b in zip(q10, q90):
                        if a is None or b is None:
                            continue
                        spreads.append(float(b - a))
                    if len(spreads) >= 2:
                        tail_range[feat] = float(max(spreads) - min(spreads))
                    else:
                        tail_range[feat] = None
                else:
                    tail_range[feat] = None

            plot_path, plot_all_stats_path = _pick_plot_path(factor_dir, summary)
            fail_mode = _classify_fail_mode(rho_mean, sc_mean) if not passed else "pass"

            rows.append(
                FactorRow(
                    group=group_name,
                    factor_dir=str(factor_dir),
                    factor_name=factor_name,
                    passed=passed,
                    pass_source=pass_source,
                    n_quantiles=n_quantiles,
                    plot_path=plot_path,
                    plot_all_stats_path=plot_all_stats_path,
                    rho_mean=rho_mean,
                    sc_mean=sc_mean,
                    tail_range=tail_range,
                    fail_mode=fail_mode,
                )
            )
            n += 1

    # Group summaries + exemplars
    out: Dict[str, Any] = {"groups": {}, "exemplars": {}, "rows": []}
    for r in rows:
        out["rows"].append(
            {
                "group": r.group,
                "factor_dir": r.factor_dir,
                "factor_name": r.factor_name,
                "passed": r.passed,
                "pass_source": r.pass_source,
                "n_quantiles": r.n_quantiles,
                "plot_path": r.plot_path,
                "plot_all_stats_path": r.plot_all_stats_path,
                "rho_mean": r.rho_mean,
                "sc_mean": r.sc_mean,
                "tail_range": r.tail_range,
                "fail_mode": r.fail_mode,
            }
        )

    by_group: Dict[str, List[FactorRow]] = {}
    for r in rows:
        by_group.setdefault(r.group, []).append(r)

    # Pick a target n_quantiles so exemplar plots have consistent x-axis across groups.
    if str(args.target_n_quantiles).strip().lower() != "auto":
        try:
            target_nq = int(args.target_n_quantiles)
        except Exception as e:
            raise SystemExit(f"Invalid --target-n-quantiles={args.target_n_quantiles}. Use int or 'auto'. error={e}")
    else:
        # Choose n that maximizes availability across groups:
        # score(n) = min_g count_g(n), tie-break by sum_g count_g(n).
        candidates: Dict[int, Dict[str, int]] = {}
        for g, g_rows in by_group.items():
            counts: Dict[int, int] = {}
            for r in g_rows:
                if r.n_quantiles:
                    counts[int(r.n_quantiles)] = counts.get(int(r.n_quantiles), 0) + 1
            for nq, c in counts.items():
                candidates.setdefault(int(nq), {})[g] = int(c)
        best_nq = None
        best_score = (-1, -1)
        for nq, per_g in candidates.items():
            if not by_group:
                continue
            minc = min(per_g.get(g, 0) for g in by_group.keys())
            sumc = sum(per_g.get(g, 0) for g in by_group.keys())
            score = (minc, sumc)
            if score > best_score:
                best_score = score
                best_nq = nq
        target_nq = int(best_nq) if best_nq is not None else 10

    out["target_n_quantiles"] = target_nq

    for g, g_rows in by_group.items():
        passed_rows = [r for r in g_rows if r.passed]
        fail_rows = [r for r in g_rows if not r.passed]

        abs_rho_dir = [abs(r.rho_mean["dir"]) for r in g_rows if r.rho_mean.get("dir") is not None]
        abs_rho_pos = [abs(r.rho_mean["pos"]) for r in g_rows if r.rho_mean.get("pos") is not None]
        abs_rho_mag = [abs(r.rho_mean["mag"]) for r in g_rows if r.rho_mean.get("mag") is not None]

        counts_by_nq: Dict[int, int] = {}
        for r in g_rows:
            if r.n_quantiles:
                counts_by_nq[int(r.n_quantiles)] = counts_by_nq.get(int(r.n_quantiles), 0) + 1

        out["groups"][g] = {
            "n_total": len(g_rows),
            "n_passed": len(passed_rows),
            "pass_rate": (len(passed_rows) / len(g_rows) if g_rows else 0.0),
            "abs_rho_dir": _summ(abs_rho_dir),
            "abs_rho_pos": _summ(abs_rho_pos),
            "abs_rho_mag": _summ(abs_rho_mag),
            "counts_by_n_quantiles": {str(k): v for k, v in sorted(counts_by_nq.items())},
            "fail_modes": {
                k: sum(1 for r in fail_rows if r.fail_mode == k)
                for k in sorted({r.fail_mode for r in fail_rows})
            },
        }

        def _filter_by_nq(rows_: List[FactorRow]) -> List[FactorRow]:
            exact = [r for r in rows_ if int(r.n_quantiles or 0) == int(target_nq)]
            if exact:
                return exact
            # Fallback: closest n_quantiles to target.
            finite = [r for r in rows_ if r.n_quantiles]
            if not finite:
                return rows_
            best = min(finite, key=lambda r: abs(int(r.n_quantiles) - int(target_nq)))
            return [best]

        def _best(rows_: List[FactorRow]) -> Optional[FactorRow]:
            if not rows_:
                return None
            rows_ = _filter_by_nq(rows_)
            return max(rows_, key=lambda r: _overall_pass_like_score(r.rho_mean, r.tail_range))

        def _worst(rows_: List[FactorRow]) -> Optional[FactorRow]:
            if not rows_:
                return None
            rows_ = _filter_by_nq(rows_)
            # Pick a "clear" failure: first choose the most common fail mode in this group,
            # then pick the lowest-structure instance within it.
            counts: Dict[str, int] = {}
            for r in rows_:
                counts[r.fail_mode] = counts.get(r.fail_mode, 0) + 1
            mode = max(counts.items(), key=lambda kv: kv[1])[0]
            candidates = [r for r in rows_ if r.fail_mode == mode] or rows_
            return min(candidates, key=lambda r: _overall_pass_like_score(r.rho_mean, r.tail_range))

        best = _best(passed_rows)
        reference = _best(g_rows)
        worst = _worst(fail_rows)
        # For "pass", prefer exemplars that already have the figure-friendly plot available.
        # This avoids permission issues when some result dirs are read-only.
        if best is not None and not (Path(best.factor_dir) / "stage2_all_features_all_stats.png").exists():
            alt = None
            for rr in _filter_by_nq(passed_rows):
                if (Path(rr.factor_dir) / "stage2_all_features_all_stats.png").exists():
                    alt = rr
                    break
            if alt is not None:
                best = alt
        out["exemplars"][g] = {
            "pass": (best.factor_dir if best else ""),
            "reference": (reference.factor_dir if reference else ""),
            "fail": (worst.factor_dir if worst else ""),
        }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # Write a small, paper-friendly markdown summary.
    md_lines = ["# Validation Stage Analysis (Stage2) — Summary", ""]
    md_lines.append(f"- target_n_quantiles: {target_nq}")
    md_lines.append("")
    for g in [name for name, _p in groups]:
        if g not in out["groups"]:
            continue
        gs = out["groups"][g]
        md_lines.append(f"## {g}")
        md_lines.append(f"- n_total: {gs['n_total']}")
        md_lines.append(f"- pass_rate: {gs['pass_rate']:.3f} ({gs['n_passed']}/{gs['n_total']})")
        md_lines.append(f"- |rho|(DIR_mean): mean={gs['abs_rho_dir'].get('mean','')}, median={gs['abs_rho_dir'].get('median','')}")
        md_lines.append(f"- |rho|(POS_mean): mean={gs['abs_rho_pos'].get('mean','')}, median={gs['abs_rho_pos'].get('median','')}")
        md_lines.append(f"- |rho|(MAG_mean): mean={gs['abs_rho_mag'].get('mean','')}, median={gs['abs_rho_mag'].get('median','')}")
        if gs["fail_modes"]:
            md_lines.append(f"- fail_modes: {gs['fail_modes']}")
        ex = out["exemplars"].get(g, {})
        if ex:
            md_lines.append(f"- exemplar_pass: `{ex.get('pass','')}`")
            md_lines.append(f"- exemplar_reference: `{ex.get('reference','')}`")
            md_lines.append(f"- exemplar_fail: `{ex.get('fail','')}`")
        md_lines.append("")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines).strip() + "\n", encoding="utf-8")

    print(f"[analysis] wrote: {out_json}")
    print(f"[analysis] wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
