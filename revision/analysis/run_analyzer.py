from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd


_RE_OUTER_ITER_KEY = re.compile(r"^outer_iter_(?P<n>\d+)$")


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def specs_dir(self) -> Path:
        return self.run_dir / "specs"

    @property
    def data_dir(self) -> Path:
        return self.run_dir / "data"

    @property
    def reports_dir(self) -> Path:
        return self.run_dir / "reports"

    def log_path(self) -> Path:
        return self.logs_dir / "run.log"

    def spec(self, name: str) -> Path:
        return self.specs_dir / name

    def data(self, name: str) -> Path:
        return self.data_dir / name

    def data_any(self, name: str) -> Path:
        """
        Return an existing data file path, handling iter-suffixed parquet outputs.

        Examples:
          - stage4_trades.parquet  -> stage4_trades.parquet (if exists) else stage4_trades_iter_1.parquet, ...
          - stage4_positions.parquet -> stage4_positions_iter_*.parquet fallback
        """
        direct = self.data(name)
        if direct.exists():
            return direct

        p = Path(name)
        if p.suffix.lower() != ".parquet":
            raise FileNotFoundError(f"Data file not found: {direct}")

        stem = p.stem
        candidates = sorted(self.data_dir.glob(f"{stem}_iter_*.parquet"))
        if not candidates:
            raise FileNotFoundError(f"Data file not found: {direct} (no iter-suffixed parquet found)")

        def _iter_num(path: Path) -> int:
            m = re.search(r"_iter_(\d+)\.parquet$", path.name)
            return int(m.group(1)) if m else -1

        # Prefer the highest iteration (latest).
        return max(candidates, key=_iter_num)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_json(path: Path) -> Any:
    return json.loads(_read_text(path))


def _unwrap_outer_iter(obj: Any) -> Any:
    """
    Some runs save specs as:
      {"outer_iter_1": {...}, "outer_iter_2": {...}, ...}
    Return the latest iteration payload when that pattern is detected.
    """
    if not isinstance(obj, dict) or not obj:
        return obj

    iter_keys: list[tuple[int, str]] = []
    for k in obj.keys():
        if not isinstance(k, str):
            continue
        m = _RE_OUTER_ITER_KEY.match(k)
        if m:
            iter_keys.append((int(m.group("n")), k))

    if not iter_keys:
        return obj

    _, best_key = max(iter_keys, key=lambda t: t[0])
    return obj.get(best_key)


def read_json_optional(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    return _unwrap_outer_iter(read_json(path))


def read_parquet_any(path: Path) -> pd.DataFrame:
    """
    Read parquet with pandas first; fall back to polars if needed.
    """
    try:
        return pd.read_parquet(path)
    except Exception:
        import polars as pl

        return pl.read_parquet(path).to_pandas()


def load_run(run_dir: str | Path) -> dict[str, Any]:
    """
    Load a single run directory (specs/logs/data pointers).
    Returns a dict with keys like: hypothesis, data_split, stage2, stage3, stage4, ...
    """
    rp = RunPaths(Path(run_dir))
    out: dict[str, Any] = {"paths": rp}

    out["log_text"] = _read_text(rp.log_path()) if rp.log_path().exists() else ""

    run_config_path = rp.run_dir / "run_config.json"
    out["run_config"] = read_json_optional(run_config_path) if run_config_path.exists() else None

    for name, filename in [
        ("hypothesis", "hypothesis.json"),
        ("observation_plan", "observation_plan.json"),
        ("formula_bundle", "formula_bundle.json"),
        ("data_split", "data_split.json"),
        ("refinement_history", "refinement_history.json"),
        ("stage2_summary", "stage2_summary.json"),
        ("stage3_result", "stage3_result.json"),
        ("stage3_ticker_details", "stage3_ticker_details.json"),
        ("stage4_summary", "stage4_summary.json"),
    ]:
        out[name] = read_json_optional(rp.spec(filename))

    # Normalize: hypothesis.json is often {"hypotheses": [ {..} ]}. Expose the first hypothesis as a flat dict.
    hyp_raw = out.get("hypothesis")
    if isinstance(hyp_raw, dict) and isinstance(hyp_raw.get("hypotheses"), list) and hyp_raw["hypotheses"]:
        first = hyp_raw["hypotheses"][0]
        out["hypothesis_flat"] = first if isinstance(first, dict) else {}
    else:
        out["hypothesis_flat"] = hyp_raw if isinstance(hyp_raw, dict) else {}

    # Normalize: data_split may be stored under run_config.json in newer runs.
    if out.get("data_split") is None:
        rc = out.get("run_config")
        if isinstance(rc, dict):
            ds = rc.get("data_split") or (rc.get("config") or {}).get("data_split")
            out["data_split"] = ds if isinstance(ds, dict) else None

    return out


_RE_STAGE2_BLOCK = re.compile(
    r"Stage2 Result \\(Iteration (?P<iter>\\d+)\\):\\s*\\n"
    r".*?Total formulas:\\s*(?P<total>\\d+)\\s*\\n"
    r".*?Passed:\\s*(?P<passed>\\d+)\\s*\\n"
    r".*?Failed:\\s*(?P<failed>\\d+)\\s*\\n"
    r".*?Pass rate:\\s*(?P<rate>[0-9.]+)%\\s*",
    re.MULTILINE,
)


def parse_run_log_summary(log_text: str) -> dict[str, Any]:
    """
    Extract a compact summary from run.log (iterations, stage verdicts).
    """
    stage2_iters = []
    for m in _RE_STAGE2_BLOCK.finditer(log_text):
        stage2_iters.append(
            {
                "iteration": int(m.group("iter")),
                "total_formulas": int(m.group("total")),
                "passed": int(m.group("passed")),
                "failed": int(m.group("failed")),
                "pass_rate_pct": float(m.group("rate")),
            }
        )

    stage3_verdict = None
    m3 = re.search(r"Stage3 Result:\\s*\\n.*?Overall Verdict:\\s*(PASS|FAIL)", log_text, re.MULTILINE)
    if m3:
        stage3_verdict = m3.group(1)

    stage4_started = "Stage4:" in log_text

    return {
        "n_stage2_iterations": len(stage2_iters),
        "stage2_iterations": stage2_iters,
        "stage3_verdict": stage3_verdict,
        "stage4_started": stage4_started,
    }


def stage2_results_df(stage2_summary: dict[str, Any] | None) -> pd.DataFrame:
    """
    Row per formula, with PASS/FAIL verdict and evidence fields.
    """
    if not stage2_summary:
        return pd.DataFrame()
    results = stage2_summary.get("results", [])
    if not isinstance(results, list):
        return pd.DataFrame()
    df = pd.DataFrame(results)
    # Normalize a few expected names if present.
    if "formula_name" in df.columns and "formula_id" not in df.columns:
        df = df.rename(columns={"formula_name": "formula_id"})
    if "obs_id" in df.columns and "observation_id" not in df.columns:
        df = df.rename(columns={"obs_id": "observation_id"})
    return df


def refinement_history_df(refinement_history: dict[str, Any] | None) -> pd.DataFrame:
    if not refinement_history:
        return pd.DataFrame()
    iters = refinement_history.get("iterations", [])
    if isinstance(iters, list):
        return pd.DataFrame(iters)
    return pd.DataFrame()


def stage3_passed_combinations_df(stage3_result: dict[str, Any] | None) -> pd.DataFrame:
    if not stage3_result:
        return pd.DataFrame()
    ids = stage3_result.get("passed_combination_ids", []) or []
    names = stage3_result.get("passed_combination_names", []) or []
    rows = []
    for i in range(max(len(ids), len(names))):
        rows.append(
            {
                "instance_id": ids[i] if i < len(ids) else None,
                "formula_names": names[i] if i < len(names) else None,
                "combo_key": "|".join(sorted(names[i])) if i < len(names) and isinstance(names[i], list) else None,
            }
        )
    return pd.DataFrame(rows)


def stage3_combo_summary_df(stage3_ticker_details: dict[str, Any] | None) -> pd.DataFrame:
    """
    Aggregate Stage3 combo outcomes across tickers (pass/fail + S2_ratio improvement).
    """
    if not stage3_ticker_details or not isinstance(stage3_ticker_details, dict):
        return pd.DataFrame()

    combo_pass_fail: dict[str, list[str]] = {}
    combo_conf: dict[str, list[float]] = {}
    # cross-ticker quadrant aggregation by strictness-level position (first/last)
    combo_first_s1: dict[str, list[float]] = {}
    combo_first_s2: dict[str, list[float]] = {}
    combo_last_s1: dict[str, list[float]] = {}
    combo_last_s2: dict[str, list[float]] = {}

    for ticker, ticker_result in stage3_ticker_details.items():
        if not isinstance(ticker_result, dict):
            continue
        all_combo = ticker_result.get("all_combination_results", [])
        if not isinstance(all_combo, list):
            continue

        for cr in all_combo:
            if not isinstance(cr, dict):
                continue
            names = cr.get("combination_names") or []
            if not isinstance(names, list) or not names:
                # fallback from combination objects
                comb = cr.get("combination") or []
                if isinstance(comb, list):
                    names = [c.get("name") for c in comb if isinstance(c, dict) and c.get("name")]
            if not names:
                continue
            combo_key = "|".join(sorted(map(str, names)))

            verdict = str(cr.get("verdict", "UNKNOWN"))
            combo_pass_fail.setdefault(combo_key, []).append(verdict)
            try:
                combo_conf.setdefault(combo_key, []).append(float(cr.get("confidence", 0.0)))
            except Exception:
                combo_conf.setdefault(combo_key, []).append(0.0)

            strictness_results = cr.get("strictness_results", [])
            if isinstance(strictness_results, list) and len(strictness_results) >= 2:
                first = strictness_results[0]
                last = strictness_results[-1]

                def _extract_s1s2(sr: Any) -> tuple[float, float]:
                    if not isinstance(sr, dict):
                        return 0.0, 0.0
                    quad = sr.get("quadrant_stats", {})
                    if not isinstance(quad, dict):
                        return 0.0, 0.0
                    s1 = float(quad.get("s1_true_positive", 0) or 0)
                    s2 = float(quad.get("s2_false_positive", 0) or 0)
                    return s1, s2

                f_s1, f_s2 = _extract_s1s2(first)
                l_s1, l_s2 = _extract_s1s2(last)

                combo_first_s1.setdefault(combo_key, []).append(f_s1)
                combo_first_s2.setdefault(combo_key, []).append(f_s2)
                combo_last_s1.setdefault(combo_key, []).append(l_s1)
                combo_last_s2.setdefault(combo_key, []).append(l_s2)

    rows = []
    for combo_key, verdicts in combo_pass_fail.items():
        n_eval = len(verdicts)
        n_pass = sum(1 for v in verdicts if v == "PASS")
        pass_rate = n_pass / n_eval if n_eval else 0.0

        confs = combo_conf.get(combo_key, [])
        avg_conf = sum(confs) / len(confs) if confs else 0.0

        # S2_ratio improvement = (S2/(S1+S2) at first) - (S2/(S1+S2) at last)
        f_s1 = sum(combo_first_s1.get(combo_key, []))
        f_s2 = sum(combo_first_s2.get(combo_key, []))
        l_s1 = sum(combo_last_s1.get(combo_key, []))
        l_s2 = sum(combo_last_s2.get(combo_key, []))

        f_ratio = (f_s2 / (f_s1 + f_s2)) if (f_s1 + f_s2) > 0 else None
        l_ratio = (l_s2 / (l_s1 + l_s2)) if (l_s1 + l_s2) > 0 else None
        s2_impr = (f_ratio - l_ratio) if (f_ratio is not None and l_ratio is not None) else None

        rows.append(
            {
                "combo_key": combo_key,
                "n_eval": n_eval,
                "n_pass": n_pass,
                "pass_rate": pass_rate,
                "avg_confidence": avg_conf,
                "s2_ratio_first": f_ratio,
                "s2_ratio_last": l_ratio,
                "s2_ratio_improvement": s2_impr,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["pass_rate", "s2_ratio_improvement"], ascending=[False, False], na_position="last")
    return df


def stage4_combinations_df(stage4_summary: dict[str, Any] | None) -> pd.DataFrame:
    if not stage4_summary:
        return pd.DataFrame()
    combos = stage4_summary.get("all_combinations", [])
    if not isinstance(combos, list):
        return pd.DataFrame()
    df = pd.DataFrame(combos)
    if "formula_names" in df.columns:
        df["combo_key"] = df["formula_names"].apply(
            lambda xs: "|".join(sorted(xs)) if isinstance(xs, list) else None
        )
    return df
