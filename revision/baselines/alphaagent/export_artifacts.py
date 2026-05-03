#!/usr/bin/env python3
"""
Export AlphaAgent run artifacts (hypotheses + generated factors + formulas + code) from log/ to a CSV.

This is intentionally a standalone script (stdlib-only) to avoid touching the core pipeline logic.

Examples
  # Pick latest run under ./log and write to ./results/<run>/log_artifacts.csv
  python export_artifacts.py

  # Export a specific run
  python export_artifacts.py --log_dir log/2026-01-29_03-56-51-507305

  # Embed factor.py contents (can make CSV large)
  python export_artifacts.py --include-code --code-max-chars 50000
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_RUN_TS_FMT = "%Y-%m-%d_%H-%M-%S-%f"
_LOG_TS_FMT = "%Y-%m-%d %H:%M:%S.%f"
_LOG_PREFIX_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\s+\|")


def _parse_run_timestamp_from_stem(stem: str) -> Optional[datetime]:
    try:
        return datetime.strptime(stem, _RUN_TS_FMT)
    except Exception:
        return None


def _has_any_run_logs(run_dir: Path) -> bool:
    if not run_dir.exists() or not run_dir.is_dir():
        return False
    try:
        next(run_dir.rglob("common_logs.log"))
        return True
    except StopIteration:
        return False


def _resolve_log_dir(log_dir: Path) -> Path:
    """
    Accept either:
      - a run directory like log/2026-01-29_03-56-51-507305, or
      - the parent log directory like log/ (auto-picks the latest run).
    """
    if not log_dir.exists() or not log_dir.is_dir():
        return log_dir
    if _parse_run_timestamp_from_stem(log_dir.name) is not None:
        return log_dir

    candidates: List[Tuple[datetime, float, Path]] = []
    for child in log_dir.iterdir():
        if not child.is_dir():
            continue
        ts = _parse_run_timestamp_from_stem(child.name)
        if ts is None:
            continue
        try:
            mtime = child.stat().st_mtime
        except Exception:
            mtime = 0.0
        candidates.append((ts, mtime, child))

    candidates.sort(key=lambda t: (t[0], t[1], str(t[2])), reverse=True)
    for _, _, child in candidates:
        if _has_any_run_logs(child):
            return child
    return log_dir


@dataclass(frozen=True)
class _LLMResponseEvent:
    timestamp: datetime
    source_log: Path
    obj: Dict[str, Any]


def _parse_log_ts(line: str) -> Optional[datetime]:
    m = _LOG_PREFIX_RE.match(line)
    if not m:
        return None
    try:
        return datetime.strptime(m.group("ts"), _LOG_TS_FMT)
    except Exception:
        return None


def _slice_json_block(lines: List[str]) -> str:
    raw = "\n".join(lines).strip()
    # Best-effort: keep only from first "{" to last "}".
    start = raw.find("{")
    end = raw.rfind("}")
    if 0 <= start < end:
        return raw[start : end + 1].strip()
    return raw


def _iter_llm_response_events(log_path: Path) -> Iterable[_LLMResponseEvent]:
    """
    Parse alphaagent.oai.llm_utils logs, extracting JSON blocks that follow:
      ... - Response:
      { ...json... }
    """
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        ts = _parse_log_ts(line)
        if ts is None or " - Response:" not in line:
            i += 1
            continue

        i += 1
        json_lines: List[str] = []
        while i < len(lines) and not _LOG_PREFIX_RE.match(lines[i]):
            json_lines.append(lines[i])
            i += 1

        raw_json = _slice_json_block(json_lines)
        try:
            obj = json.loads(raw_json)
        except Exception:
            # If parsing fails, skip this event rather than crashing export.
            continue
        if isinstance(obj, dict):
            yield _LLMResponseEvent(timestamp=ts, source_log=log_path, obj=obj)


def _is_hypothesis_obj(obj: Dict[str, Any]) -> bool:
    return "hypothesis" in obj and any(k.startswith("concise_") for k in obj.keys())


def _is_factor_gen_obj(obj: Dict[str, Any]) -> bool:
    if not obj:
        return False
    # Top-level keys are factor names; values are dicts with expression/formulation.
    for v in obj.values():
        if not isinstance(v, dict):
            return False
        if "expression" not in v:
            return False
    return True


def _extract_factor_workspaces_from_run(run_dir: Path) -> Dict[str, Path]:
    """
    Parse run logs for lines like:
      File Factor[FactorName]: /abs/path/to/git_ignore_folder/RD-Agent_workspace/<uuid>
    """
    mapping: Dict[str, Path] = {}
    pattern = re.compile(r"File Factor\[(?P<name>[^]]+)\]\s*:\s*(?P<path>/\S+)")

    all_logs = list(run_dir.rglob("common_logs.log"))
    candidates = [p for p in all_logs if "d" in p.parts] or all_logs
    candidates.sort(key=lambda p: str(p))

    for log_path in candidates:
        try:
            for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                m = pattern.search(line)
                if not m:
                    continue
                name = m.group("name").strip()
                ws = Path(m.group("path")).expanduser()
                mapping.setdefault(name, ws)
        except Exception:
            continue
    return mapping


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def main(
    log_dir: str = "./log",
    out: str | None = None,
    include_code: bool = False,
    code_max_chars: int = 50000,
) -> str:
    run_dir = _resolve_log_dir(Path(log_dir))
    if not run_dir.exists() or not run_dir.is_dir():
        raise SystemExit(f"log_dir not found or not a directory: {run_dir}")

    # Collect response events (hypothesis gen + factor gen) in time order.
    events: List[_LLMResponseEvent] = []
    for p in sorted(run_dir.rglob("r/llm_messages/**/common_logs.log"), key=lambda x: str(x)):
        events.extend(list(_iter_llm_response_events(p)))
    events.sort(key=lambda e: (e.timestamp, str(e.source_log)))

    factor_ws_by_name = _extract_factor_workspaces_from_run(run_dir)

    # Default output: results/<run>/log_artifacts.csv
    if out is None:
        out_path = Path("results") / run_dir.name / "log_artifacts.csv"
    else:
        out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_dir",
        "hypothesis",
        "concise_knowledge",
        "concise_observation",
        "concise_justification",
        "concise_specification",
        "factor_name",
        "factor_description",
        "factor_formulation",
        "factor_expression",
        "workspace_path",
        "factor_py_path",
        "source_llm_log",
    ]

    cur_hyp: Optional[Dict[str, Any]] = None
    cur_hyp_ts: Optional[datetime] = None
    hyp_idx = -1
    row_count = 0

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for ev in events:
            if _is_hypothesis_obj(ev.obj):
                hyp_idx += 1
                cur_hyp = ev.obj
                cur_hyp_ts = ev.timestamp
                continue

            if not _is_factor_gen_obj(ev.obj):
                continue

            if cur_hyp is None:
                # Factor generation without a prior hypothesis in logs: still export, but leave hyp columns empty.
                cur_hyp = {}
                cur_hyp_ts = None
                hyp_idx = max(hyp_idx, 0)

            for factor_name, payload in ev.obj.items():
                if not isinstance(payload, dict):
                    continue

                ws = factor_ws_by_name.get(str(factor_name), Path())
                factor_py = (ws / "factor.py") if ws else Path()

                w.writerow(
                    {
                        "run_dir": str(run_dir),
                        "hypothesis": str(cur_hyp.get("hypothesis", "")) if cur_hyp else "",
                        "concise_knowledge": str(cur_hyp.get("concise_knowledge", "")) if cur_hyp else "",
                        "concise_observation": str(cur_hyp.get("concise_observation", "")) if cur_hyp else "",
                        "concise_justification": str(cur_hyp.get("concise_justification", "")) if cur_hyp else "",
                        "concise_specification": str(cur_hyp.get("concise_specification", "")) if cur_hyp else "",
                        "factor_name": str(factor_name),
                        "factor_description": str(payload.get("description", "")),
                        "factor_formulation": str(payload.get("formulation", "")),
                        "factor_expression": str(payload.get("expression", "")),
                        "workspace_path": str(ws) if ws else "",
                        "factor_py_path": str(factor_py) if factor_py and factor_py.exists() else "",
                        "source_llm_log": str(ev.source_log),
                    }
                )
                row_count += 1

    print(f"Wrote {row_count} rows -> {out_path}")
    return str(out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", default="./log", help="Run dir (log/<ts>) or parent (log/).")
    ap.add_argument("--out", default=None, help="Output CSV path. Default: results/<run>/log_artifacts.csv")
    ap.add_argument("--include-code", action="store_true", help="Embed factor.py content into CSV.")
    ap.add_argument("--code-max-chars", type=int, default=50000, help="Max chars of code when --include-code.")
    args = ap.parse_args()
    try:
        main(
            log_dir=args.log_dir,
            out=args.out,
            include_code=bool(args.include_code),
            code_max_chars=int(args.code_max_chars),
        )
    except BrokenPipeError:
        # Allow piping to head, etc.
        sys.exit(0)
