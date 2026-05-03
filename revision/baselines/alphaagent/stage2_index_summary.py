#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from textwrap import shorten
from typing import Any, Iterable


@dataclass(frozen=True)
class VerdictCounts:
    n_total: int = 0
    n_pass: int = 0
    n_fail: int = 0
    n_other: int = 0
    n_mismatch_passed_vs_verdict: int = 0

    def add(self, other: "VerdictCounts") -> "VerdictCounts":
        return VerdictCounts(
            n_total=self.n_total + other.n_total,
            n_pass=self.n_pass + other.n_pass,
            n_fail=self.n_fail + other.n_fail,
            n_other=self.n_other + other.n_other,
            n_mismatch_passed_vs_verdict=(
                self.n_mismatch_passed_vs_verdict + other.n_mismatch_passed_vs_verdict
            ),
        )

    def pass_rate(self) -> float | None:
        denom = self.n_pass + self.n_fail
        if denom == 0:
            return None
        return self.n_pass / denom


def _iter_stage2_index_files(input_path: Path, recursive: bool) -> Iterable[Path]:
    if input_path.is_file():
        if input_path.name == "stage2_index.json":
            yield input_path
        return

    if not input_path.is_dir():
        return

    if recursive:
        yield from input_path.rglob("stage2_index.json")
    else:
        p = input_path / "stage2_index.json"
        if p.exists():
            yield p


def _normalize_verdict(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip().upper()
    return str(v).strip().upper()


def _summarize_index_file(path: Path) -> VerdictCounts:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))

    if not isinstance(obj, list):
        raise ValueError(f"expected JSON list at {path}, got {type(obj).__name__}")

    n_pass = 0
    n_fail = 0
    n_other = 0
    n_mismatch = 0

    for row in obj:
        if not isinstance(row, dict):
            n_other += 1
            continue

        verdict = _normalize_verdict(row.get("llm_verdict", row.get("llm_veridct")))
        if verdict == "PASS":
            n_pass += 1
        elif verdict == "FAIL":
            n_fail += 1
        else:
            n_other += 1

        passed = row.get("passed")
        if isinstance(passed, bool):
            if verdict == "PASS" and not passed:
                n_mismatch += 1
            elif verdict == "FAIL" and passed:
                n_mismatch += 1

    return VerdictCounts(
        n_total=len(obj),
        n_pass=n_pass,
        n_fail=n_fail,
        n_other=n_other,
        n_mismatch_passed_vs_verdict=n_mismatch,
    )


def _as_jsonable(counts: VerdictCounts) -> dict[str, Any]:
    scored = counts.n_pass + counts.n_fail
    return {
        "total_scored": scored,
        "total_rows": counts.n_total,
        "pass": counts.n_pass,
        "fail": counts.n_fail,
        "skipped_other": counts.n_other,
        "mismatch_passed_vs_verdict": counts.n_mismatch_passed_vs_verdict,
        "pass_rate": counts.pass_rate(),
    }


def _write_plot_stacked_counts(
    *,
    labels: list[str],
    pass_counts: list[int],
    fail_counts: list[int],
    out_path: Path,
    title: str,
    include_other: bool,
    other_counts: list[int] | None = None,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"matplotlib is required for --plot (error={type(e).__name__}: {e})") from e

    n = len(labels)
    fig_w = max(9.0, 1.2 * n)
    fig_h = 5.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    x = list(range(n))
    ax.bar(x, pass_counts, label="PASS", color="#2ca02c")
    ax.bar(x, fail_counts, bottom=pass_counts, label="FAIL", color="#d62728")
    if include_other:
        if other_counts is None:
            raise ValueError("include_other=True requires other_counts")
        ax.bar(
            x,
            other_counts,
            bottom=[p + f for p, f in zip(pass_counts, fail_counts)],
            label="OTHER",
            color="#7f7f7f",
        )

    ax.set_title(title)
    ax.set_ylabel("count")
    ax.set_xticks(x)
    ax.set_xticklabels([shorten(s, width=40, placeholder="…") for s in labels], rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()

    if out_path.exists() and out_path.is_dir():
        out_path = out_path / "stage2_index_counts.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_plot_bars(
    *,
    labels: list[str],
    counts: list[int],
    colors: list[str] | None,
    out_path: Path,
    title: str,
    ylabel: str = "count",
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"matplotlib is required for --plot (error={type(e).__name__}: {e})") from e

    n = len(labels)
    fig_w = max(7.0, 1.6 * n)
    fig_h = 5.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    x = list(range(n))
    ax.bar(x, counts, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()

    if out_path.exists() and out_path.is_dir():
        out_path = out_path / "stage2_index_counts.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Summarize PASS/FAIL counts from stage2_index.json across one or more directories.\n"
            "Looks for 'llm_verdict' (and also supports the common typo 'llm_veridct')."
        )
    )
    ap.add_argument(
        "paths",
        nargs="+",
        help="Directories (or stage2_index.json files) to scan.",
    )
    ap.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only check <dir>/stage2_index.json (no recursive search).",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON only.",
    )
    ap.add_argument(
        "--plot",
        default="",
        help="Write a stacked bar chart PNG. Example: --plot /tmp/stage2_index.png (or a directory path).",
    )
    ap.add_argument(
        "--plot-by",
        choices=["overall", "input", "file"],
        default="input",
        help="Plot grouping: overall PASS/FAIL, by input argument, or by each discovered stage2_index.json file.",
    )
    ap.add_argument(
        "--include-other",
        action="store_true",
        help="Include OTHER (non PASS/FAIL) in the stacked bars. Default: excluded.",
    )
    ap.add_argument(
        "--plot-title",
        default="",
        help="Optional plot title override.",
    )
    args = ap.parse_args(argv)

    recursive = not args.no_recursive
    root_to_files: dict[str, list[Path]] = {}
    all_files: set[Path] = set()

    for raw in args.paths:
        root = Path(raw).expanduser()
        files = sorted({p.resolve() for p in _iter_stage2_index_files(root, recursive=recursive)})
        root_to_files[raw] = files
        all_files.update(files)

    files_sorted = sorted(all_files)
    if not files_sorted:
        if args.json:
            print(json.dumps({"error": "no stage2_index.json files found"}, ensure_ascii=False, indent=2))
        else:
            print("No stage2_index.json files found.", file=sys.stderr)
        return 2

    per_file: dict[str, dict[str, Any]] = {}
    per_root: dict[str, dict[str, Any]] = {}
    overall = VerdictCounts()
    file_counts: dict[Path, VerdictCounts] = {}
    verdict_values = Counter()
    errors: list[dict[str, str]] = []

    for f in files_sorted:
        try:
            c = _summarize_index_file(f)
            overall = overall.add(c)
            file_counts[f] = c
            per_file[str(f)] = _as_jsonable(c)

            # also collect raw verdict value distribution for quick sanity checks
            obj = json.loads(f.read_text(encoding="utf-8", errors="replace"))
            if isinstance(obj, list):
                for row in obj:
                    if isinstance(row, dict):
                        verdict_values[_normalize_verdict(row.get("llm_verdict", row.get("llm_veridct")))] += 1
        except Exception as e:
            errors.append({"file": str(f), "error_type": type(e).__name__, "error": str(e)})
            print(f"[stage2_index_summary] skip: {f} ({type(e).__name__}: {e})", file=sys.stderr)

    for root_raw, files in root_to_files.items():
        root_counts = VerdictCounts()
        for f in files:
            if f in file_counts:
                root_counts = root_counts.add(file_counts[f])
        per_root[root_raw] = {
            "files_found": len(files),
            **_as_jsonable(root_counts),
        }

    out = {
        "overall": _as_jsonable(overall),
        "by_input": per_root,
        "by_file": per_file,
        "verdict_value_counts": dict(verdict_values),
        "errors": errors,
        "n_files_skipped": len(errors),
        "n_files_total": len(files_sorted),
        "recursive": recursive,
    }

    if args.plot:
        plot_path = Path(args.plot).expanduser()
        pr = overall.pass_rate()
        pr_str = "n/a" if pr is None else f"{pr:.3f}"
        title = args.plot_title or f"stage2_index PASS/FAIL (files={len(files_sorted)} pass_rate={pr_str})"

        if args.plot_by == "overall":
            labels = ["PASS", "FAIL"]
            counts = [int(overall.n_pass), int(overall.n_fail)]
            colors = ["#2ca02c", "#d62728"]
            if args.include_other:
                labels.append("OTHER")
                counts.append(int(overall.n_other))
                colors.append("#7f7f7f")
            _write_plot_bars(labels=labels, counts=counts, colors=colors, out_path=plot_path, title=title)
        elif args.plot_by == "input":
            labels = list(per_root.keys())
            pass_counts = [int(per_root[k]["pass"]) for k in labels]
            fail_counts = [int(per_root[k]["fail"]) for k in labels]
            other_counts = [int(per_root[k]["skipped_other"]) for k in labels]
            _write_plot_stacked_counts(
                labels=labels,
                pass_counts=pass_counts,
                fail_counts=fail_counts,
                other_counts=other_counts if args.include_other else None,
                include_other=args.include_other,
                out_path=plot_path,
                title=title,
            )
        else:
            labels = list(per_file.keys())
            pass_counts = [int(per_file[k]["pass"]) for k in labels]
            fail_counts = [int(per_file[k]["fail"]) for k in labels]
            other_counts = [int(per_file[k]["skipped_other"]) for k in labels]
            _write_plot_stacked_counts(
                labels=labels,
                pass_counts=pass_counts,
                fail_counts=fail_counts,
                other_counts=other_counts if args.include_other else None,
                include_other=args.include_other,
                out_path=plot_path,
                title=title,
            )
        if plot_path.exists() and plot_path.is_dir():
            print(f"[stage2_index_summary] wrote plot: {plot_path / 'stage2_index_counts.png'}", file=sys.stderr)
        else:
            print(f"[stage2_index_summary] wrote plot: {plot_path}", file=sys.stderr)

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    pr = overall.pass_rate()
    pr_str = "n/a" if pr is None else f"{pr:.3f}"
    scored = overall.n_pass + overall.n_fail
    print(
        f"files={len(files_sorted)} scored={scored} PASS={overall.n_pass} FAIL={overall.n_fail} "
        f"skip_other={overall.n_other} pass_rate={pr_str}"
    )
    if overall.n_mismatch_passed_vs_verdict:
        print(f"mismatch(passed vs llm_verdict)={overall.n_mismatch_passed_vs_verdict}", file=sys.stderr)

    print("\nBy input:")
    for root_raw, summary in per_root.items():
        pr = summary["pass_rate"]
        pr_str = "n/a" if pr is None else f"{pr:.3f}"
        print(
            f"- {root_raw}: files={summary['files_found']} scored={summary['total_scored']} "
            f"PASS={summary['pass']} FAIL={summary['fail']} skip_other={summary['skipped_other']} pass_rate={pr_str}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
