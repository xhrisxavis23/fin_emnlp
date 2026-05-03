#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import shorten
from typing import Any, Iterable


@dataclass(frozen=True)
class PassFailCounts:
    n_total: int = 0
    n_pass: int = 0
    n_fail: int = 0
    n_other: int = 0

    def add(self, other: "PassFailCounts") -> "PassFailCounts":
        return PassFailCounts(
            n_total=self.n_total + other.n_total,
            n_pass=self.n_pass + other.n_pass,
            n_fail=self.n_fail + other.n_fail,
            n_other=self.n_other + other.n_other,
        )

    def pass_rate(self) -> float | None:
        denom = self.n_pass + self.n_fail
        if denom == 0:
            return None
        return self.n_pass / denom


def _iter_stage2_summary_files(input_path: Path, recursive: bool) -> Iterable[Path]:
    if input_path.is_file():
        if input_path.name == "stage2_summary.json":
            yield input_path
        return

    if not input_path.is_dir():
        return

    if recursive:
        yield from input_path.rglob("stage2_summary.json")
    else:
        p = input_path / "stage2_summary.json"
        if p.exists():
            yield p


def _normalize_verdict(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip().upper()
    return str(v).strip().upper()


def _infer_passed(obj: dict[str, Any]) -> bool | None:
    passed = obj.get("passed")
    if isinstance(passed, bool):
        return passed

    llm_judgment = obj.get("llm_judgment")
    if isinstance(llm_judgment, dict):
        verdict = _normalize_verdict(llm_judgment.get("verdict"))
        if verdict == "PASS":
            return True
        if verdict == "FAIL":
            return False

    verdict = _normalize_verdict(obj.get("llm_verdict", obj.get("llm_veridct")))
    if verdict == "PASS":
        return True
    if verdict == "FAIL":
        return False

    return None


def _summarize_summary_file(path: Path) -> PassFailCounts:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))

    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object at {path}, got {type(obj).__name__}")

    inferred = _infer_passed(obj)
    if inferred is True:
        return PassFailCounts(n_total=1, n_pass=1)
    if inferred is False:
        return PassFailCounts(n_total=1, n_fail=1)
    return PassFailCounts(n_total=1, n_other=1)


def _as_jsonable(counts: PassFailCounts) -> dict[str, Any]:
    scored = counts.n_pass + counts.n_fail
    return {
        "total_scored": scored,
        "total_rows": counts.n_total,
        "pass": counts.n_pass,
        "fail": counts.n_fail,
        "skipped_other": counts.n_other,
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
        out_path = out_path / "stage2_summary_counts.png"
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
        out_path = out_path / "stage2_summary_counts.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Aggregate PASS/FAIL ratio across stage2_summary.json files.\n"
            "Primarily uses 'passed' (bool), and falls back to llm verdict fields when needed."
        )
    )
    ap.add_argument("paths", nargs="+", help="Directories (or stage2_summary.json files) to scan.")
    ap.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only check <dir>/stage2_summary.json (no recursive search).",
    )
    ap.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    ap.add_argument(
        "--details",
        action="store_true",
        help="Print per-file PASS/FAIL line (can be long).",
    )
    ap.add_argument(
        "--plot",
        default="",
        help="Write a stacked bar chart PNG. Example: --plot /tmp/stage2_summary.png (or a directory path).",
    )
    ap.add_argument(
        "--plot-by",
        choices=["overall", "input", "file"],
        default="input",
        help="Plot grouping: overall PASS/FAIL, by input argument, or by each discovered stage2_summary.json file.",
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
        files = sorted({p.resolve() for p in _iter_stage2_summary_files(root, recursive=recursive)})
        root_to_files[raw] = files
        all_files.update(files)

    files_sorted = sorted(all_files)
    if not files_sorted:
        if args.json:
            print(json.dumps({"error": "no stage2_summary.json files found"}, ensure_ascii=False, indent=2))
        else:
            print("No stage2_summary.json files found.", file=sys.stderr)
        return 2

    per_file: dict[str, dict[str, Any]] = {}
    per_root: dict[str, dict[str, Any]] = {}
    overall = PassFailCounts()
    file_counts: dict[Path, PassFailCounts] = {}
    errors: list[dict[str, str]] = []

    for f in files_sorted:
        try:
            c = _summarize_summary_file(f)
            overall = overall.add(c)
            file_counts[f] = c
            per_file[str(f)] = _as_jsonable(c)
        except Exception as e:
            errors.append({"file": str(f), "error_type": type(e).__name__, "error": str(e)})
            print(f"[stage2_summary_aggregate] skip: {f} ({type(e).__name__}: {e})", file=sys.stderr)

    for root_raw, files in root_to_files.items():
        root_counts = PassFailCounts()
        for f in files:
            if f in file_counts:
                root_counts = root_counts.add(file_counts[f])
        per_root[root_raw] = {"files_found": len(files), **_as_jsonable(root_counts)}

    out = {
        "overall": _as_jsonable(overall),
        "by_input": per_root,
        "by_file": per_file,
        "errors": errors,
        "n_files_skipped": len(errors),
        "n_files_total": len(files_sorted),
        "recursive": recursive,
    }

    if args.plot:
        plot_path = Path(args.plot).expanduser()
        pr = overall.pass_rate()
        pr_str = "n/a" if pr is None else f"{pr:.3f}"
        title = args.plot_title or f"stage2_summary PASS/FAIL (files={len(files_sorted)} pass_rate={pr_str})"

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
            print(f"[stage2_summary_aggregate] wrote plot: {plot_path / 'stage2_summary_counts.png'}", file=sys.stderr)
        else:
            print(f"[stage2_summary_aggregate] wrote plot: {plot_path}", file=sys.stderr)

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    pr = overall.pass_rate()
    pr_str = "n/a" if pr is None else f"{pr:.3f}"
    print(
        f"files={len(files_sorted)} scored={overall.n_pass + overall.n_fail} PASS={overall.n_pass} FAIL={overall.n_fail} "
        f"skip_other={overall.n_other} pass_rate={pr_str}"
    )

    print("\nBy input:")
    for root_raw, summary in per_root.items():
        pr = summary["pass_rate"]
        pr_str = "n/a" if pr is None else f"{pr:.3f}"
        print(
            f"- {root_raw}: files={summary['files_found']} scored={summary['total_scored']} "
            f"PASS={summary['pass']} FAIL={summary['fail']} skip_other={summary['skipped_other']} pass_rate={pr_str}"
        )

    if args.details:
        print("\nDetails:")
        for f in files_sorted:
            v = per_file.get(str(f))
            if not v:
                continue
            verdict = "OTHER"
            if v["pass"] == 1:
                verdict = "PASS"
            elif v["fail"] == 1:
                verdict = "FAIL"
            print(f"- {verdict} {f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
