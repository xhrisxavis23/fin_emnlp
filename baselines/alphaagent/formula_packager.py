#!/usr/bin/env python3
"""
formula_packager: formula JSON -> AlphaAgent-compatible factor workspaces

Takes formulas from a JSON file (e.g. example.json created from example_ours.json),
or a directory that contains many JSON files of the same shape,
and writes one factor workspace per formula under `git_ignore_folder/RD-Agent_workspace/<uuid>/`.

Each workspace contains:
  - factor.py        : computes factor values from daily_pv.h5 and writes result.h5
  - daily_pv.h5      : symlink to a source daily_pv.h5
  - formula_spec.json: original metadata (id/name/definition/polarity/obs...)

Then you can run Stage2:
  python stage2.py --factor-ws git_ignore_folder/RD-Agent_workspace/<uuid> --plot
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
from typing import Any, Dict, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class Formula:
    formula_id: str
    formula_name: str
    definition: str
    polarity: str = "higher_is_more_true"
    obs_id: str = ""
    obs_description: str = ""


@dataclass(frozen=True)
class PackagedFormula:
    source_file: str
    created_at: str
    formula: Formula
    normalized_expr: str
    daily_pv_source: str
    notes: List[str]


def _safe_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", name or "").strip("_")
    return s or "unnamed_formula"


def _ensure_symlink(link_path: Path, target: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()
    except FileNotFoundError:
        pass
    link_path.symlink_to(target)


def _load_formulas(path: Path) -> List[Formula]:
    obj = json.loads(path.read_text(encoding="utf-8"))

    # Supported shapes:
    # 1) {"formulas":[{...}, ...], ...}
    # 2) {"formula":{...}, ...}
    # 3) [{...}, {...}]
    if isinstance(obj, dict) and isinstance(obj.get("formulas"), list):
        items = obj["formulas"]
    elif isinstance(obj, dict) and isinstance(obj.get("formula"), dict):
        items = [obj["formula"]]
    elif isinstance(obj, list):
        items = obj
    else:
        raise ValueError(f"Unsupported JSON shape in {path}")

    out: List[Formula] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        out.append(
            Formula(
                formula_id=str(item.get("formula_id") or item.get("id") or item.get("name") or "").strip(),
                formula_name=str(item.get("formula_name") or item.get("name") or item.get("formula_id") or "").strip(),
                definition=str(item.get("definition") or item.get("expr") or "").strip(),
                polarity=str(item.get("polarity") or "higher_is_more_true").strip(),
                obs_id=str(item.get("obs_id") or "").strip(),
                obs_description=str(item.get("obs_description") or "").strip(),
            )
        )

    # Minimal validation
    cleaned: List[Formula] = []
    for f in out:
        if not f.definition:
            continue
        fid = f.formula_id or f.formula_name or _safe_name(f.definition)[:32]
        fname = f.formula_name or f.formula_id or fid
        cleaned.append(
            Formula(
                formula_id=fid,
                formula_name=fname,
                definition=f.definition,
                polarity=f.polarity or "higher_is_more_true",
                obs_id=f.obs_id,
                obs_description=f.obs_description,
            )
        )
    if not cleaned:
        raise ValueError(f"No usable formulas found in {path}")
    return cleaned


def _iter_input_json_files(path: Path) -> List[Path]:
    """
    Accept either:
      - a JSON file path
      - a directory containing many JSON files (non-recursive)

    Returns a sorted list of JSON file paths.
    """
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".json")
        if not files:
            raise ValueError(f"No .json files found under directory: {path}")
        return files
    raise ValueError(f"Input path is neither a file nor a directory: {path}")


def _normalize_expr_to_qlib_vars(expr: str) -> str:
    """
    Convert bare OHLCV names to Qlib-style '$' vars where appropriate.
    Example: close -> $close, but do not double-prefix $close.
    """
    s = (expr or "").strip()
    # Normalize common variants (case-insensitive) first to canonical bare tokens.
    # Keep this conservative: only rewrite whole words.
    replacements = {
        "open": "$open",
        "high": "$high",
        "low": "$low",
        "close": "$close",
        "volume": "$volume",
        "vol": "$volume",
        "return": "$return",
        "returns": "$return",
        "amount": "$amount",
    }
    for bare, qlib in replacements.items():
        # Replace word-boundary bare token not preceded by '$'
        s = re.sub(rf"(?<!\\$)\\b{re.escape(bare)}\\b", qlib, s, flags=re.IGNORECASE)
    return s


def _is_safe_expr(expr: str) -> tuple[bool, str]:
    """
    factor.py will eval() the parsed expression. Treat expr as untrusted.
    Enforce a conservative allowlist to block code injection.
    """
    if not expr:
        return False, "expr is empty"
    if any(ch in expr for ch in ("\n", "\r", "`", ";")):
        return False, "expr contains newline/backtick/semicolon"
    if "'" in expr or '"' in expr:
        return False, "expr contains quotes"
    lowered = expr.lower()
    banned = [
        "__",
        "import",
        "exec",
        "eval",
        "open(",
        "os.",
        "sys.",
        "subprocess",
        "pathlib",
    ]
    if any(b in lowered for b in banned):
        return False, f"expr contains banned token ({next(b for b in banned if b in lowered)!r})"
    if not re.fullmatch(r"[A-Za-z0-9_\$\(\)\[\]\+\-\*\/\.,:<>=!%\s]+", expr):
        return False, "expr contains unexpected characters"
    return True, ""


def _render_factor_py(*, expr: str, name: str) -> str:
    # Same style as AlphaAgent workspaces: expr_parser + function_lib + eval on df columns.
    return (
        "\n"
        "import os\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "\n"
        "from alphaagent.components.coder.factor_coder.expr_parser import parse_expression, parse_symbol\n"
        "from alphaagent.components.coder.factor_coder.function_lib import *\n"
        "\n"
        "\n"
        "def calculate_factor(expr: str, name: str) -> None:\n"
        "    df = pd.read_hdf('./daily_pv.h5', key='data')\n"
        "\n"
        "    expr2 = parse_symbol(expr, df.columns)\n"
        "    expr2 = parse_expression(expr2)\n"
        "\n"
        "    for col in df.columns:\n"
        "        expr2 = expr2.replace(col[1:], f\"df['{col}']\")\n"
        "\n"
        "    df[name] = eval(expr2)\n"
        "    result = df[name].astype(np.float64)\n"
        "\n"
        "    if os.path.exists('result.h5'):\n"
        "        os.remove('result.h5')\n"
        "    result.to_hdf('result.h5', key='data')\n"
        "\n"
        "\n"
        "if __name__ == '__main__':\n"
        f"    expr = {json.dumps(expr)}\n"
        f"    name = {json.dumps(name)}\n"
        "    calculate_factor(expr, name)\n"
    )


def create_workspace(
    *,
    formula: Formula,
    normalized_expr: str,
    workspace_root: Path,
    daily_pv_path: Path,
    source_file: str,
) -> Path:
    ws_id = uuid.uuid4().hex
    ws_dir = workspace_root / ws_id
    ws_dir.mkdir(parents=True, exist_ok=False)

    notes = [
        "Expression normalized to Qlib-style vars ($close/$open/...).",
        "factor.py uses alphaagent expr_parser + function_lib to evaluate on daily_pv.h5.",
        "Run `python factor.py` (or stage2.py --run-factor-if-missing) to create result.h5.",
    ]
    packaged = PackagedFormula(
        source_file=source_file,
        created_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        formula=formula,
        normalized_expr=normalized_expr,
        daily_pv_source=str(daily_pv_path.resolve()),
        notes=notes,
    )
    (ws_dir / "formula_spec.json").write_text(json.dumps(asdict(packaged), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ws_dir / "factor.py").write_text(_render_factor_py(expr=normalized_expr, name=formula.formula_name), encoding="utf-8")
    _ensure_symlink(ws_dir / "daily_pv.h5", daily_pv_path.resolve())
    if formula.obs_description:
        (ws_dir / "obs_description.txt").write_text(formula.obs_description.strip() + "\n", encoding="utf-8")
    return ws_dir


def _filter_formulas(formulas: Sequence[Formula], wanted: Optional[set[str]]) -> List[Formula]:
    if not wanted:
        return list(formulas)
    out: List[Formula] = []
    for f in formulas:
        if f.formula_id in wanted or f.formula_name in wanted:
            out.append(f)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="example.json",
        help="JSON containing formulas, or a directory of many JSONs (default: example.json)",
    )
    ap.add_argument(
        "--formula",
        action="append",
        default=[],
        help="Filter by formula_id or formula_name (repeatable). Default: package all.",
    )
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
    ap.add_argument("--run", action="store_true", help="Run factor.py after writing each workspace (produces result.h5)")
    ap.add_argument("--allow-unsafe", action="store_true", help="Skip expression safety checks")
    ap.add_argument("--index-out", default="", help="Optional path to write an index JSON (list of packaged workspaces)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent

    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = repo_root / in_path
    if not in_path.exists():
        raise SystemExit(f"Input JSON not found: {in_path}")

    ws_root = Path(args.workspace_root)
    if not ws_root.is_absolute():
        ws_root = repo_root / ws_root

    daily_pv = Path(args.daily_pv)
    if not daily_pv.is_absolute():
        daily_pv = repo_root / daily_pv
    if not daily_pv.exists():
        raise SystemExit(f"daily_pv.h5 not found: {daily_pv}")

    wanted = set(args.formula) if args.formula else None

    import subprocess

    rows: List[Dict[str, Any]] = []
    n_packaged = 0
    input_files = _iter_input_json_files(in_path)
    for src in input_files:
        formulas = _load_formulas(src)
        formulas = _filter_formulas(formulas, wanted)
        if not formulas:
            continue
        for f in formulas:
            expr = _normalize_expr_to_qlib_vars(f.definition)
            ok, why = _is_safe_expr(expr)
            if not ok and not args.allow_unsafe:
                raise SystemExit(f"Unsafe expr rejected for {f.formula_id}/{f.formula_name}: {why}. expr={expr!r}")

            ws_dir = create_workspace(
                formula=f,
                normalized_expr=expr,
                workspace_root=ws_root,
                daily_pv_path=daily_pv,
                source_file=str(src),
            )
            print(f"[formula_packager] wrote workspace: {ws_dir} ({f.formula_id}/{f.formula_name})")
            if args.run:
                subprocess.check_call([sys.executable, "factor.py"], cwd=str(ws_dir))
                print(f"[formula_packager] wrote: {ws_dir / 'result.h5'}")
            rows.append(
                {
                    "source_file": str(src),
                    "formula_id": f.formula_id,
                    "formula_name": f.formula_name,
                    "workspace": str(ws_dir),
                    "expr": expr,
                }
            )
            n_packaged += 1

    if wanted and n_packaged == 0:
        raise SystemExit("No formulas matched --formula filters.")

    if args.index_out:
        idx_path = Path(args.index_out)
        if not idx_path.is_absolute():
            idx_path = repo_root / idx_path
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        idx_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[formula_packager] wrote index: {idx_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
