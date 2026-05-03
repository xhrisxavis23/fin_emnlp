#!/usr/bin/env python3
"""
Extract a "hypothesis - process(reasoning) - artifact" triple for judging.

Supports:
  - OURS bundle JSONs: examples_ous_bundle/*.json
  - AlphaAgent run outputs: results/<run_ts>/{log_artifacts.csv, stage2/<factor_name>/...}

Examples
  # OURS bundle
  python extract_process_triple.py ours --path examples_ous_bundle/11.json

  # AlphaAgent run + factor
  python extract_process_triple.py alpha --run results/2026-01-29_03-56-51-507305 --factor_name "Downward_Deviation_Mean_Reversion_Factor_5D"

  # Use with pair judge (heuristic)
  python - <<'PY'
  import json, subprocess
  left=json.loads(subprocess.check_output(['python','extract_process_triple.py','ours','--path','examples_ous_bundle/11.json']))
  right=json.loads(subprocess.check_output(['python','extract_process_triple.py','alpha','--run','results/2026-01-29_03-56-51-507305','--factor_name','Downward_Deviation_Mean_Reversion_Factor_5D']))
  cmd=[
    'python','judge_process_pair.py','--mode','heuristic',
    '--left_name','OURS','--left_hypothesis',left['hypothesis'],'--left_process',left['process'],'--left_artifact',left['artifact'],
    '--right_name','ALPHAAGENT','--right_hypothesis',right['hypothesis'],'--right_process',right['process'],'--right_artifact',right['artifact'],
  ]
  print(subprocess.check_output(cmd, text=True))
  PY
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return text[:max_chars].rstrip() + f"\n\n...[truncated {omitted} chars]"


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}, got {type(data).__name__}")
    return data


def _read_first_matching_row(csv_path: Path, *, factor_name: str) -> Dict[str, str]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("factor_name") or "").strip() == factor_name:
                return dict(row)
    raise FileNotFoundError(f"factor_name={factor_name!r} not found in {csv_path}")


def extract_ours_bundle(path: Path, *, include_raw: bool, max_process_chars: int) -> Dict[str, Any]:
    data = _read_json(path)
    formula = data.get("formula") or {}
    output = data.get("output") or {}

    artifact = str(formula.get("definition") or "")
    hypothesis = str(formula.get("obs_description") or "")

    llm_raw = output.get("llm_raw_response")
    llm_judgment = output.get("llm_judgment")
    result = output.get("result")

    parts: list[str] = []
    parts.append("SOURCE: OURS bundle (stage2 validation)")
    parts.append(f"bundle_path: {path.as_posix()}")
    parts.append("")
    parts.append("[ARTIFACT]")
    parts.append(artifact.strip())
    parts.append("")
    parts.append("[HYPOTHESIS]")
    parts.append(hypothesis.strip())
    parts.append("")
    parts.append("[VALIDATION_PROCESS]")

    judge_parts: list[str] = []
    judge_parts.append("SOURCE: OURS bundle (stage2 validation)")
    judge_parts.append(f"bundle_path: {path.as_posix()}")
    judge_parts.append("")
    judge_parts.append("[VALIDATION_PROCESS]")
    if include_raw and isinstance(llm_raw, str) and llm_raw.strip():
        parts.append("llm_raw_response(JSON string):")
        parts.append(llm_raw.strip())
        parts.append("")
        judge_parts.append("llm_raw_response(JSON string):")
        judge_parts.append(llm_raw.strip())
        judge_parts.append("")
    if isinstance(llm_judgment, dict):
        parts.append("llm_judgment(extracted):")
        parts.append(_json_dumps(llm_judgment))
        parts.append("")
        judge_parts.append("llm_judgment(extracted):")
        judge_parts.append(_json_dumps(llm_judgment))
        judge_parts.append("")
    if isinstance(result, dict):
        parts.append("validator_result(extracted):")
        parts.append(_json_dumps(result))
        parts.append("")
        judge_parts.append("validator_result(extracted):")
        judge_parts.append(_json_dumps(result))
        judge_parts.append("")

    return {
        "side": "OURS",
        "source": "examples_ous_bundle",
        "id": str(formula.get("formula_id") or path.stem),
        "obs_id": str(formula.get("obs_id") or ""),
        "hypothesis": hypothesis.strip(),
        "artifact": artifact.strip(),
        "process": _truncate("\n".join(parts).strip(), max_process_chars),
        "judge_process": _truncate("\n".join(judge_parts).strip(), max_process_chars),
    }


def _summarize_stage2_evidence(evidence: Dict[str, Any]) -> Dict[str, Any]:
    warnings = evidence.get("warnings")
    features = evidence.get("features")
    bins = evidence.get("bins")
    bin_counts = evidence.get("bin_counts")
    out: Dict[str, Any] = {
        "warnings_count": len(warnings) if isinstance(warnings, list) else None,
        "feature_keys": sorted(list(features.keys())) if isinstance(features, dict) else None,
        "bins_count": len(bins) if isinstance(bins, list) else None,
        "bin_counts": bin_counts if isinstance(bin_counts, list) else None,
    }
    return out


def extract_alpha_run(
    run_dir: Path,
    *,
    factor_name: str,
    include_stage2_summary: bool,
    include_stage2_evidence: str,
    max_process_chars: int,
) -> Dict[str, Any]:
    artifacts_csv = run_dir / "log_artifacts.csv"
    if not artifacts_csv.exists():
        raise FileNotFoundError(f"Missing {artifacts_csv}")
    row = _read_first_matching_row(artifacts_csv, factor_name=factor_name)

    hypothesis = (row.get("hypothesis") or "").strip()
    artifact = (row.get("factor_expression") or "").strip()
    factor_formulation = (row.get("factor_formulation") or "").strip()
    factor_description = (row.get("factor_description") or "").strip()
    concise_knowledge = (row.get("concise_knowledge") or "").strip()
    concise_observation = (row.get("concise_observation") or "").strip()
    concise_justification = (row.get("concise_justification") or "").strip()
    concise_specification = (row.get("concise_specification") or "").strip()

    stage2_dir = run_dir / "stage2" / factor_name
    stage2_summary_path = stage2_dir / "stage2_summary.json"
    stage2_evidence_path = stage2_dir / "stage2_evidence.json"

    stage2_summary: Optional[Dict[str, Any]] = None
    stage2_evidence: Optional[Dict[str, Any]] = None
    if include_stage2_summary and stage2_summary_path.exists():
        stage2_summary = _read_json(stage2_summary_path)
    if include_stage2_evidence != "none" and stage2_evidence_path.exists():
        stage2_evidence = _read_json(stage2_evidence_path)

    parts: list[str] = []
    parts.append("SOURCE: AlphaAgent run (generation + stage2 validation)")
    parts.append(f"run_dir: {run_dir.as_posix()}")
    parts.append(f"factor_name: {factor_name}")
    parts.append("")
    parts.append("[ARTIFACT]")
    parts.append(artifact)
    parts.append("")
    parts.append("[HYPOTHESIS]")
    parts.append(hypothesis or factor_description)
    parts.append("")
    parts.append("[GENERATION_PROCESS]")

    judge_parts: list[str] = []
    judge_parts.append("SOURCE: AlphaAgent run (generation + stage2 validation)")
    judge_parts.append(f"run_dir: {run_dir.as_posix()}")
    judge_parts.append(f"factor_name: {factor_name}")
    judge_parts.append("")
    judge_parts.append("[GENERATION_PROCESS]")
    if concise_knowledge:
        parts.append("concise_knowledge:")
        parts.append(concise_knowledge)
        parts.append("")
        judge_parts.append("concise_knowledge:")
        judge_parts.append(concise_knowledge)
        judge_parts.append("")
    if concise_observation:
        parts.append("concise_observation:")
        parts.append(concise_observation)
        parts.append("")
        judge_parts.append("concise_observation:")
        judge_parts.append(concise_observation)
        judge_parts.append("")
    if concise_justification:
        parts.append("concise_justification:")
        parts.append(concise_justification)
        parts.append("")
        judge_parts.append("concise_justification:")
        judge_parts.append(concise_justification)
        judge_parts.append("")
    if concise_specification:
        parts.append("concise_specification:")
        parts.append(concise_specification)
        parts.append("")
        judge_parts.append("concise_specification:")
        judge_parts.append(concise_specification)
        judge_parts.append("")
    if factor_formulation:
        parts.append("factor_formulation:")
        parts.append(factor_formulation)
        parts.append("")
        judge_parts.append("factor_formulation:")
        judge_parts.append(factor_formulation)
        judge_parts.append("")
    if factor_description and (not hypothesis or factor_description != hypothesis):
        parts.append("factor_description:")
        parts.append(factor_description)
        parts.append("")
        judge_parts.append("factor_description:")
        judge_parts.append(factor_description)
        judge_parts.append("")

    parts.append("[VALIDATION_PROCESS]")
    judge_parts.append("[VALIDATION_PROCESS]")
    if stage2_summary is not None:
        parts.append(f"stage2_summary_path: {stage2_summary_path.as_posix()}")
        parts.append(_json_dumps(stage2_summary))
        parts.append("")
        judge_parts.append(f"stage2_summary_path: {stage2_summary_path.as_posix()}")
        judge_parts.append(_json_dumps(stage2_summary))
        judge_parts.append("")
    else:
        parts.append(
            f"stage2_summary_path: ({'skipped' if not include_stage2_summary else 'missing'}) {stage2_summary_path.as_posix()}"
        )
        parts.append("")
        judge_parts.append(
            f"stage2_summary_path: ({'skipped' if not include_stage2_summary else 'missing'}) {stage2_summary_path.as_posix()}"
        )
        judge_parts.append("")
    if include_stage2_evidence == "none":
        parts.append(f"stage2_evidence_path: (skipped) {stage2_evidence_path.as_posix()}")
        parts.append("")
        judge_parts.append(f"stage2_evidence_path: (skipped) {stage2_evidence_path.as_posix()}")
        judge_parts.append("")
    else:
        if stage2_evidence is not None:
            parts.append(f"stage2_evidence_path: {stage2_evidence_path.as_posix()}")
            if include_stage2_evidence == "full":
                parts.append(_json_dumps(stage2_evidence))
            else:
                parts.append(_json_dumps(_summarize_stage2_evidence(stage2_evidence)))
            parts.append("")
            judge_parts.append(f"stage2_evidence_path: {stage2_evidence_path.as_posix()}")
            if include_stage2_evidence == "full":
                judge_parts.append(_json_dumps(stage2_evidence))
            else:
                judge_parts.append(_json_dumps(_summarize_stage2_evidence(stage2_evidence)))
            judge_parts.append("")
        else:
            parts.append(f"stage2_evidence_path: (missing) {stage2_evidence_path.as_posix()}")
            parts.append("")
            judge_parts.append(f"stage2_evidence_path: (missing) {stage2_evidence_path.as_posix()}")
            judge_parts.append("")

    return {
        "side": "ALPHAAGENT",
        "source": "results/<run>/log_artifacts.csv + results/<run>/stage2/*",
        "id": factor_name,
        "hypothesis": (hypothesis or factor_description).strip(),
        "artifact": artifact,
        "process": _truncate("\n".join(parts).strip(), max_process_chars),
        "judge_process": _truncate("\n".join(judge_parts).strip(), max_process_chars),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ours = sub.add_parser("ours")
    ours.add_argument("--path", required=True)
    ours.add_argument("--include_raw", action="store_true", help="Include output.llm_raw_response (can be long).")
    ours.add_argument("--max_process_chars", type=int, default=12000)

    alpha = sub.add_parser("alpha")
    alpha.add_argument("--run", required=True, help="Run directory under results/, e.g. results/2026-01-29_03-56-51-507305")
    alpha.add_argument("--factor_name", required=True)
    alpha.add_argument("--include_stage2_summary", action="store_true", help="Include stage2_summary.json (can be long).")
    alpha.add_argument("--include_stage2_evidence", choices=["none", "summary", "full"], default="summary")
    alpha.add_argument("--max_process_chars", type=int, default=12000)

    args = ap.parse_args(argv)

    if args.cmd == "ours":
        obj = extract_ours_bundle(
            Path(str(args.path)),
            include_raw=bool(args.include_raw),
            max_process_chars=int(args.max_process_chars),
        )
    else:
        obj = extract_alpha_run(
            Path(str(args.run)),
            factor_name=str(args.factor_name),
            include_stage2_summary=bool(args.include_stage2_summary),
            include_stage2_evidence=str(args.include_stage2_evidence),
            max_process_chars=int(args.max_process_chars),
        )

    print(json.dumps(obj, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
