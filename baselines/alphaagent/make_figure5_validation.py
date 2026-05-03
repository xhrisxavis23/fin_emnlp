#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    for fp in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ):
        try:
            return ImageFont.truetype(fp, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _load_img(path: Path) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def _pick_img(factor_dir: Path) -> Optional[Path]:
    cand = factor_dir / "stage2_all_features_all_stats.png"
    if cand.exists():
        return cand
    # fallback to any stage2_*.png
    for p in sorted(factor_dir.glob("stage2_*.png")):
        return p
    return None


def _ensure_all_stats_png(*, factor_dir: Path, cache_path: Path) -> Optional[Path]:
    """
    Ensure we have a stage2_all_features_all_stats-like PNG for this factor dir.
    Writes into cache_path (workspace-writable) to avoid permission issues inside results/.
    """
    src = factor_dir / "stage2_all_features_all_stats.png"
    if src.exists():
        return src
    dist_csv = factor_dir / "stage2_distributions.csv"
    if not dist_csv.exists():
        return None
    try:
        import make_fig5_all_stats_2x2 as gen  # type: ignore
    except Exception:
        return None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    stats = ["mean", "q10", "q90", "kurtosis", "skewness"]
    try:
        maybe = gen.render_fig5_2x2_all_stats(dist_csv=dist_csv, out_path=cache_path, stats=stats)
    except Exception:
        maybe = None
    return cache_path if maybe else None


def _text_w(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    try:
        box = draw.textbbox((0, 0), text, font=font)
        return int(box[2] - box[0])
    except Exception:
        return int(len(text) * 12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--analysis-json",
        default="mk_images/validation_stage_analysis.json",
        help="JSON produced by analyze_validation_stage.py",
    )
    ap.add_argument(
        "--out",
        default="mk_images/figure5_validation.png",
        help="Output PNG path.",
    )
    ap.add_argument(
        "--layout",
        choices=["pass_only", "pass_fail"],
        default="pass_fail",
        help="pass_only = 1 row (PASS), pass_fail = 2 rows (PASS/FAIL).",
    )
    ap.add_argument(
        "--skip-groups-without-pass",
        action="store_true",
        help="If enabled, replace PASS cell with a placeholder for groups that have no PASS exemplar.",
    )
    ap.add_argument(
        "--use-reference-for-missing-pass",
        action="store_true",
        help="If enabled, fill missing PASS cells with the group's 'reference' exemplar (closest-to-aligned, may still fail Stage2).",
    )
    ap.add_argument(
        "--group-order",
        default="ours,alphaagent,alpha101",
        help="Comma-separated group order (must match keys in analysis JSON).",
    )
    args = ap.parse_args()

    analysis_path = Path(args.analysis_json)
    obj = json.loads(analysis_path.read_text(encoding="utf-8"))
    exemplars: Dict[str, Dict[str, str]] = obj.get("exemplars") or {}

    group_order = [s.strip() for s in str(args.group_order).split(",") if s.strip()]
    rows = ["pass"] if args.layout == "pass_only" else ["pass", "fail"]
    row_labels = None
    if args.layout == "pass_fail":
        row_labels = {"pass": "ALIGNED", "fail": "MISALIGNED"}

    # Load images
    images: Dict[Tuple[str, str], Image.Image] = {}
    cell_kind: Dict[Tuple[str, str], str] = {}
    missing: List[str] = []
    for g in group_order:
        ex = exemplars.get(g) or {}
        for r in rows:
            if r == "pass" and args.skip_groups_without_pass and not ex.get("pass"):
                missing.append(f"{g}:{r} (no PASS in group)")
                continue
            d = ex.get(r, "")
            if r == "pass" and (not d) and args.use_reference_for_missing_pass:
                d = ex.get("reference", "")
                if d:
                    cell_kind[(g, r)] = "reference"
            if not d:
                missing.append(f"{g}:{r} (no exemplar)")
                continue
            factor_dir = Path(d)
            cache_path = Path("mk_images") / "fig5_cache" / f"{g}_{r}.png"
            p = _ensure_all_stats_png(factor_dir=factor_dir, cache_path=cache_path) or _pick_img(factor_dir)
            if not p:
                missing.append(f"{g}:{r} (no png in {d})")
                continue
            img = _load_img(p)
            if not img:
                missing.append(f"{g}:{r} (failed to read {p})")
                continue
            images[(g, r)] = img
            cell_kind.setdefault((g, r), r)

    if not images:
        raise SystemExit("No images could be loaded. Generate stage2_all_features_all_stats.png first.")

    # Normalize cell sizes (use first image size)
    cell_w, cell_h = next(iter(images.values())).size
    for k, im in list(images.items()):
        if im.size != (cell_w, cell_h):
            images[k] = im.resize((cell_w, cell_h), Image.Resampling.LANCZOS)

    pad = 24
    header_h = 56
    row_label_w = 86 if len(rows) > 1 else 0
    title_h = header_h

    grid_w = row_label_w + len(group_order) * cell_w + (len(group_order) - 1) * pad
    grid_h = len(rows) * cell_h + (len(rows) - 1) * pad
    width = pad * 2 + grid_w
    height = pad * 2 + title_h + pad + grid_h + (36 if missing else 0)

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font_head = _font(24)
    font_row = _font(22)
    font_small = _font(16)

    # Column headers
    x0 = pad + row_label_w
    y0 = pad
    for i, g in enumerate(group_order):
        label = g
        cx = x0 + i * (cell_w + pad)
        tw = _text_w(draw, label, font_head)
        draw.text((cx + (cell_w - tw) / 2, y0 + 10), label, fill=(0, 0, 0), font=font_head)

    # Cells
    grid_y0 = pad + title_h + pad
    for r_i, r in enumerate(rows):
        cy = grid_y0 + r_i * (cell_h + pad)
        if row_label_w:
            rl = (row_labels.get(r, r.upper()) if row_labels else r.upper())
            tw = _text_w(draw, rl, font_row)
            draw.text((pad + (row_label_w - tw) / 2, cy + 10), rl, fill=(0, 0, 0), font=font_row)
        for c_i, g in enumerate(group_order):
            cx = pad + row_label_w + c_i * (cell_w + pad)
            im = images.get((g, r))
            if im is None:
                # placeholder
                draw.rectangle((cx, cy, cx + cell_w, cy + cell_h), outline=(220, 220, 220), width=2)
                draw.text((cx + 10, cy + 10), "missing", fill=(120, 120, 120), font=font_small)
            else:
                canvas.paste(im, (cx, cy))
                if cell_kind.get((g, r)) == "reference":
                    tag = "REFERENCE"
                    draw.rectangle((cx + 8, cy + 8, cx + 140, cy + 34), fill=(255, 255, 255), outline=(210, 210, 210), width=1)
                    draw.text((cx + 16, cy + 10), tag, fill=(120, 120, 120), font=font_small)

    if missing:
        msg = "Missing: " + "; ".join(missing[:6]) + (" ..." if len(missing) > 6 else "")
        draw.text((pad, height - 28), msg, fill=(120, 120, 120), font=font_small)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG")

    # Also emit PDF for paper drafts.
    pdf_path = out_path.with_suffix(".pdf")
    canvas.save(pdf_path, format="PDF")

    print(f"[fig5] wrote: {out_path}")
    print(f"[fig5] wrote: {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
