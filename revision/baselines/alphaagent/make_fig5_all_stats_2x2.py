#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import random
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


RAW_FEATURES = ("mag", "dir", "vol", "pos")
PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2", "#EECA3B", "#FF9DA6"]


def _safe_float(x: object) -> Optional[float]:
    try:
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _iter_factor_dirs(roots: List[Path]) -> Iterable[Path]:
    for root in roots:
        root = root.expanduser()
        if not root.exists():
            continue
        if (root / "stage2_distributions.csv").exists():
            yield root
            continue
        for p in root.rglob("stage2_distributions.csv"):
            yield p.parent


def _read_stage2_distributions_csv(path: Path) -> Tuple[List[int], Dict[str, Dict[str, List[Optional[float]]]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    buckets: List[int] = []
    for r in rows:
        b = _safe_float(r.get("bucket"))
        if b is None:
            continue
        buckets.append(int(b))

    data: Dict[str, Dict[str, List[Optional[float]]]] = {feat: {} for feat in RAW_FEATURES}
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


def _parse_color(c: str) -> Tuple[int, int, int]:
    c = (c or "").strip()
    if c.startswith("#") and len(c) == 7:
        return (int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16))
    return (0, 0, 0)


def _text_w(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    try:
        box = draw.textbbox((0, 0), text, font=font)
        return int(box[2] - box[0])
    except Exception:
        return int(len(text) * 12)


def _normalize_0_1(values: List[Optional[float]]) -> List[Optional[float]]:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return [None for _ in values]
    vmin = float(min(finite))
    vmax = float(max(finite))
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        return [None if v is None or not math.isfinite(v) else 0.5 for v in values]
    denom = vmax - vmin
    out: List[Optional[float]] = []
    for v in values:
        if v is None or not math.isfinite(v):
            out.append(None)
        else:
            out.append((float(v) - vmin) / denom)
    return out


def render_fig5_2x2_all_stats(
    *,
    dist_csv: Path,
    out_path: Path,
    stats: List[str],
    width: int = 2400,
    height: int = 1280,
    highlight: str = "random",
    seed: Optional[int] = None,
    split_out_paths: Optional[Dict[str, Path]] = None,
) -> Optional[str]:
    buckets, data = _read_stage2_distributions_csv(dist_csv)
    if not buckets:
        return None
    n_quantiles = max(buckets)
    buckets = list(range(1, n_quantiles + 1))

    # Assemble per-feature series
    stats = [s for s in stats if any(s in data.get(feat, {}) for feat in RAW_FEATURES)]
    if not stats:
        return None
    stat_colors = {st: PALETTE[i % len(PALETTE)] for i, st in enumerate(stats)}
    feature_titles = {"mag": "MAG", "dir": "DIR", "vol": "VOL", "pos": "POS"}

    highlight_candidates = [s for s in ("mean", "q10", "q90") if s in stats]
    if highlight == "random" and not highlight_candidates:
        highlight = "none"

    highlight_stat: Optional[str] = None
    if highlight == "random":
        if seed is None:
            # Deterministic per factor dir unless user overrides; avoids paper figures changing across runs.
            s = str(dist_csv.parent).encode("utf-8")
            seed = int.from_bytes(hashlib.sha256(s).digest()[:8], "little", signed=False)
        rng = random.Random(int(seed))
        highlight_stat = rng.choice(highlight_candidates)

    per_feat: Dict[str, List[Tuple[str, str, List[Optional[float]]]]] = {}
    for feat in RAW_FEATURES:
        sers: List[Tuple[str, str, List[Optional[float]]]] = []
        for st in stats:
            vals = data.get(feat, {}).get(st)
            if not vals or len(vals) < len(buckets):
                continue
            vals = vals[: len(buckets)]
            sers.append((st, stat_colors[st], _normalize_0_1(vals)))
        per_feat[feat] = sers

    if not any(per_feat.get(f) for f in RAW_FEATURES):
        return None

    pad = 28
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font_med = _font(18)
    font_small = _font(17)

    # Unified legend at the top (wrapped into columns).
    leg_x0 = pad + 10
    leg_y0 = pad + 8
    leg_x1 = width - pad - 10
    available_w = max(1, int(leg_x1 - leg_x0))
    legend_stats = [highlight_stat] if highlight_stat else list(stats)
    legend_stats = [s for s in legend_stats if s]
    item_ws = [int(42 + 10 + _text_w(draw, st, font_med) + 30) for st in legend_stats]
    max_item_w = max(item_ws) if item_ws else 1
    max_cols = 5
    ncol = min(len(legend_stats), max_cols, max(1, int(available_w // max_item_w)))
    col_w = float(available_w) / float(ncol) if ncol else float(available_w)
    line_h = 36
    items: List[Tuple[int, int, str, str]] = []
    for i, st in enumerate(legend_stats):
        row = int(i // ncol) if ncol else 0
        col = int(i % ncol) if ncol else 0
        x = int(leg_x0 + col * col_w)
        y = int(leg_y0 + row * line_h)
        items.append((x, y, st, stat_colors.get(st, "#4C78A8")))

    legend_bottom = leg_y0
    if items:
        nrows = int(math.ceil(len(items) / ncol)) if ncol else 1
        box_left = int(leg_x0 - 10)
        box_top = int(leg_y0 - 8)
        box_right = int(leg_x1)
        box_bottom = int(leg_y0 + nrows * line_h + 6)
        legend_bottom = box_bottom
        draw.rectangle((box_left, box_top, box_right, box_bottom), fill=(255, 255, 255), outline=(220, 220, 220), width=1)
        for x, y, st, color in items:
            draw.line((x, y + 14, x + 42, y + 14), fill=_parse_color(color), width=5)
            draw.text((x + 52, y + 2), st, fill=(0, 0, 0), font=font_med)

    # Layout
    grid_gap = 18
    top = int(legend_bottom + 18)
    panel_w = int((width - 2 * pad - grid_gap) / 2)
    panel_h = int((height - pad - top - grid_gap) / 2)

    y_pad = 0.05
    y_vmin, y_vmax = -y_pad, 1.0 + y_pad

    grid_col = (205, 205, 205)
    ax_col = (153, 153, 153)
    tick_col = (85, 85, 85)

    def _draw_panel(*, x0: int, y0: int, w: int, h: int, feat: str, series: List[Tuple[str, str, List[Optional[float]]]]):
        draw.rectangle((x0, y0, x0 + w, y0 + h), outline=(221, 221, 221), width=1)
        draw.text((x0 + 10, y0 + 8), feature_titles.get(feat, feat), fill=(0, 0, 0), font=font_med)

        ax_left = x0 + 58
        ax_right = x0 + w - 18
        ax_top = y0 + 42
        ax_bottom = y0 + h - 34
        if ax_bottom - ax_top < 120 or ax_right - ax_left < 200:
            return

        draw.line((ax_left, ax_top, ax_left, ax_bottom), fill=ax_col, width=1)
        draw.line((ax_left, ax_bottom, ax_right, ax_bottom), fill=ax_col, width=1)

        tick_step = 1 if len(buckets) <= 10 else 2
        for q in buckets:
            xi = ax_left + (buckets.index(q) * ((ax_right - ax_left) / (len(buckets) - 1)) if len(buckets) > 1 else (ax_right - ax_left) / 2)
            draw.line((xi, ax_top, xi, ax_bottom), fill=grid_col, width=1)
        for q in buckets[::tick_step]:
            xi = ax_left + (buckets.index(q) * ((ax_right - ax_left) / (len(buckets) - 1)) if len(buckets) > 1 else (ax_right - ax_left) / 2)
            draw.line((xi, ax_bottom, xi, ax_bottom + 4), fill=ax_col, width=1)
            draw.text((xi - 10, ax_bottom + 7), f"Q{q}", fill=tick_col, font=font_small)

        for tval in (0.0, 0.5, 1.0):
            yi = ax_bottom - (float(tval) - float(y_vmin)) * ((ax_bottom - ax_top) / (float(y_vmax) - float(y_vmin)))
            draw.line((ax_left, yi, ax_right, yi), fill=grid_col, width=1)
            draw.line((ax_left - 4, yi, ax_left, yi), fill=ax_col, width=1)

        grey = (170, 170, 170)
        grey_line_w = 4
        grey_marker_r = 7
        hi_line_w = 8
        hi_marker_r = 14

        for _st, color, vals in series:
            is_hi = (highlight_stat is not None and _st == highlight_stat)
            draw_col = _parse_color(color) if is_hi or highlight_stat is None else grey
            line_w = hi_line_w if is_hi or highlight_stat is None else grey_line_w
            marker_r = hi_marker_r if is_hi or highlight_stat is None else grey_marker_r
            pts: List[Tuple[float, float]] = []
            for i, q in enumerate(buckets):
                v = vals[i] if i < len(vals) else None
                if v is None or not math.isfinite(float(v)):
                    continue
                xi = ax_left + (buckets.index(q) * ((ax_right - ax_left) / (len(buckets) - 1)) if len(buckets) > 1 else (ax_right - ax_left) / 2)
                yi = ax_bottom - (float(v) - float(y_vmin)) * ((ax_bottom - ax_top) / (float(y_vmax) - float(y_vmin)))
                pts.append((xi, yi))
            if len(pts) >= 2:
                draw.line(pts, fill=draw_col, width=line_w)
            for xi, yi in pts:
                draw.ellipse((xi - marker_r, yi - marker_r, xi + marker_r, yi + marker_r), fill=draw_col)

    coords = [
        (pad, top, panel_w, panel_h),  # MAG
        (pad + panel_w + grid_gap, top, panel_w, panel_h),  # DIR
        (pad, top + panel_h + grid_gap, panel_w, panel_h),  # VOL
        (pad + panel_w + grid_gap, top + panel_h + grid_gap, panel_w, panel_h),  # POS
    ]
    for (x0, y0, w, h), feat in zip(coords, RAW_FEATURES):
        _draw_panel(x0=x0, y0=y0, w=w, h=h, feat=feat, series=per_feat.get(feat, []))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")

    # Optionally also write per-panel crops (MAG/DIR/VOL/POS).
    if split_out_paths:
        for (x0, y0, w, h), feat in zip(coords, RAW_FEATURES):
            p = split_out_paths.get(feat)
            if not p:
                continue
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                crop = img.crop((int(x0), int(y0), int(x0 + w), int(y0 + h)))
                crop.save(p, format="PNG")
            except Exception:
                # Best-effort: don't fail the main plot if a crop can't be written.
                continue
    return str(out_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "roots",
        nargs="+",
        help="One or more Stage2 output dirs (or parents) that contain stage2_distributions.csv.",
    )
    ap.add_argument(
        "--stats",
        default="mean,q10,q90,kurtosis,skewness",
        help="Comma-separated stats to overlay in each subplot.",
    )
    ap.add_argument(
        "--highlight",
        choices=["random", "none"],
        default="random",
        help="Highlight one of {mean,q10,q90} and render the rest in gray (default: random).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for --highlight=random. Default: deterministic per factor dir.",
    )
    ap.add_argument(
        "--out-name",
        default="stage2_all_features_all_stats.png",
        help="Output filename to write inside each factor dir.",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    ap.add_argument(
        "--split-panels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write 4 cropped images (MAG/DIR/VOL/POS) alongside the 2x2 figure (default: on). Disable with --no-split-panels.",
    )
    args = ap.parse_args()

    roots = [Path(p) for p in args.roots]
    stats = [s.strip() for s in str(args.stats).split(",") if s.strip()]

    wrote = 0
    for factor_dir in sorted(set(_iter_factor_dirs(roots))):
        dist_csv = factor_dir / "stage2_distributions.csv"
        if not dist_csv.exists():
            continue
        out_path = factor_dir / args.out_name
        if out_path.exists() and not args.overwrite:
            continue
        split_out_paths = None
        if args.split_panels:
            split_out_paths = {
                feat: out_path.with_name(f"{out_path.stem}_{feat}{out_path.suffix}") for feat in RAW_FEATURES
            }
        try:
            maybe = render_fig5_2x2_all_stats(
                dist_csv=dist_csv,
                out_path=out_path,
                stats=stats,
                highlight=args.highlight,
                seed=args.seed,
                split_out_paths=split_out_paths,
            )
        except PermissionError as e:
            print(f"[fig5] skip (permission): {out_path} ({e})")
            maybe = None
        except Exception as e:
            print(f"[fig5] skip (error): {out_path} ({e})")
            maybe = None
        if maybe:
            wrote += 1
            print(f"[fig5] wrote: {maybe}")
            if split_out_paths:
                for feat, p in split_out_paths.items():
                    if p.exists():
                        print(f"[fig5] wrote: {p}")

    if wrote == 0:
        print("[fig5] nothing written (no matching dirs, empty CSVs, or outputs already exist).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
