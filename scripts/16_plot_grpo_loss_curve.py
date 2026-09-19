#!/usr/bin/env python3
"""Render policy-gradient and raw-KL curves from a local offline W&B run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


def load_rows(run_dir: Path) -> list[dict[str, float]]:
    run_file = next(run_dir.glob("run-*.wandb"), None)
    if run_file is None:
        raise FileNotFoundError(f"No run-*.wandb under {run_dir}")
    store = DataStore(); store.open_for_scan(str(run_file)); rows = []
    try:
        while True:
            data = store.scan_data()
            if data is None:
                break
            record = wandb_internal_pb2.Record(); record.ParseFromString(data)
            if record.WhichOneof("record_type") != "history":
                continue
            row = {}
            for item in record.history.item:
                key = item.key or "/".join(item.nested_key)
                if key:
                    row[key] = json.loads(item.value_json)
            if all(key in row for key in ("_step", "actor/pg_loss", "actor/kl_loss")):
                rows.append({"step": float(row["_step"]), "pg": float(row["actor/pg_loss"]), "kl": float(row["actor/kl_loss"])})
    finally:
        store.close()
    return rows


def moving_average(values: list[float], window: int) -> list[float]:
    return [sum(values[max(0, i - window + 1):i + 1]) / min(i + 1, window) for i in range(len(values))]


def svg_curve(points: list[tuple[float, float]], x, y) -> str:
    return " ".join(f"{x(step):.2f},{y(value):.2f}" for step, value in points)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, default=500)
    parser.add_argument("--stop-step", type=int, default=557)
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()
    rows = load_rows(args.wandb_run)
    if not rows:
        raise RuntimeError("No actor pg/KL metric rows found")
    steps = [row["step"] for row in rows]
    pg = [row["pg"] for row in rows]
    kl = [row["kl"] for row in rows]
    pg_ma, kl_ma = moving_average(pg, args.window), moving_average(kl, args.window)
    width, height, left, right = 1200, 760, 95, 45
    panel_h, top1, top2 = 230, 75, 435
    plot_w = width - left - right
    min_step, max_step = min(steps), max(max(steps), args.stop_step) + 8

    def x(step: float) -> float:
        return left + (step - min_step) / (max_step - min_step) * plot_w

    def panel(title: str, values: list[float], smoothed: list[float], top: int, color: str, y_label: str) -> str:
        low, high = min(values + smoothed), max(values + smoothed)
        padding = max((high - low) * .12, 0.001)
        low, high = low - padding, high + padding
        def y(value: float) -> float:
            return top + panel_h - (value - low) / (high - low) * panel_h
        grid, labels = [], []
        for index in range(5):
            value = low + (high - low) * index / 4
            yy = y(value)
            grid.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{width-right}" y2="{yy:.2f}" stroke="#d9e1e8"/>')
            labels.append(f'<text x="{left-10}" y="{yy+5:.2f}" text-anchor="end">{value:.3f}</text>')
        raw = svg_curve(list(zip(steps, values)), x, y)
        average = svg_curve(list(zip(steps, smoothed)), x, y)
        checkpoint_x, stop_x = x(args.checkpoint_step), x(args.stop_step)
        return f'''<text class="paneltitle" x="{left}" y="{top-18}">{title}</text>
{''.join(grid)}<rect x="{left}" y="{top}" width="{plot_w}" height="{panel_h}" fill="none" stroke="#73808c"/>
<polyline points="{raw}" fill="none" stroke="{color}" stroke-width="1.1" opacity=".45"/>
<polyline points="{average}" fill="none" stroke="{color}" stroke-width="3"/>
<line x1="{checkpoint_x:.2f}" y1="{top}" x2="{checkpoint_x:.2f}" y2="{top+panel_h}" stroke="#2e8b57" stroke-width="2" stroke-dasharray="7,5"/>
<line x1="{stop_x:.2f}" y1="{top}" x2="{stop_x:.2f}" y2="{top+panel_h}" stroke="#b22222" stroke-width="2" stroke-dasharray="7,5"/>
{''.join(labels)}<text class="ylabel" x="25" y="{top+panel_h/2}" text-anchor="middle" transform="rotate(-90 25 {top+panel_h/2})">{y_label}</text>'''

    ticks = []
    for step in range(100, max_step + 1, 100):
        xx = x(step)
        ticks.append(f'<line x1="{xx:.2f}" y1="{top2}" x2="{xx:.2f}" y2="{top2+panel_h}" stroke="#edf1f4"/><text x="{xx:.2f}" y="{top2+panel_h+28}" text-anchor="middle">{step}</text>')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/><style>text{{font-family:Arial,sans-serif;font-size:15px;fill:#263238}} .title{{font-size:24px;font-weight:bold}} .paneltitle{{font-size:19px;font-weight:bold}} .ylabel{{font-size:17px}}</style>
<text class="title" x="{width/2}" y="35" text-anchor="middle">GRPO training loss diagnostics</text>
{panel("Policy-gradient loss (actor/pg_loss)", pg, pg_ma, top1, "#315f92", "PG loss")}
{panel("Raw KL divergence (actor/kl_loss)", kl, kl_ma, top2, "#b22222", "Raw KL")}
{''.join(ticks)}
<text x="{width/2}" y="{height-20}" text-anchor="middle" font-size="18">Optimizer update</text>
<line x1="{width-350}" y1="57" x2="{width-315}" y2="57" stroke="#2e8b57" stroke-width="2" stroke-dasharray="7,5"/><text x="{width-305}" y="62">Checkpoint step {args.checkpoint_step}</text>
<line x1="{width-350}" y1="80" x2="{width-315}" y2="80" stroke="#b22222" stroke-width="2" stroke-dasharray="7,5"/><text x="{width-305}" y="85">Stopped near step {args.stop_step}</text>
<text x="{left}" y="{height-48}" font-size="13">Lines: per-step metric (faint) and {args.window}-step moving average (solid). Raw KL is not yet multiplied by kl_loss_coef=0.001.</text>
</svg>''', encoding="utf-8")
    print(f"wrote={args.output} rows={len(rows)} steps={int(min(steps))}-{int(max(steps))}")


if __name__ == "__main__":
    main()
