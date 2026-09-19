#!/usr/bin/env python3
"""Render a GRPO reward curve from a local offline W&B run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


def read_reward_history(run_dir: Path) -> list[tuple[int, float]]:
    run_file = next(run_dir.glob("run-*.wandb"), None)
    if run_file is None:
        raise FileNotFoundError(f"No run-*.wandb file under {run_dir}")

    store = DataStore()
    store.open_for_scan(str(run_file))
    rows: list[tuple[int, float]] = []
    try:
        while True:
            data = store.scan_data()
            if data is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            if record.WhichOneof("record_type") != "history":
                continue

            values: dict[str, object] = {}
            for item in record.history.item:
                key = item.key or "/".join(item.nested_key)
                if key:
                    values[key] = json.loads(item.value_json)
            if "_step" in values and "critic/rewards/mean" in values:
                rows.append((int(values["_step"]), float(values["critic/rewards/mean"])))
    finally:
        store.close()
    return rows


def moving_average(values: list[float], window: int) -> list[float]:
    return [sum(values[max(0, i - window + 1) : i + 1]) / min(i + 1, window) for i in range(len(values))]


def render_svg(
    steps: tuple[int, ...], rewards: tuple[float, ...], smoothed: list[float], output: Path, checkpoint_step: int, stop_step: int, window: int
) -> None:
    width, height = 1200, 620
    left, right, top, bottom = 90, 40, 70, 90
    plot_width, plot_height = width - left - right, height - top - bottom
    minimum_step, maximum_step = min(steps), max(max(steps), stop_step) + 8

    def x(step: int) -> float:
        return left + (step - minimum_step) / (maximum_step - minimum_step) * plot_width

    def y(value: float) -> float:
        return top + (1.0 - value) * plot_height

    def polyline(values: list[float] | tuple[float, ...]) -> str:
        return " ".join(f"{x(step):.2f},{y(value):.2f}" for step, value in zip(steps, values))

    grid = []
    labels = []
    for tick in range(0, 5):
        value = tick / 4
        yy = y(value)
        grid.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{width-right}" y2="{yy:.2f}" stroke="#d9e1e8"/>')
        labels.append(f'<text x="{left-12}" y="{yy+5:.2f}" text-anchor="end">{value:.2f}</text>')
    x_tick_step = 100
    for tick in range(((minimum_step + x_tick_step - 1) // x_tick_step) * x_tick_step, maximum_step + 1, x_tick_step):
        xx = x(tick)
        grid.append(f'<line x1="{xx:.2f}" y1="{top}" x2="{xx:.2f}" y2="{height-bottom}" stroke="#edf1f4"/>')
        labels.append(f'<text x="{xx:.2f}" y="{height-bottom+28}" text-anchor="middle">{tick}</text>')

    output.write_text(
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<style>text{{font-family:Arial,sans-serif;font-size:16px;fill:#263238}} .title{{font-size:24px;font-weight:bold}} .label{{font-size:18px}}</style>
<text class="title" x="{width/2}" y="35" text-anchor="middle">GRPO outcome-only training reward</text>
{''.join(grid)}
<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#73808c"/>
<polyline points="{polyline(rewards)}" fill="none" stroke="#76a5d1" stroke-width="1.2" opacity="0.58"/>
<polyline points="{polyline(smoothed)}" fill="none" stroke="#0b4f8a" stroke-width="3"/>
<line x1="{x(checkpoint_step):.2f}" y1="{top}" x2="{x(checkpoint_step):.2f}" y2="{height-bottom}" stroke="#2e8b57" stroke-width="2" stroke-dasharray="7,5"/>
<line x1="{x(stop_step):.2f}" y1="{top}" x2="{x(stop_step):.2f}" y2="{height-bottom}" stroke="#b22222" stroke-width="2" stroke-dasharray="7,5"/>
{''.join(labels)}
<text class="label" x="{width/2}" y="{height-24}" text-anchor="middle">Optimizer update</text>
<text class="label" x="25" y="{height/2}" text-anchor="middle" transform="rotate(-90 25 {height/2})">Reward mean (4 rollouts/group)</text>
<rect x="{width-365}" y="{top+15}" width="325" height="104" fill="white" stroke="#b8c3cc"/>
<line x1="{width-345}" y1="{top+38}" x2="{width-305}" y2="{top+38}" stroke="#76a5d1" stroke-width="2"/><text x="{width-295}" y="{top+44}">Per-step reward mean</text>
<line x1="{width-345}" y1="{top+64}" x2="{width-305}" y2="{top+64}" stroke="#0b4f8a" stroke-width="3"/><text x="{width-295}" y="{top+70}">{window}-step moving average</text>
<line x1="{width-345}" y1="{top+90}" x2="{width-305}" y2="{top+90}" stroke="#2e8b57" stroke-width="2" stroke-dasharray="7,5"/><text x="{width-295}" y="{top+96}">Checkpoint step {checkpoint_step}</text>
</svg>''',
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output SVG path")
    parser.add_argument("--checkpoint-step", type=int, default=500)
    parser.add_argument("--stop-step", type=int, default=557)
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()

    rows = read_reward_history(args.wandb_run)
    if not rows:
        raise RuntimeError("No critic/rewards/mean history found")
    steps, rewards = zip(*rows)
    smoothed = moving_average(list(rewards), args.window)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    render_svg(steps, rewards, smoothed, args.output, args.checkpoint_step, args.stop_step, args.window)
    print(f"wrote={args.output}")
    print(f"steps={min(steps)}-{max(steps)} rows={len(rows)} overall_mean={sum(rewards) / len(rewards):.6f}")


if __name__ == "__main__":
    main()
