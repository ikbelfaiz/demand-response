from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def _svg_panels(output: str | Path, panels: list[tuple[str, list[tuple[str, np.ndarray]]]],
                ylabel: str = "") -> None:
    width, panel_h, left, right = 1200, 190, 85, 25
    height = 45 + panel_h * len(panels)
    colors = ("#1769aa", "#d1495b", "#2a9d8f", "#7b2cbf")
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:sans-serif;font-size:12px}.title{font-size:14px;font-weight:bold}</style>']
    for pidx, (title, series) in enumerate(panels):
        top = 30 + pidx * panel_h; plot_h = panel_h - 45; plot_w = width - left - right
        finite = np.concatenate([x[np.isfinite(x)] for _, x in series if np.isfinite(x).any()]) if series else np.array([0.])
        ymin, ymax = (float(finite.min()), float(finite.max())) if len(finite) else (0., 1.)
        if ymax <= ymin: ymax = ymin + 1
        parts += [f'<text class="title" x="{left}" y="{top-8}">{title}</text>',
                  f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#777"/>',
                  f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#777"/>',
                  f'<text x="5" y="{top+15}">{ymax:.3g}</text>', f'<text x="5" y="{top+plot_h}">{ymin:.3g}</text>']
        for sidx, (name, values) in enumerate(series):
            step = max(1, int(np.ceil(len(values) / 2000)))
            idx = np.arange(0, len(values), step); vals = values[idx]
            good = np.isfinite(vals)
            if good.any():
                xs = left + plot_w * idx[good] / max(len(values)-1, 1)
                ys = top + plot_h * (1 - (vals[good] - ymin) / (ymax - ymin))
                points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
                color = colors[sidx % len(colors)]
                parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.2"/>')
                parts.append(f'<text x="{left+plot_w-130}" y="{top+14+sidx*14}" fill="{color}">{name}</text>')
    parts.append('</svg>')
    Path(output).write_text("\n".join(parts), encoding="utf-8")


def plot_history(history_csv: str | Path, output: str | Path) -> None:
    with Path(history_csv).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    epochs = [int(x["epoch"]) + 1 for x in rows]
    train = [float(x["train_loss"]) for x in rows]
    val = [float(x["validation_loss"]) for x in rows]
    plt = _plt()
    if plt is None:
        _svg_panels(output, [("Training and validation loss", [("training", np.asarray(train)), ("validation", np.asarray(val))])])
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, train, marker="o", label="training"); ax.plot(epochs, val, marker="o", label="validation")
    ax.set(xlabel="Epoch", ylabel="Multi-task loss", title="Training and validation loss")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(output, dpi=150); plt.close(fig)


def plot_inference(prediction_csv: str | Path, output: str | Path) -> None:
    with Path(prediction_csv).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    x = np.arange(len(rows)); aggregate = np.asarray([float(r["aggregate_power_w"]) for r in rows])
    names = ("ac", "water_heater", "washing_machine")
    plt = _plt()
    panel_data = [("Aggregate power (W)", [("aggregate", aggregate)])]
    prepared = []
    for name in names:
        pred = np.asarray([float(r[f"{name}_power_w_predicted"]) if r[f"{name}_power_w_predicted"] else np.nan for r in rows])
        true = np.asarray([float(r[f"{name}_power_w_true"]) if r[f"{name}_power_w_true"] else np.nan for r in rows])
        prob = np.asarray([float(r[f"{name}_activity_probability"]) if r[f"{name}_activity_probability"] else np.nan for r in rows])
        reference = np.asarray([float(r[f"{name}_activity_reference"]) if r[f"{name}_activity_reference"] else np.nan for r in rows])
        prepared.append((name, pred, true, prob, reference))
        panel_data.append((("Primary AC" if name == "ac" else name.replace("_", " ")) + " power (W)", [("true", true), ("predicted", pred)]))
    for name, _, _, prob, reference in prepared:
        panel_data.append((name.replace("_", " ") + " activity", [("reference", reference), ("probability", prob)]))
    if plt is None:
        _svg_panels(output, panel_data)
        return
    fig, axes = plt.subplots(7, 1, figsize=(13, 13), sharex=True)
    axes[0].plot(x, aggregate, lw=.8); axes[0].set_ylabel("Aggregate W")
    for j, (name, pred, true, prob, reference) in enumerate(prepared):
        axes[1 + j].plot(x, true, lw=.8, alpha=.75, label="true")
        axes[1 + j].plot(x, pred, lw=.8, label="predicted")
        axes[1 + j].set_ylabel(("Primary AC" if name == "ac" else name.replace("_", " ")) + " W")
        axes[1 + j].legend(loc="upper right")
        axes[4 + j].plot(x, reference, lw=.7, alpha=.55, label="reference")
        axes[4 + j].plot(x, prob, lw=.8, label="probability"); axes[4 + j].axhline(.5, color="black", ls="--", lw=.7)
        axes[4 + j].set_ylim(-.05, 1.05); axes[4 + j].set_ylabel(name + " p(active)")
        axes[4 + j].legend(loc="upper right")
    axes[-1].set_xlabel("Minute index"); fig.suptitle("Standalone causal NILM predictions")
    fig.tight_layout(); fig.savefig(output, dpi=150); plt.close(fig)


def plot_household_errors(metrics_csv: str | Path, output: str | Path) -> None:
    with Path(metrics_csv).open(encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["stratum"] == "overall" and r["household"] != "ALL"]
    households = sorted({r["household"] for r in rows}); names = ("ac", "water_heater", "washing_machine")
    plt = _plt(); x = np.arange(len(households))
    series = []
    for name in names:
        lookup = {r["household"]: float(r["power_mae_w"]) for r in rows if r["appliance"] == name}
        series.append((name, np.asarray([lookup.get(h, np.nan) for h in households])))
    if plt is None:
        _svg_panels(output, [("Per-household appliance power MAE (W)", series)])
        return
    fig, ax = plt.subplots(figsize=(12, 5)); width = .25
    for j, name in enumerate(names):
        lookup = {r["household"]: float(r["power_mae_w"]) for r in rows if r["appliance"] == name}
        ax.bar(x + (j - 1) * width, [lookup.get(h, np.nan) for h in households], width, label=name)
    ax.set_xticks(x, households, rotation=45); ax.set_ylabel("Power MAE (W)")
    ax.set_title("Per-household appliance error"); ax.legend(); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(output, dpi=150); plt.close(fig)
