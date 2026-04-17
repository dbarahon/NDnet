#!/usr/bin/env python3
"""
Plot training/validation loss and loss components vs epoch from a JSONL training log.

Usage:
  python plot_trainlog.py train_log.jsonl out.png

Notes:
- Handles "broken" JSON lines where a single JSON object may be split across multiple lines
  (as in your snippet).
- Aggregates by epoch using mean over all events within an epoch.
- Validation event names vary by codebase; edit VAL_EVENTS if needed.
"""

import sys
import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# --- customize if your log uses different names ---
TRAIN_EVENTS = {"loss_step", "train_loss", "train_epoch", "loss_train"}
VAL_EVENTS   = {"val_loss", "valid_loss", "validation_loss", "loss_val", "eval_loss", "loss_eval"}

LOSS_KEYS = ["L_site", "L_nd", "L_anchor", "L_grad", "L_sig", "L_nd_gauss", "L_campaign", "L_campaign_gauss"]


def iter_json_objects(path):
    """
    Yield JSON dicts from a file that is "JSONL-ish".
    If JSON objects are split across multiple lines, buffer until valid.
    """
    buf = ""
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # Append and try parsing
            if buf:
                buf += " " + line
            else:
                buf = line

            try:
                obj = json.loads(buf)
                yield obj
                buf = ""
            except json.JSONDecodeError:
                # Not complete yet; keep buffering
                continue

    # If leftover buffer can't be parsed, ignore (or raise if you prefer)
    if buf.strip():
        try:
            yield json.loads(buf)
        except json.JSONDecodeError:
            pass


def mean_or_nan(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    return float(np.nanmean(x))


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_trainlog.py train_log.jsonl [out.png]")
        sys.exit(2)

    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) >= 3 else "train_log_plots.png"

    # Collect values by epoch
    train_loss_by_epoch = defaultdict(list)
    val_loss_by_epoch   = defaultdict(list)
    comps_by_epoch      = {k: defaultdict(list) for k in LOSS_KEYS}

    # Some logs store loss under "loss" for both train/val; we use event name to decide.
    for rec in iter_json_objects(in_path):
        epoch = rec.get("epoch", None)
        if epoch is None:
            continue

        event = rec.get("event", "")
        # Decide train vs val
        is_val = (event in VAL_EVENTS) or ("val" in event.lower()) or ("valid" in event.lower()) or ("eval" in event.lower())
        is_train = (event in TRAIN_EVENTS) or (event == "loss_step") or ("train" in event.lower())

        # Loss value: prefer explicit train_loss/val_loss keys, else "loss"
        if is_val:
            v = rec.get("val_loss", rec.get("validation_loss", rec.get("loss", None)))
            if v is not None:
                val_loss_by_epoch[int(epoch)].append(float(v))
        elif is_train:
            v = rec.get("train_loss", rec.get("loss", None))
            if v is not None:
                train_loss_by_epoch[int(epoch)].append(float(v))

            # components (training-side)
            for k in LOSS_KEYS:
                if k in rec and rec[k] is not None:
                    comps_by_epoch[k][int(epoch)].append(float(rec[k]))

    # Build epoch axis
    all_epochs = sorted(set(train_loss_by_epoch.keys()) | set(val_loss_by_epoch.keys()))
    if not all_epochs:
        raise RuntimeError("No epochs found. Check that your file has an 'epoch' field.")

    train_loss = [mean_or_nan(train_loss_by_epoch[e]) for e in all_epochs]
    val_loss   = [mean_or_nan(val_loss_by_epoch[e]) for e in all_epochs]

    comps = {k: [mean_or_nan(comps_by_epoch[k][e]) for e in all_epochs] for k in LOSS_KEYS}

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # Plot 1: train vs val
    ax = axes[0]
    ax.plot(all_epochs, train_loss, label="train_loss")
    if not np.all(np.isnan(val_loss)):
        ax.plot(all_epochs, val_loss, label="val_loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Train/Val Loss vs Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 2: components
    ax = axes[1]
    for k in LOSS_KEYS:
        y = comps[k]
        if not np.all(np.isnan(y)):
            ax.plot(all_epochs, y, label=k)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss component")
    ax.set_title("Loss Components vs Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle(in_path, fontsize=10)
    fig.savefig(out_path, dpi=200)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

