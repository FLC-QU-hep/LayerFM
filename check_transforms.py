"""
Visualise how different num_points transforms behave on the actual dataset.
Produces check_transforms.pdf in the same directory.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm

from layerfm.preprocessing import compose

# ── config ────────────────────────────────────────────────────────────────────
DATA_FILE = "../pion-clouds/files/npy_array/points_HGx9new/input_for_pcFM.h5"
N_SAMPLES = 1_000
LAYERS_TO_SHOW = [5, 20, 40, 60, 74]  # early / mid / core / late / edge

TRANSFORMS = {
    "Log + StandardScaler\n(logPOINTS run)": [
        ["Log", {"alpha": 0.5}],
        ["StandardScaler", {"shape": [1, 78]}],
    ],
    "Dequantize + QuantileTransformer\n(quantileALL run)": [
        ["Dequantize", {}],
        ["QuantileTransformer", {"n_quantiles": 1000, "shape": [1, 78]}],
    ],
    "Dequantize + Log + QuantileTransformer\n(proposed)": [
        ["Dequantize", {}],
        ["Log", {"alpha": 0.5}],
        ["QuantileTransformer", {"n_quantiles": 1000, "shape": [1, 78]}],
    ],
}

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "check_transforms.pdf")

# ── load data ─────────────────────────────────────────────────────────────────
data_path = os.path.join(os.path.dirname(__file__), DATA_FILE)
print(f"Loading {N_SAMPLES} samples from {data_path} ...")
with h5py.File(data_path, "r") as f:
    num_points = torch.tensor(
        f["num_points"][:N_SAMPLES], dtype=torch.float32
    )  # (N, 78)
print("Done.\n")

# ── build figure ──────────────────────────────────────────────────────────────
n_transforms = len(TRANSFORMS)
n_layers = len(LAYERS_TO_SHOW)

# Layout:
#   rows   = one per selected layer  +  1 summary row at the bottom
#   cols   = raw  +  one per transform
fig, axes = plt.subplots(
    n_layers + 1,
    n_transforms + 1,
    figsize=(4.5 * (n_transforms + 1), 3.5 * (n_layers + 1)),
)
fig.suptitle(
    "num_points transformations — distribution in latent space", fontsize=14, y=1.01
)

# reference Gaussian
x_gauss = np.linspace(-4, 4, 100)
y_gauss = norm.pdf(x_gauss)

# ── per-layer rows ────────────────────────────────────────────────────────────
for row_i, layer in enumerate(LAYERS_TO_SHOW):
    raw_vals = num_points[:, layer].numpy()
    frac_zero = (raw_vals == 0).mean()

    # --- raw column ---
    ax = axes[row_i, 0]
    ax.hist(raw_vals, bins=60, density=True, color="steelblue", alpha=0.7)
    ax.set_title(f"Layer {layer}  (raw)\nzeros: {frac_zero:.0%}", fontsize=9)
    ax.set_xlabel("num_points")
    ax.set_ylabel("density")

    # --- transform columns ---
    for col_i, (name, trafo_cfg) in enumerate(TRANSFORMS.items()):
        trafo = compose(trafo_cfg)
        with torch.no_grad():
            t_vals = trafo.fit(num_points.clone())[:, layer].numpy()

        ax = axes[row_i, col_i + 1]
        # clip extreme outliers for display
        lo, hi = np.percentile(t_vals[np.isfinite(t_vals)], [0.5, 99.5])
        t_plot = t_vals[np.isfinite(t_vals)]
        t_plot = t_plot[(t_plot >= lo) & (t_plot <= hi)]

        ax.hist(t_plot, bins=30, density=True, color="mediumpurple", alpha=0.7)
        ax.plot(x_gauss, y_gauss, "r--", lw=1.5, label="N(0,1)")
        if row_i == 0:
            ax.set_title(name, fontsize=8)
        ax.set_xlabel("transformed value")
        ax.set_ylabel("density")

        # annotate mean / std of the transformed values
        mu = np.nanmean(t_vals)
        std = np.nanstd(t_vals)
        ax.text(
            0.97,
            0.95,
            f"μ={mu:.2f}\nσ={std:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
        )

# ── summary row: ALL layer values pooled ─────────────────────────────────────
# raw column: total num_points per shower
ax_raw_sum = axes[n_layers, 0]
total_raw = num_points.sum(dim=1).numpy()
ax_raw_sum.hist(total_raw, bins=80, density=True, color="steelblue", alpha=0.7)
ax_raw_sum.set_title("Total num_points\nper shower (raw)", fontsize=9)
ax_raw_sum.set_xlabel("sum over 78 layers")
ax_raw_sum.set_ylabel("density")

for col_i, (name, trafo_cfg) in enumerate(TRANSFORMS.items()):
    trafo = compose(trafo_cfg)
    with torch.no_grad():
        t_all = trafo.fit(num_points.clone())  # (N, 78)

    # pool ALL per-layer transformed values (excluding inactive zeros mapped to same spot)
    t_flat = t_all.numpy().ravel()
    t_flat = t_flat[np.isfinite(t_flat)]
    lo, hi = np.percentile(t_flat, [0.1, 99.9])
    t_plot = t_flat[(t_flat >= lo) & (t_flat <= hi)]

    ax = axes[n_layers, col_i + 1]
    ax.hist(t_plot, bins=80, density=True, color="mediumpurple", alpha=0.7)
    ax.plot(x_gauss, y_gauss, "r--", lw=1.5, label="N(0,1)")
    ax.set_title("All layer values pooled\n(after transform)", fontsize=9)
    ax.set_xlabel("transformed value")
    ax.set_ylabel("density")

    mu_all = np.nanmean(t_flat)
    std_all = np.nanstd(t_flat)
    ax.text(
        0.97,
        0.95,
        f"μ={mu_all:.2f}\nσ={std_all:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
    )
    ax.legend(fontsize=7)

fig.tight_layout()
fig.savefig(OUTPUT_FILE, bbox_inches="tight")
print(f"Saved to {OUTPUT_FILE}")
