import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import binned_statistic

# run this code: python pointcountfm/plotting_withTandE.py results/20260603_141325_LayerFM_PandT_multiLoss

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, help="Path to the result folder.")
params = parser.parse_args()
path = params.path
path = path if path.endswith("/") else path + "/"

name = "output_FM.h5"

# generation
# subprocess.run(
#     [
#         "python",
#         "pointcountfm/inference_cond_file.py",
#         f"{path}compiled.pt",
#         "../pion-clouds/dataset/hdf5_HGx9new/all_interactions_pions_regular_ECAL+HCAL_10-90GeV_10.slcio.root_with_time.hdf5",
#         "-o",
#         f"{path}{name}",
#         "--pdg-codes",
#         "211",
#     ]
# )


def plot_num_points_multi_energy(
    gen_flat_points,
    gen_E0,
    g4_flat_points,
    geant4_energies,
    path,
    name,
    energies=[20, 50, 80],
):
    colors = ["royalblue", "purple", "crimson"]
    energy_configs = [(e, 0.5) for e in energies]  # (center, half-window)

    plt.figure(figsize=(8, 6))
    bins = np.linspace(g4_flat_points.min(), g4_flat_points.max(), 50)
    for (energy, half_window), color in zip(energy_configs, colors):
        mask_gen = (gen_E0 >= energy - half_window) & (gen_E0 <= energy + half_window)
        mask_g4 = (geant4_energies >= energy - half_window) & (
            geant4_energies <= energy + half_window
        )

        gen_pts = gen_flat_points[mask_gen]
        g4_pts = g4_flat_points[mask_g4]

        plt.hist(
            g4_pts,
            bins=bins,
            density=True,
            color=color,
            alpha=0.3,
            label=f"Geant4 {energy} GeV",
        )
        plt.hist(
            gen_pts,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            color=color,
            label=f"Generated {energy} GeV",
        )

    plt.title("Total Num Points at 20, 50 & 80 GeV")
    plt.xlabel("Total Num Points")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path + "plots/{}_multi_energy.png".format(name.split(".")[0]))
    plt.close()


confg_path = path + "conf.yaml"
with open(confg_path) as file:
    config = yaml.safe_load(file)

generated_pl = h5py.File(path + name, "r")
geant4_pl = h5py.File(config["data"]["data_file"], "r")

print("generated keys: ", generated_pl.keys())
print("geant4 keys: ", geant4_pl.keys())

gen_E0 = generated_pl["energies"][:].squeeze()
g4_E0 = geant4_pl["energies"][:].squeeze()
print("incident energy generated: ", gen_E0.shape)
print("incident energy geant4: ", g4_E0.shape)

if g4_E0.shape[0] > gen_E0.shape[0]:
    print("selecting only the same amount of Geant4 events as generated events")
    idx_list = []
    for E in np.unique(gen_E0):
        n_gen = np.sum(gen_E0 == E)
        g4_indices = np.where(g4_E0 == E)[0]
        idx_list.append(g4_indices[:n_gen])
    idx = np.sort(np.concatenate(idx_list))
    geant4_energies = g4_E0[idx]
    g4_data_points = geant4_pl["num_points"][idx]
    if "min_time_per_layer" in geant4_pl:
        g4_data_time = geant4_pl["min_time_per_layer"][idx]
    if "energy_per_layer" in geant4_pl:
        g4_data_energy = geant4_pl["energy_per_layer"][idx]
elif gen_E0.shape[0] < g4_E0.shape[0]:
    print(
        "error: more Geant4 events than generated events, this should not happen for plotting"
    )
else:
    print("same amount of events in generated and Geant4, no need to select")
    geant4_energies = g4_E0
    g4_data_points = geant4_pl["num_points"][:]
    if "min_time_per_layer" in geant4_pl:
        g4_data_time = geant4_pl["min_time_per_layer"][:]
    if "energy_per_layer" in geant4_pl:
        g4_data_energy = geant4_pl["energy_per_layer"][:]

if "min_time_per_layer" in geant4_pl:
    g4_data_time[np.isnan(g4_data_time)] = 0.0

if "energy_per_layer" in geant4_pl:
    g4_data_energy[np.isnan(g4_data_energy)] = 0.0

gen_data_points = generated_pl["num_points_per_layer"][:]
print("gen points ", gen_data_points.min(), gen_data_points.max())

if "min_time_per_layer" in generated_pl:
    gen_data_time = generated_pl["min_time_per_layer"][:]
    print("gen min time ", gen_data_time.min(), gen_data_time.max())

if "energy_per_layer" in generated_pl:
    gen_data_energy = generated_pl["energy_per_layer"][:]
    print("gen energy per layer ", gen_data_energy.min(), gen_data_energy.max())

os.makedirs(path + "plots", exist_ok=True)

# ── Total num points ──────────────────────────────────────────────────────────
g4_flat_points = g4_data_points.sum(axis=1).flatten()
gen_flat_points = gen_data_points.sum(axis=1).flatten()
bins = np.linspace(g4_flat_points.min(), g4_flat_points.max(), 50)

plt.figure(figsize=(6, 5))
plt.hist(
    g4_flat_points,
    bins=bins,
    density=True,
    color="lightgray",
    alpha=0.8,
    label="Geant4",
)
plt.hist(
    gen_flat_points,
    bins=bins,
    density=True,
    histtype="step",
    linewidth=2,
    color="purple",
    label="Generated",
)
plt.title("Total Num Points")
plt.xlabel("Total Num Points")
plt.ylabel("Density")
plt.xlim([g4_flat_points.min(), g4_flat_points.max()])
plt.legend()
plt.tight_layout()
plt.savefig(path + "plots/{}.png".format(name.split(".")[0]))
plt.close()

plot_num_points_multi_energy(
    gen_flat_points,
    gen_E0,
    g4_flat_points,
    geant4_energies,
    path,
    name,
    energies=[20, 50, 80],
)

# ── Min time per layer ────────────────────────────────────────────────────────
if "min_time_per_layer" in geant4_pl and "min_time_per_layer" in generated_pl:
    plt.figure(figsize=(6, 5))
    value_ = 0.01
    # bins_log = np.logspace(
    #     np.log10(g4_data_time[g4_data_time > 0].min() + value_),
    #     np.log10(g4_data_time.max() + value_),
    #     50,
    # )
    bins = np.linspace(g4_data_time.min(), g4_data_time.max(), 50)
    plt.hist(
        g4_data_time.flatten() + value_,
        bins=bins,
        density=True,
        color="lightgray",
        alpha=0.8,
        label="Geant4",
    )
    plt.hist(
        gen_data_time.flatten() + value_,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        color="purple",
        label="Generated",
    )
    plt.title("Min Time per Layer")
    plt.xlabel("Min Time per Layer")
    plt.ylabel("Density")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path + "plots/{}_min_time.png".format(name.split(".")[0]))
    plt.close()

    # Min time vs num points (binned)
    g4_points = g4_data_points.flatten()
    gen_points = gen_data_points.flatten()
    g4_time = g4_data_time.flatten()
    gen_time = gen_data_time.flatten()

    bins = np.linspace(0, g4_points.max(), 50)

    g4_mean_time, edges, _ = binned_statistic(
        g4_points, g4_time, statistic="mean", bins=bins
    )
    gen_mean_time, _, _ = binned_statistic(
        gen_points, gen_time, statistic="mean", bins=bins
    )

    g4_std_time, _, _ = binned_statistic(g4_points, g4_time, statistic="std", bins=bins)
    gen_std_time, _, _ = binned_statistic(
        gen_points, gen_time, statistic="std", bins=bins
    )

    g4_count, _, _ = binned_statistic(g4_points, g4_time, statistic="count", bins=bins)
    gen_count, _, _ = binned_statistic(
        gen_points, gen_time, statistic="count", bins=bins
    )

    # Standard error of the mean: σ / √n
    g4_sem = g4_std_time / np.sqrt(np.maximum(g4_count, 1))
    gen_sem = gen_std_time / np.sqrt(np.maximum(gen_count, 1))

    centers = 0.5 * (edges[:-1] + edges[1:])

    plt.figure(figsize=(7, 5))

    plt.plot(centers, g4_mean_time, color="black", linewidth=2, label="Geant4")
    plt.fill_between(
        centers,
        g4_mean_time - g4_sem,
        g4_mean_time + g4_sem,
        color="black",
        alpha=0.2,
    )

    plt.plot(centers, gen_mean_time, color="purple", linewidth=2, label="Generated")
    plt.fill_between(
        centers,
        gen_mean_time - gen_sem,
        gen_mean_time + gen_sem,
        color="purple",
        alpha=0.2,
    )

    plt.xlabel("Num Points per Layer")
    plt.ylabel("Mean Min Time per Layer")
    plt.title("Min Time vs Num Points")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path + f"plots/{name.split('.')[0]}_binned_time_vs_points.png")
    plt.close()

# ── Energy per layer ──────────────────────────────────────────────────────────
if "energy_per_layer" in geant4_pl and "energy_per_layer" in generated_pl:
    # Total energy per event
    g4_flat_energy = g4_data_energy.sum(axis=1).flatten()
    gen_flat_energy = gen_data_energy.sum(axis=1).flatten()
    bins = np.linspace(g4_flat_energy.min(), g4_flat_energy.max(), 50)

    plt.figure(figsize=(6, 5))
    plt.hist(
        g4_flat_energy,
        bins=bins,
        density=True,
        color="lightgray",
        alpha=0.8,
        label="Geant4",
    )
    plt.hist(
        gen_flat_energy,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        color="purple",
        label="Generated",
    )
    plt.title("Total Energy per Event")
    plt.xlabel("Total Energy per Event")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path + f"plots/{name.split('.')[0]}_total_energy.png")
    plt.close()

    # Energy per layer distribution
    plt.figure(figsize=(6, 5))
    bins = np.linspace(g4_data_energy.min(), g4_data_energy.max(), 50)
    plt.hist(
        g4_data_energy.flatten(),
        bins=bins,
        density=True,
        color="lightgray",
        alpha=0.8,
        label="Geant4",
    )
    plt.hist(
        gen_data_energy.flatten(),
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        color="purple",
        label="Generated",
    )
    plt.title("Energy per Layer")
    plt.xlabel("Energy per Layer")
    plt.ylabel("Density")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path + f"plots/{name.split('.')[0]}_energy_per_layer.png")
    plt.close()

    # Energy per layer vs num points (binned)
    g4_points = g4_data_points.flatten()
    gen_points = gen_data_points.flatten()
    g4_energy = g4_data_energy.flatten()
    gen_energy = gen_data_energy.flatten()
    bins = np.linspace(0, g4_points.max(), 50)
    g4_mean_energy, edges, _ = binned_statistic(
        g4_points, g4_energy, statistic="mean", bins=bins
    )
    g4_std_energy, _, _ = binned_statistic(
        g4_points, g4_energy, statistic="std", bins=bins
    )
    gen_mean_energy, _, _ = binned_statistic(
        gen_points, gen_energy, statistic="mean", bins=bins
    )
    gen_std_energy, _, _ = binned_statistic(
        gen_points, gen_energy, statistic="std", bins=bins
    )

    g4_count, _, _ = binned_statistic(
        g4_points, g4_energy, statistic="count", bins=bins
    )
    gen_count, _, _ = binned_statistic(
        gen_points, gen_energy, statistic="count", bins=bins
    )
    centers = 0.5 * (edges[:-1] + edges[1:])

    plt.figure(figsize=(7, 5))
    plt.plot(centers, g4_mean_energy, color="black", linewidth=2, label="Geant4")
    plt.fill_between(
        centers,
        g4_mean_energy - g4_std_energy / np.sqrt(np.maximum(g4_count, 1)),
        g4_mean_energy + g4_std_energy / np.sqrt(np.maximum(g4_count, 1)),
        color="black",
        alpha=0.2,
    )
    plt.plot(centers, gen_mean_energy, color="purple", linewidth=2, label="Generated")
    plt.fill_between(
        centers,
        gen_mean_energy - gen_std_energy / np.sqrt(np.maximum(gen_count, 1)),
        gen_mean_energy + gen_std_energy / np.sqrt(np.maximum(gen_count, 1)),
        color="purple",
        alpha=0.2,
    )
    plt.xlabel("Num Points per Layer")
    plt.ylabel("Mean Energy per Layer")
    plt.title("Energy vs Num Points")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path + f"plots/{name.split('.')[0]}_binned_energy_vs_points.png")
    plt.close()

generated_pl.close()
geant4_pl.close()
