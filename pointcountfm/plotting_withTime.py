import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml

# generation

# python pointcountfm/inference_cond_file.py results/20260527_140728_PointCountFM_withTime/compiled.pt
# /data/dust/user/mmozzani/pion-clouds/dataset/hdf5_HGx9new/all_interactions_pions_regular_ECAL+HCAL_10-90GeV_10.slcio.root_with_time.hdf5
# -o results/20260527_140728_PointCountFM_withTime/output_pcFM.h5 --pdg-codes 211

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, help="Path to the result folder.")
params = parser.parse_args()
path = params.path

name = "output_pcFM.h5"

confg_path = path + "conf.yaml"
with open(confg_path) as file:
    config = yaml.safe_load(file)

generated_pl = h5py.File(
    path + name,
    "r",
)
geant4_pl = h5py.File(
    config["data"]["data_file"],
    "r",
)

print("generated keys: ", generated_pl.keys())
print("geant4 keys: ", geant4_pl.keys())

gen_E0 = generated_pl["energies"][:].squeeze()
# gen_E0 = generated_pl["energy"][:].squeeze()
g4_E0 = geant4_pl["energies"][:].squeeze()
print("incident energy generated: ", gen_E0.shape)
print("incident energy geant4: ", g4_E0.shape)

if g4_E0.shape[0] > gen_E0.shape[0]:
    print("selecting only the same amount of Geant4 events as generated events")
    idx_list = []
    for E in np.unique(gen_E0):
        n_gen = np.sum(gen_E0 == E)  # how many generated events at energy E
        g4_indices = np.where(g4_E0 == E)[0]  # all Geant4 events at energy E

        idx_list.append(g4_indices[:n_gen])  # take exactly the same amount

    idx = np.sort(np.concatenate(idx_list))
    geant4_energies = g4_E0[idx]
    g4_data = geant4_pl["num_points"][idx]

elif gen_E0.shape[0] < g4_E0.shape[0]:
    print(
        "error: more Geant4 events than generated events, this should not happen for plotting"
    )
else:
    print("same amount of events in generated and Geant4, no need to select")
    geant4_energies = g4_E0
    g4_data = geant4_pl["num_points"]


# gen_data = generated_pl["num_points"][:]
gen_data = generated_pl["observables"]["num_points_per_layer"][:]

g4_flat = g4_data.sum(axis=1).flatten()
gen_flat = gen_data.sum(axis=1).flatten()

# bins = np.histogram_bin_edges(np.concatenate([g4_flat, gen_flat]), bins=50)
bins = 50

os.makedirs(path + "plots", exist_ok=True)
plt.figure(figsize=(6, 5))

plt.hist(g4_flat, bins=bins, density=True, color="lightgray", alpha=0.8, label="Geant4")

plt.hist(
    gen_flat,
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
plt.xlim([g4_flat.min(), g4_flat.max()])
# plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig(path + "plots/{}.png".format(name.split(".")[0]))
