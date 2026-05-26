import argparse
from pathlib import Path

import h5py
import numpy as np

PDG_PION_PLUS = 211
DIRECTION_90DEG = np.array([0.0, 1.0, 0.0])  # 90 degrees, adjust if needed


def parse_arguments(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert numpy arrays to showerdata-compatible HDF5 format"
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Path to folder of numpy arrays (points.npy, energy.npy and mint.npy)",
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="Path to the output HDF5 file",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> None:
    parsed_args = parse_arguments(args)

    condE = np.load(parsed_args.folder / "cond_energy.npy")  # (N, L)
    energy = np.load(parsed_args.folder / "energy.npy").astype(np.float32)  # (N, L)
    min_time = np.load(parsed_args.folder / "mint.npy").astype(np.float32)  # (N, L)
    points = np.load(parsed_args.folder / "points.npy").astype(np.int64)  # (N, L, 3)

    assert energy.shape == min_time.shape, (
        f"Shape mismatch: energy {energy.shape} vs min_time {min_time.shape}"
    )

    n_showers, n_layers = energy.shape
    print(f"Loaded {n_showers} showers, {n_layers} layers")

    # fixed fields
    directions = np.tile(DIRECTION_90DEG, (n_showers, 1))  # (N, 3)
    labels = np.full(n_showers, 0, dtype=np.int64)  # single class: pion+

    print(f"PDG: {PDG_PION_PLUS} (pion+), label: 0")
    print(f"Direction: {DIRECTION_90DEG} (90 degrees)")

    with h5py.File(parsed_args.output_file, "w") as f:
        f.create_dataset("directions", data=directions)
        f.create_dataset("energies", data=condE)
        f.create_dataset("labels", data=labels)
        f.create_dataset("num_points", data=points)
        f.create_dataset("min_time_per_layer", data=min_time)
        f.create_dataset("energy_per_layer", data=energy)
        f.attrs["pdg"] = PDG_PION_PLUS
        f.attrs["label_list"] = [PDG_PION_PLUS]

    print(f"Saved to {parsed_args.output_file}")


if __name__ == "__main__":
    main()
