import argparse
import time
from collections.abc import Iterable

import h5py
import showerdata
import torch
import yaml


def parse_args(args: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Inference on a model")
    parser.add_argument("model", type=str, help="Model file")
    parser.add_argument(
        "input_file", type=str, help="Input file with conditioning data"
    )
    parser.add_argument(
        "--pdg-codes",
        type=int,
        nargs="+",
        default=[11, -11, 22, 130, 211, -211, 321, -321, 2112, -2112, 2212, -2212],
        help="List of PDG codes corresponding to particle classes.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help=(
            "Output file with generated samples. If not specified, points"
            "per layer will be added to the input file."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Whether to overwrite the output file if it exists."
            "If input and output files are the same, this flag"
            "determines whether to overwrite the dataset within the"
            "file if it exists."
        ),
    )
    return parser.parse_args(args)


def to_labels(pdg_codes: torch.Tensor, pdgs: Iterable[int]) -> torch.Tensor:
    labels = torch.full(pdg_codes.shape, -1, dtype=torch.int64)
    for label, pdg in enumerate(pdgs):
        labels[pdg_codes == pdg] = label
    return labels


def main():
    args = parse_args()
    torch.set_num_threads(1)

    print("Loading model...")
    start = time.time()
    model = torch.jit.load(args.model)
    print(model)
    model.eval()
    print(f"Model loaded in {(time.time() - start) * 1000.0:.1f}ms")

    print("Compiling inference function...")
    start = time.time()

    class InferenceModel(torch.nn.Module):
        def __init__(self, model: torch.jit.ScriptModule):
            super().__init__()
            self.model = model

        def _check_hallucinations(
            self, result: torch.Tensor, condition: torch.Tensor
        ) -> torch.Tensor:
            while result[:, :78].max() > 1e4 or result[:, :78].min() < -1e4:
                print("re-sampling...")
                raw = self.model(condition)
                list_ = []
                for j in range(len(raw)):
                    list_.append(raw[j])
                result = torch.cat(list_, dim=-1)
            return result

        def forward(self, conditions: torch.Tensor) -> torch.Tensor:
            results = []
            for i in range(conditions.size(0)):
                result = self.model(conditions[[i]])
                list_ = []
                for j in range(len(result)):
                    list_.append(result[j])
                results.append(torch.cat(list_, dim=-1))
                results[-1] = self._check_hallucinations(results[-1], conditions[[i]])

            return torch.cat(results, dim=0)

    inference = InferenceModel(model)
    inference = inference.to(torch.float32)
    inference = torch.jit.script(inference)
    # inference = torch.jit.optimize_for_inference(inference)
    print(f"Inference function compiled in {(time.time() - start) * 1000.0:.1f}ms")

    print("Warming up...")
    start = time.time()
    example_conditions = torch.concatenate(
        (
            torch.tensor([[50.0]]),
            torch.nn.functional.one_hot(
                torch.tensor([0]), num_classes=len(args.pdg_codes)
            ).to(torch.float32),
            torch.tensor([[0.0, 1.0, 0.0]]),
        ),
        dim=1,
    )
    print(example_conditions.shape)
    inference(example_conditions)
    inference(example_conditions)
    print(f"Warmup done in {(time.time() - start) * 1000.0:.1f}ms")

    print("Loading conditioning data...")
    start = time.time()
    f = h5py.File(args.input_file, "r")
    if "energies" in f:
        input_len = showerdata.get_file_length(args.input_file)
        cond_data = showerdata.load_inc_particles(
            args.input_file, start=input_len - 50000
        )
        conditions = torch.concatenate(
            (
                torch.from_numpy(cond_data.energies).to(torch.float32),
                torch.nn.functional.one_hot(
                    to_labels(torch.from_numpy(cond_data.pdg), args.pdg_codes),
                    num_classes=len(args.pdg_codes),
                ).to(torch.float32),
                torch.from_numpy(cond_data.directions).to(torch.float32),
            ),
            dim=1,
        )
    else:
        with open("config/config.yaml") as file:
            config = yaml.safe_load(file)
        num_showers = f[list(f.keys())[0]].shape[0]
        cond_data = h5py.File(config["data"]["data_file"], "r")
        conditions = torch.concatenate(
            (
                torch.from_numpy(cond_data["energies"][:num_showers]).to(torch.float32),
                torch.from_numpy(cond_data["labels"][:num_showers])
                .unsqueeze(-1)
                .to(torch.float32),
                torch.from_numpy(cond_data["directions"][:num_showers]).to(
                    torch.float32
                ),
            ),
            dim=-1,
        )

    print(f"Conditioning data loaded in {(time.time() - start) * 1000.0:.1f}ms")
    print("Running inference...")
    start = time.time()
    results = inference(conditions)
    print(f"Inference done in {(time.time() - start):.2f}s")
    print(
        f"Average inference time: {(time.time() - start) / len(conditions) * 1000.0:.1f}ms"
    )

    print("Saving results...")
    results = torch.clamp(results, min=0.0)
    results_points = (results[:, :78] + 0.5).to(torch.int32)
    results_time = results[:, 78:].to(torch.float32)
    results_time = torch.clamp(results_time, max=200.0)
    results_time[results_points == 0] = 0.0
    results_energy = results[:, int(78 * 2) :].to(torch.float32)
    start = time.time()
    output_file = args.output if args.output else args.input_file
    if "energies" in f:
        if output_file != args.input_file:
            showerdata.save(cond_data, output_file, overwrite=args.overwrite)
        showerdata.observables.save_observables_to_file(
            output_file,
            {
                "num_points_per_layer": results_points.numpy(),
                "min_time_per_layer": results_time.numpy(),
                "energy_per_layer": results_energy.numpy(),
            },
            overwrite=args.overwrite,
        )
    else:
        dict_to_save = {
            "num_points_per_layer": results_points.numpy(),
            "min_time_per_layer": results_time.numpy(),
            "energy_per_layer": results_energy.numpy(),
            "energies": cond_data["energies"][:num_showers],
            "labels": cond_data["labels"][:num_showers],
            "directions": cond_data["directions"][:num_showers],
        }
        with h5py.File(output_file, "a" if not args.overwrite else "w") as f_out:
            if "num_points_per_layer" in f_out and not args.overwrite:
                raise ValueError(
                    "Dataset 'num_points_per_layer' already exists in the output file. Use --overwrite to overwrite it."
                )
            for key, values in dict_to_save.items():
                f_out.create_dataset(
                    key,
                    data=values,
                )
    print(f"Results saved in {(time.time() - start) * 1000.0:.1f}ms")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
