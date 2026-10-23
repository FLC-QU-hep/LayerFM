import argparse
import os
import time
from collections.abc import Iterable

import h5py
import showerdata
import torch
import torch.nn as nn
import yaml


def parse_args(args: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Inference on a model")
    parser.add_argument(
        "model", type=str, help="Model file (compiled.pt or checkpoint.pt)"
    )
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
    parser.add_argument(
        "--num-layers",
        type=int,
        default=78,
        help="Number of calorimeter layers (= features per output channel). Must match the trained model.",
    )
    parser.add_argument(
        "--from-checkpoint",
        action="store_true",
        help=(
            "Load from a checkpoint.pt (raw state dict) instead of a compiled TorchScript model. "
            "Requires conf.yaml in the same directory as the checkpoint, or use --config."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config yaml, used with --from-checkpoint and to read the "
        "conditions of a non-showerdata input file. "
        "Defaults to conf.yaml in the model directory.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of ODE integration steps. Only used with --from-checkpoint.",
    )
    parser.add_argument(
        "--distill",
        action="store_true",
        help=(
            "Load a distilled one-step student (checkpoint_distill.pt) instead of the "
            "teacher. Only used with --from-checkpoint; --steps is ignored."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for generation. Only used with --from-checkpoint.",
    )
    return parser.parse_args(args)


def to_labels(pdg_codes: torch.Tensor, pdgs: Iterable[int]) -> torch.Tensor:
    labels = torch.full(pdg_codes.shape, -1, dtype=torch.int64)
    for label, pdg in enumerate(pdgs):
        labels[pdg_codes == pdg] = label
    return labels


class _CheckpointSampler(nn.Module):
    """Python sampler built from a checkpoint + config; mirrors the Sampler in trainer.py."""

    def __init__(
        self,
        flow: nn.Module,
        transform_inc: nn.Module,
        transform_num_points: nn.Module,
        transform_min_time: nn.Module,
        transform_en_perl: nn.Module,
        num_layers: int,
        dim_data: int,
        steps: int,
        use_directions: bool = True,
        use_labels: bool = True,
    ) -> None:
        super().__init__()
        self.flow = flow
        self.transform_inc = transform_inc
        self.transform_num_points = transform_num_points
        self.transform_min_time = transform_min_time
        self.transform_en_perl = transform_en_perl
        self.num_layers = num_layers
        self.dim_data = dim_data
        self.steps = steps
        self.use_time = dim_data > num_layers
        self.use_enperl = dim_data > 2 * num_layers
        self.use_directions = use_directions
        self.use_labels = use_labels

    @torch.no_grad()
    def forward(self, condition: torch.Tensor) -> list[torch.Tensor]:
        cond = condition.clone().float()
        cond[:, :1] = self.transform_inc(cond[:, :1])
        samples = self.flow.sample(
            (cond.shape[0], self.dim_data), self.steps, condition=cond
        )
        n = self.num_layers
        result = [self.transform_num_points.inverse(samples[:, :n])]
        if self.use_time:
            result.append(self.transform_min_time.inverse(samples[:, n : n * 2]))
        if self.use_enperl:
            result.append(self.transform_en_perl.inverse(samples[:, n * 2 : n * 3]))
        return result


class _DistilledFlow(nn.Module):
    """Gives Distilled the flow.sample(shape, steps, condition=...) signature.

    The student is a single noise -> data map, so `steps` is accepted and ignored.
    """

    def __init__(self, student: nn.Module) -> None:
        super().__init__()
        self.student = student

    def sample(
        self, shape: tuple[int, int], steps: int, condition: torch.Tensor
    ) -> torch.Tensor:
        return self.student.sample(shape, condition)


def load_from_checkpoint(
    model_path: str,
    config_path: str | None = None,
    steps: int = 50,
    distill: bool = False,
) -> _CheckpointSampler:
    """Rebuild a _CheckpointSampler from checkpoint.pt + conf.yaml."""
    from layerfm import flow_matching
    from layerfm.data_loader import get_loaders

    config_dir = os.path.dirname(os.path.abspath(model_path))
    if config_path is None:
        config_path = os.path.join(config_dir, "conf.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found at {config_path}. "
            "Pass --config to specify the path explicitly."
        )

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"].copy()

    # Build the CNF from config
    flow = flow_matching.build_flow(cfg["model"])

    # Load weights
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    if distill:
        # Distilled() copies the teacher's network/buffers, then the student's own
        # state dict overwrites them, so the teacher weights here are irrelevant.
        student = flow_matching.Distilled(flow)
        student.load_state_dict(ckpt["model"])
        student.eval()
        flow = _DistilledFlow(student)
    else:
        flow.load_state_dict(ckpt["model"])
        flow.eval()
    flow = flow.to(torch.float32)

    # The input transforms are fitted on the training data (same split as training).
    # transforms.pt next to the checkpoint caches them, so sampling does not need the data.
    transforms_path = os.path.join(config_dir, "transforms.pt")
    if os.path.exists(transforms_path):
        transforms = torch.load(transforms_path, map_location="cpu", weights_only=False)
    else:
        print("Fitting transforms on training data...")
        data_cfg["device"] = "cpu"
        train_loader, _ = get_loaders(**data_cfg)
        transforms = {
            name: getattr(train_loader, name)
            for name in (
                "transform_inc",
                "transform_num_points",
                "transform_min_time",
                "transform_en_perl",
            )
        }
        transforms["dim_data"] = train_loader.data.shape[1]
        torch.save(transforms, transforms_path)
        print(f"saved the fitted transforms to {transforms_path}")

    num_layers = data_cfg.get("num_layers", 78)
    dim_data = transforms.pop("dim_data")

    return _CheckpointSampler(
        flow=flow,
        **transforms,
        num_layers=num_layers,
        dim_data=dim_data,
        steps=steps,
        use_directions=data_cfg.get("use_directions", True),
        use_labels=data_cfg.get("use_labels", True),
    )


def _run_compiled_inference(
    model_jit: torch.jit.ScriptModule,
    conditions: torch.Tensor,
    n_layers: int,
    pdg_codes: list[int],
) -> torch.Tensor:
    """Run inference using a compiled TorchScript model."""

    class InferenceModel(torch.nn.Module):
        def __init__(self, model: torch.jit.ScriptModule, n_layers: int):
            super().__init__()
            self.model = model
            self.n_layers: int = n_layers

        def _check_hallucinations(
            self, result: torch.Tensor, condition: torch.Tensor
        ) -> torch.Tensor:
            while (
                result[:, : self.n_layers].max() > 1e4
                or result[:, : self.n_layers].min() < -1e4
            ):
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
                if result[-1].shape[1] != self.n_layers:
                    result = result[:-1]
                for j in range(len(result)):
                    list_.append(result[j])
                results.append(torch.cat(list_, dim=-1))
                results[-1] = self._check_hallucinations(results[-1], conditions[[i]])
            return torch.cat(results, dim=0)

    print("Compiling inference function...")
    start = time.time()
    inference = InferenceModel(model_jit, n_layers=n_layers)
    inference = inference.to(torch.float32)
    inference = torch.jit.script(inference)
    print(f"Inference function compiled in {(time.time() - start) * 1000.0:.1f}ms")

    print("Warming up...")
    start = time.time()
    example_conditions = torch.concatenate(
        (
            torch.tensor([[50.0]]),
            torch.nn.functional.one_hot(
                torch.tensor([0]), num_classes=len(pdg_codes)
            ).to(torch.float32),
            torch.tensor([[0.0, 1.0, 0.0]]),
        ),
        dim=1,
    )
    inference(example_conditions)
    inference(example_conditions)
    print(f"Warmup done in {(time.time() - start) * 1000.0:.1f}ms")

    print("Running inference...")
    start = time.time()
    results = inference(conditions)
    print(f"Inference done in {(time.time() - start):.2f}s")
    return results


def _run_checkpoint_inference(
    sampler: _CheckpointSampler,
    conditions: torch.Tensor,
    n_layers: int,
    batch_size: int = 1024,
) -> torch.Tensor:
    """Run inference using a Python _CheckpointSampler."""
    print("Running inference...")
    start = time.time()
    results = []
    n = conditions.size(0)
    for batch_start in range(0, n, batch_size):
        batch_cond = conditions[batch_start : batch_start + batch_size]
        while True:
            result_list = sampler(batch_cond)
            result = torch.cat(result_list, dim=-1)
            bad = (result[:, :n_layers] > 1e4) | (result[:, :n_layers] < -1e4)
            if not bad.any():
                break
            # re-sample only the bad rows
            bad_idx = bad.any(dim=1).nonzero(as_tuple=True)[0]
            print(f"re-sampling {bad_idx.numel()} hallucinated rows...")
            while bad_idx.numel() > 0:
                fix_list = sampler(batch_cond[bad_idx])
                fix = torch.cat(fix_list, dim=-1)
                result[bad_idx] = fix
                still_bad = (fix[:, :n_layers] > 1e4) | (fix[:, :n_layers] < -1e4)
                bad_idx = bad_idx[still_bad.any(dim=1)]
        results.append(result)
        done = min(batch_start + batch_size, n)
        print(f"  {done}/{n} samples generated")
    results_tensor = torch.cat(results, dim=0)
    elapsed = time.time() - start
    print(f"Inference done in {elapsed:.2f}s")
    print(f"Average inference time: {elapsed / n * 1000.0:.1f}ms per sample")
    return results_tensor


def main():
    args = parse_args()
    torch.set_num_threads(1)

    print("Loading model...")
    start = time.time()

    if args.from_checkpoint:
        sampler = load_from_checkpoint(
            args.model, args.config, steps=args.steps, distill=args.distill
        )
        print(f"Model loaded in {(time.time() - start) * 1000.0:.1f}ms")
        use_compiled = False
    else:
        model_jit = torch.jit.load(args.model)
        print(model_jit)
        model_jit.eval()
        print(f"Model loaded in {(time.time() - start) * 1000.0:.1f}ms")
        use_compiled = True

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
        # the run's own config, as in load_from_checkpoint
        config_path = args.config or os.path.join(
            os.path.dirname(os.path.abspath(args.model)), "conf.yaml"
        )
        with open(config_path) as file:
            config = yaml.safe_load(file)
        num_showers = f[list(f.keys())[0]].shape[0]
        cond_data = h5py.File(config["data"]["data_file"], "r")
        use_labels = config["data"].get("use_labels", True)
        use_directions = config["data"].get("use_directions", True)
        cond_parts = [
            torch.from_numpy(cond_data["energies"][:num_showers]).to(torch.float32)
        ]
        if use_labels and "labels" in cond_data:
            # one-hot, as in data_loader.load_data_file
            num_classes = (
                config["data"].get("num_classes")
                or int(cond_data["labels"][:].max()) + 1
            )
            cond_parts.append(
                torch.nn.functional.one_hot(
                    torch.from_numpy(cond_data["labels"][:num_showers]).long(),
                    num_classes=num_classes,
                ).to(torch.float32)
            )
        if use_directions and "directions" in cond_data:
            cond_parts.append(
                torch.from_numpy(cond_data["directions"][:num_showers]).to(
                    torch.float32
                )
            )
        conditions = torch.concatenate(cond_parts, dim=-1)
    print(f"Conditioning data loaded in {(time.time() - start) * 1000.0:.1f}ms")

    if use_compiled:
        results = _run_compiled_inference(
            model_jit, conditions, args.num_layers, args.pdg_codes
        )
    else:
        results = _run_checkpoint_inference(
            sampler, conditions, args.num_layers, args.batch_size
        )

    print("Saving results...")
    print(results.shape)
    layers = args.num_layers
    results = torch.clamp(results, min=0.0)
    results_points = (results[:, :layers] + 0.5).to(torch.int32)
    if results.shape[1] > layers:
        results_time = results[:, layers : int(layers * 2)].to(torch.float32)
        results_time = torch.clamp(results_time, max=200.0)
        results_time[results_points == 0] = 0.0
    if results.shape[1] > (layers * 2):
        results_energy = results[:, int(layers * 2) :].to(torch.float32)
    start = time.time()
    output_file = args.output if args.output else args.input_file

    if "energies" in f:
        if output_file != args.input_file:
            showerdata.save(cond_data, output_file, overwrite=args.overwrite)
        showerdata.observables.save_observables_to_file(
            output_file,
            {
                "num_points_per_layer": results_points.numpy(),
                "min_time_per_layer": results_time.numpy()
                if results.shape[1] > layers
                else None,
                "energy_per_layer": results_energy.numpy()
                if results.shape[1] > layers * 2
                else None,
            },
            overwrite=args.overwrite,
        )
    else:
        dict_to_save = {
            "num_points_per_layer": results_points.numpy(),
            "min_time_per_layer": results_time.numpy()
            if results.shape[1] > layers
            else None,
            "energy_per_layer": results_energy.numpy()
            if results.shape[1] > layers * 2
            else None,
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
                if values is not None:
                    f_out.create_dataset(
                        key,
                        data=values,
                    )
    print(f"Results saved in {(time.time() - start) * 1000.0:.1f}ms")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
