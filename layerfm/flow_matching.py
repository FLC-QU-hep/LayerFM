import copy
from collections.abc import Callable

import torch
from torch import Tensor, nn

from layerfm import models

__all__ = ["CNF", "build_flow"]


def build_flow(model_config: dict) -> "CNF":
    """Network + conditional flow matching model from the `model:` config section.

    The optional `model.flow` keys are passed to CNF; `model.flow.type`, if present,
    must be 'cnf' (the only objective in LayerFM)."""
    config = copy.deepcopy(model_config)
    flow_config = config.pop("flow", None) or {}
    flow_type = flow_config.pop("type", "cnf")
    if flow_type != "cnf":
        raise ValueError(
            f"Unknown model.flow.type {flow_type!r}; only 'cnf' is supported."
        )
    network_class = getattr(models, config.pop("name"))
    return CNF(network_class(**config), **flow_config)


# partially based on https://gist.github.com/francois-rozet/fd6a820e052157f8ac6e2aa39e16c1aa
class CNF(nn.Module):
    def __init__(
        self,
        network: Callable[[Tensor, Tensor, Tensor], Tensor],
        frequencies: int = 3,
        acceleration: float = 0.0,
        time_weight: float = 1.0,
        energy_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.network = network
        self.acceleration = acceleration
        self.frequencies = nn.Buffer(
            torch.arange(1, frequencies + 1).reshape(1, -1) * torch.pi
        )
        n = getattr(network, "n_layers", 78)
        self.multi_loss = MultiTaskLoss(
            time_exists=network.dim_input > n,
            energy_exists=network.dim_input > 2 * n,
            time_weight=time_weight,
            energy_weight=energy_weight,
        )

    def forward(self, t: Tensor, x: Tensor, condition: Tensor) -> Tensor:
        t = self.time_trafo(t)
        t = self.frequencies * t.reshape(-1, 1)
        t = torch.cat((t.cos(), t.sin()), dim=-1)
        t = t.expand(x.shape[0], -1)

        return self.network(t, x, condition)

    def encode(self, x: Tensor, steps: int, condition: Tensor) -> Tensor:
        return self.heun_integrate(
            x0=x,
            t0=torch.zeros((), dtype=x.dtype, device=x.device),
            t1=torch.ones((), dtype=x.dtype, device=x.device),
            steps=steps,
            condition=condition,
        )

    def decode(self, z: Tensor, steps: int, condition: Tensor) -> Tensor:
        return self.heun_integrate(
            x0=z,
            t0=torch.ones((), dtype=z.dtype, device=z.device),
            t1=torch.zeros((), dtype=z.dtype, device=z.device),
            steps=steps,
            condition=condition,
        )

    def time_trafo(self, t: Tensor) -> Tensor:
        return (1 - self.acceleration) * t + self.acceleration * t**2

    def time_derivative(self, t: Tensor) -> Tensor:
        return 2 * self.acceleration * t - self.acceleration + 1

    def loss(
        self,
        x: Tensor,
        condition: Tensor,
        noise: Tensor | None = None,
        num_layers: int = 78,
    ) -> Tensor:
        # replace nan with 0 and inf with 1e4 to avoid hallucinations
        x = torch.where(x.isnan(), torch.zeros_like(x), x)

        t = torch.rand(
            [x.shape[0]] + [1] * (x.dim() - 1), device=x.device, dtype=x.dtype
        )
        t_ = self.time_trafo(t)
        z = noise if noise is not None else torch.randn_like(x)
        y = (1 - t_) * x + (1e-4 + (1 - 1e-4) * t_) * z
        u = (1 - 1e-4) * z - x
        u = self.time_derivative(t) * u

        pred = self(t.reshape(-1, 1), y, condition)
        residual = (pred - u) ** 2
        loss = self.multi_loss(residual, num_layers)
        return loss

    def sample(self, shape: tuple[int, int], steps: int, condition: Tensor) -> Tensor:
        z = torch.randn(
            *shape, device=self.frequencies.device, dtype=self.frequencies.dtype
        )
        return self.decode(z, steps, condition)

    def sample_return_z(
        self, shape: tuple[int, int], steps: int, condition: Tensor
    ) -> tuple[Tensor, Tensor]:
        z = torch.randn(
            *shape, device=self.frequencies.device, dtype=self.frequencies.dtype
        )
        return self.decode(z, steps, condition), z

    def heun_integrate(
        self,
        x0: Tensor,
        t0: Tensor,
        t1: Tensor,
        steps: int,
        condition: Tensor,
    ) -> Tensor:
        x = x0
        t = t0
        dt = (t1 - t0) / steps
        for _ in range(steps):
            df = self(t, x, condition)
            y_ = x + dt * df
            x = x + dt / 2 * (df + self(t + dt, y_, condition))
            t = t + dt

        return x

    def __repr__(self) -> str:
        network = self.network.__repr__().replace("\n", "\n  ")
        return f"{self.__class__.__name__}(\n  (network): {network}\n  frequencies/pi={(self.frequencies[0] / torch.pi).tolist()},\n  acceleration={self.acceleration}\n)"


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        time_exists: bool = False,
        energy_exists: bool = False,
        time_weight: float = 1.0,
        energy_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.time_exists = time_exists
        self.energy_exists = energy_exists
        self.time_weight = time_weight
        self.energy_weight = energy_weight

    def forward(self, residual: Tensor, num_layers: int) -> Tensor:
        w = self.sigmas()

        loss = w["points"] * residual[:, :num_layers].mean()
        if self.time_exists:
            loss = loss + w["time"] * residual[:, num_layers : 2 * num_layers].mean()
        if self.energy_exists:
            loss = loss + w["energy"] * residual[:, 2 * num_layers :].mean()
        return loss

    def sigmas(self) -> dict[str, float]:
        sigmas = {"points": 1.0}
        if self.time_exists:
            sigmas["time"] = self.time_weight
        if self.energy_exists:
            sigmas["energy"] = self.energy_weight
        return sigmas


class Distilled(nn.Module):
    def __init__(self, cnf: CNF) -> None:
        super().__init__()
        if not isinstance(cnf, CNF):
            raise NotImplementedError("Distillation is only implemented for the CNF.")
        self.network = copy.deepcopy(cnf.network)
        if cnf.acceleration != 0:
            raise NotImplementedError(
                "Distilled CNF does not support acceleration yet."
            )
        self.frequencies = nn.Buffer(cnf.frequencies.clone())

        t = torch.ones(
            (1, 1), device=cnf.frequencies.device, dtype=cnf.frequencies.dtype
        )
        t = self.frequencies * t
        t = torch.cat((t.cos(), t.sin()), dim=-1)
        self.t = nn.Buffer(t)

    def forward(self, z: Tensor, condition: Tensor) -> Tensor:
        t = self.t.repeat(z.shape[0], 1)
        return z - self.network(t, z, condition)

    def sample(self, shape: tuple[int, int], condition: Tensor) -> Tensor:
        z = torch.randn(
            *shape, device=self.frequencies.device, dtype=self.frequencies.dtype
        )
        return self(z, condition)

    def sample_return_z(
        self, shape: tuple[int, int], steps: int, condition: Tensor
    ) -> tuple[Tensor, Tensor]:
        z = torch.randn(
            *shape, device=self.frequencies.device, dtype=self.frequencies.dtype
        )
        return self(z, condition), z
