import copy
from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["CNF"]


# partially based on https://gist.github.com/francois-rozet/fd6a820e052157f8ac6e2aa39e16c1aa
class CNF(nn.Module):
    def __init__(
        self,
        network: Callable[[Tensor, Tensor, Tensor], Tensor],
        frequencies: int = 3,
        acceleration: float = 0.0,
    ) -> None:
        super().__init__()
        self.network = network
        self.acceleration = acceleration
        self.frequencies = nn.Buffer(
            torch.arange(1, frequencies + 1).reshape(1, -1) * torch.pi
        )
        self.multi_loss = MultiTaskLoss(
            time_exists=network.dim_input > 78, energy_exists=network.dim_input > 156
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
    def __init__(self, time_exists: bool = False, energy_exists: bool = False) -> None:
        super().__init__()
        # log(sigma^2) for each task - initialized to 0 (sigma=1)
        self.log_var_points = nn.Parameter(torch.zeros(1))
        self.log_var_time = nn.Parameter(torch.zeros(1)) if time_exists else None
        self.log_var_energy = nn.Parameter(torch.zeros(1)) if energy_exists else None

    def forward(self, residual: Tensor, num_layers: int) -> Tensor:
        # precision = 1 / sigma^2 = exp(-log_var)
        loss_points = residual[:, :num_layers].mean()
        loss_time = (
            residual[:, num_layers : 2 * num_layers].mean()
            if self.log_var_time is not None
            else 0.0
        )
        loss_energy = (
            residual[:, 2 * num_layers :].mean()
            if self.log_var_energy is not None
            else 0.0
        )

        precision_points = torch.exp(-self.log_var_points)
        precision_time = (
            torch.exp(-self.log_var_time) if self.log_var_time is not None else 0.0
        )
        precision_energy = (
            torch.exp(-self.log_var_energy) if self.log_var_energy is not None else 0.0
        )

        loss = precision_points * loss_points + self.log_var_points

        if self.log_var_time is not None:
            loss = loss + precision_time * loss_time + self.log_var_time

        if self.log_var_energy is not None:
            loss = loss + precision_energy * loss_energy + self.log_var_energy

        return loss


class Distilled(nn.Module):
    def __init__(self, cnf: CNF) -> None:
        super().__init__()
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
