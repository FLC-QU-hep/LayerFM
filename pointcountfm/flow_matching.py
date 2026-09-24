import copy
from collections.abc import Callable

import torch
from torch import Tensor, nn

from pointcountfm import models

__all__ = ["CNF", "MeanFlow", "build_flow"]


def build_flow(model_config: dict) -> "CNF | MeanFlow":
    """Network + generative model from the `model:` config section.

    `model.flow.type` picks the objective: 'cnf' (default, conditional flow matching,
    sampled by Heun integration) or 'mean_flow' (MeanFlow, can sample in one step).
    The remaining `model.flow` keys are passed to that class."""
    config = copy.deepcopy(model_config)
    flow_config = config.pop("flow", None) or {}
    flow_type = flow_config.pop("type", "cnf")
    network_class = getattr(models, config.pop("name"))
    if flow_type == "cnf":
        return CNF(network_class(**config), **flow_config)
    if flow_type == "mean_flow":
        # the network sees two time embeddings, t and t - r, each dim_time wide
        config["dim_time"] = 2 * config.get("dim_time", 0)
        return MeanFlow(network_class(**config), **flow_config)
    raise ValueError(
        f"Unknown model.flow.type {flow_type!r}; choices: 'cnf', 'mean_flow'."
    )


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


class MeanFlow(nn.Module):
    """MeanFlow (Geng et al., arXiv:2505.13447).

    Learns the average velocity u(z, r, t) = 1/(t - r) * int_r^t v(z_s, s) ds along the
    straight path z_t = (1 - t) x + t e (data x at t=0, noise e at t=1), so one network
    evaluation jumps from t to r: z_r = z_t - (t - r) u(z_t, r, t). A single step from
    t=1 to r=0 is a full sample.

    Trained with the MeanFlow identity u(z_t, r, t) = v - (t - r) du/dt, v = e - x, where
    du/dt = v . d_z u + d_t u is the total derivative along the path. It is computed
    exactly with forward-mode AD (torch.func.jvp, tangent (v, 0, 1)), and the target is
    stop-gradiented.

    ratio_r_neq_t : fraction of the batch trained on an interval r < t; the rest has
                    r = t, where the identity reduces to plain flow matching. The paper
                    (Table 1a) finds 0 breaks one-step sampling and 0.25-1 all work.
    adaptive_p    : per-sample loss weight sg(1 / (err^2 + adaptive_c)^p); 0 is plain
                    MSE, the paper's best is 1 (Table 1e).
    adaptive_per_channel : compute that weight once per n_layers-wide channel
                    (points | min_time | energy) instead of once over all of them.
                    The single scalar is set by whichever channel has the larger
                    residual -- for points+time that is min_time for most samples --
                    so a shower with hard timing also loses the gradient on its
                    point counts.

    t and r are the max and min of two uniform draws. The paper's logit-normal draw
    leaves the (r=0, t=1) corner that one-step sampling evaluates almost unsampled.
    """

    def __init__(
        self,
        network: Callable[[Tensor, Tensor, Tensor], Tensor],
        frequencies: int = 3,
        ratio_r_neq_t: float = 0.5,
        adaptive_p: float = 1.0,
        adaptive_c: float = 1e-3,
        adaptive_per_channel: bool = False,
        time_weight: float = 1.0,
        energy_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.network = network
        self.ratio_r_neq_t = ratio_r_neq_t
        self.adaptive_p = adaptive_p
        self.adaptive_c = adaptive_c
        self.adaptive_per_channel = adaptive_per_channel
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

    def forward(self, z: Tensor, r: Tensor, t: Tensor, condition: Tensor) -> Tensor:
        # conditioned on (t, t - r), the paper's best time embedding (Table 1c)
        emb = []
        for s in (t, t - r):
            s = self.frequencies * s.reshape(-1, 1)
            emb += [s.cos(), s.sin()]
        emb = torch.cat(emb, dim=-1).expand(z.shape[0], -1)
        return self.network(emb, z, condition)

    def decode(self, z: Tensor, steps: int, condition: Tensor) -> Tensor:
        for i in range(steps):
            t = torch.full((1, 1), 1 - i / steps, dtype=z.dtype, device=z.device)
            r = torch.full((1, 1), 1 - (i + 1) / steps, dtype=z.dtype, device=z.device)
            z = z - (t - r) * self(z, r, t, condition)
        return z

    def sample_times(self, x: Tensor) -> tuple[Tensor, Tensor]:
        shape = (x.shape[0], 1)
        a = torch.rand(shape, device=x.device, dtype=x.dtype)
        b = torch.rand(shape, device=x.device, dtype=x.dtype)
        t, r = torch.maximum(a, b), torch.minimum(a, b)
        keep_interval = torch.rand(shape, device=x.device) < self.ratio_r_neq_t
        return t, torch.where(keep_interval, r, t)

    def _adaptive_weight(self, residual: Tensor, num_layers: int) -> Tensor:
        """Per-sample loss weight sg(1 / (err^2 + c)^p), broadcast over the dims it
        was computed from (see adaptive_per_channel in the class docstring)."""
        if not self.adaptive_per_channel:
            return (residual.mean(dim=1, keepdim=True) + self.adaptive_c) ** (
                -self.adaptive_p
            )
        return torch.cat(
            [
                (
                    (c.mean(dim=1, keepdim=True) + self.adaptive_c) ** -self.adaptive_p
                ).expand_as(c)
                for c in residual.split(num_layers, dim=1)
            ],
            dim=1,
        )

    def loss(
        self,
        x: Tensor,
        condition: Tensor,
        noise: Tensor | None = None,
        num_layers: int = 78,
    ) -> Tensor:
        # replace nan with 0 to avoid hallucinations
        x = torch.where(x.isnan(), torch.zeros_like(x), x)

        e = noise if noise is not None else torch.randn_like(x)
        t, r = self.sample_times(x)
        z = (1 - t) * x + t * e
        v = e - x

        u, dudt = torch.func.jvp(
            lambda z, r, t: self(z, r, t, condition),
            (z, r, t),
            (v, torch.zeros_like(r), torch.ones_like(t)),
        )
        target = (v - (t - r) * dudt).detach()
        residual = (u - target) ** 2

        weight = self._adaptive_weight(residual, num_layers)
        weighted = self.multi_loss(weight.detach() * residual, num_layers)
        unweighted = self.multi_loss(residual, num_layers)
        # value of the unweighted loss, gradient of the weighted one: at adaptive_p=1
        # the weighted loss is err^2 / (err^2 + c), pinned at ~1, so it would make
        # the logged train/val curves useless
        return unweighted.detach() + weighted - weighted.detach()

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
            raise NotImplementedError(
                "Distillation is only implemented for the CNF (model.flow.type: cnf)."
            )
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
