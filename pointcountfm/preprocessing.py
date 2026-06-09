import math

import torch
from torch import nn

__all__ = [
    "Transformation",
    "Sequence",
    "Identity",
    "Log",
    "LogIt",
    "Affine",
    "Clamp",
    "StandardScaler",
    "QuantileTransformer",
    "Dequantize",
    "compose",
]


class Transformation(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class Sequence(Transformation):
    def __init__(self, modules: list[Transformation]) -> None:
        super().__init__()
        self.sub_modules = nn.ModuleList(modules)

    def fit(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for module in self.sub_modules:
            if isinstance(module, Transformation):
                x = module.fit(x, mask)
            else:
                raise ValueError("All sub-modules must be of type Transformation")
        return x

    def forward(self, x: torch.Tensor):
        for module in self.sub_modules:
            x = module.forward(x)
        return x

    def inverse(self, x: torch.Tensor):
        for module in self.sub_modules[::-1]:
            if isinstance(module, Transformation):
                x = module.inverse(x)
            else:
                raise ValueError("All sub-modules must be of type Transformation")
        return x


class Identity(Transformation):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Log(Transformation):
    def __init__(self, alpha: float = 1e-6, base: float = math.e) -> None:
        super().__init__()
        self.alpha = alpha
        self.log_base = math.log(base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x + self.alpha) / self.log_base

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_base * x) - self.alpha


class LogIt(Transformation):
    def __init__(self, alpha: float = 1e-6) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (1 - 2 * self.alpha) * x + self.alpha
        return torch.log(x / (1 - x))

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.sigmoid(x)
        return (x - self.alpha) / (1 - 2 * self.alpha)


class Affine(Transformation):
    def __init__(self, scale: float = 1.0, shift: float = 0.0) -> None:
        super().__init__()
        self.a = scale
        self.b = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a * x + self.b

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b) / self.a


class Clamp(Transformation):
    def __init__(self, min: float = 0.0, max: float = 1.0) -> None:
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, self.min, self.max)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x


class StandardScaler(Transformation):
    def __init__(self, shape: tuple[int]) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("std", torch.ones(shape))
        self.shape = shape

    def fit(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.bool)
        dims = tuple(torch.where(torch.tensor(self.shape) == 1)[0].tolist())
        mean = torch.sum(x * mask, dim=dims, keepdim=True)
        mean /= torch.sum(mask, dim=dims, keepdim=True)
        self.mean = mean
        x = x - mean
        std = torch.sqrt(torch.sum(x**2 * mask, dim=dims, keepdim=True))
        std /= torch.sqrt(torch.sum(mask, dim=dims, keepdim=True) - 1)
        std[std == 0] = 1
        self.std = std
        x = x / std
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


class QuantileTransformer(Transformation):
    """
    Per-feature quantile normalisation: maps the empirical distribution of each
    feature to N(0, 1) via the empirical CDF followed by the probit function.

    Handles any distribution — including the multi-modal, heavily-tailed
    min-time-per-layer distribution — without manual tuning of alpha or scale.
    Zeros (inactive layers) map to the lowest quantile and back correctly.

    Parameters
    ----------
    n_quantiles : int
        Number of quantile boundaries to store per feature. 1000 is a good
        default; increase for smoother interpolation on large datasets.
    shape : list[int]
        Shape convention identical to StandardScaler: dimensions equal to 1
        are treated as batch axes. E.g. [1, 78] for 78-feature per-sample data.
    """

    def __init__(
        self,
        n_quantiles: int = 1000,
        shape: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.n_quantiles = n_quantiles
        shape = shape if shape is not None else [1, 78]
        n_features = 1
        for s in shape:
            if s != 1:
                n_features *= s
        self.n_features = n_features
        # quantiles[q, f] = value at level q/(n_quantiles-1) for feature f
        self.register_buffer("quantiles", torch.zeros(n_quantiles, n_features))

    @torch.no_grad()
    def fit(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x2d = x.reshape(-1, self.n_features)
        levels = torch.linspace(0, 1, self.n_quantiles, device=x.device, dtype=x.dtype)
        q = torch.zeros(
            self.n_quantiles, self.n_features, device=x.device, dtype=x.dtype
        )
        for f in range(self.n_features):
            vals = x2d[:, f]
            if mask is not None:
                vals = vals[mask.reshape(-1, self.n_features)[:, f].bool()]
            vals = vals[~torch.isnan(vals)]
            vals, _ = torch.sort(vals)
            if vals.numel() == 0:
                continue
            # linear interpolation to build the quantile table
            idx = levels * (vals.numel() - 1)
            idx_lo = idx.long().clamp(0, vals.numel() - 2)
            idx_hi = idx_lo + 1
            frac = idx - idx_lo.float()
            q[:, f] = vals[idx_lo] * (1 - frac) + vals[idx_hi] * frac
        self.quantiles = q
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2d = x.reshape(-1, self.n_features)
        # searchsorted on (features, n_quantiles) vs (features, N) → (features, N)
        idx = torch.searchsorted(self.quantiles.T.contiguous(), x2d.T.contiguous())
        u = (idx.float() / (self.n_quantiles - 1)).clamp(1e-6, 1 - 1e-6).T
        # probit: uniform [0,1] → N(0,1)
        return (torch.erfinv(2 * u - 1) * math.sqrt(2)).reshape(shape)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2d = x.reshape(-1, self.n_features)
        # normal CDF: N(0,1) → [0,1]
        u = (0.5 * (1 + torch.erf(x2d / math.sqrt(2)))).clamp(0, 1)
        # interpolate in the quantile table
        idx = (u * (self.n_quantiles - 1)).clamp(0, self.n_quantiles - 1)
        idx_lo = idx.long().clamp(0, self.n_quantiles - 2)
        idx_hi = (idx_lo + 1).clamp(0, self.n_quantiles - 1)
        frac = (idx - idx_lo.float()).clamp(0, 1)
        q_T = self.quantiles.T.contiguous()  # (features, n_quantiles)
        lo = torch.gather(q_T, 1, idx_lo.T).T  # (N, features)
        hi = torch.gather(q_T, 1, idx_hi.T).T
        return (lo * (1 - frac) + hi * frac).reshape(shape)


class Dequantize(Transformation):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() + torch.rand(x.shape, dtype=torch.float32, device=x.device)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return torch.floor(x)


def compose(transformation: list[list[str | dict | list | None]] | None) -> Sequence:
    if transformation is None:
        return Sequence([Identity()])
    trafo_list = []
    attrs = globals()
    for element in transformation:
        if (element[0] not in __all__) or (
            element[0] in ["Transformation", "Sequence", "compose"]
        ):
            raise ValueError(f"Invalid transformation: {element[0]}")
        Trafo = attrs[element[0]]
        assert issubclass(Trafo, Transformation)
        if len(element) == 1 or element[1] is None:
            trafo_list.append(Trafo())
        elif isinstance(element[1], list):
            trafo_list.append(Trafo(*element[1]))
        elif isinstance(element[1], dict):
            trafo_list.append(Trafo(**element[1]))
        else:
            raise ValueError(
                f"argument for {element[0]} must be a list or a dict not {type(element[1])}"
            )

    return Sequence(trafo_list)
