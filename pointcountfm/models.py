import torch
from torch import nn

__all__ = [
    "FullyConnected",
    "ConcatSquash",
]


class FullyConnected(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_condition: int = 0,
        dim_time: int = 0,
        hidden_dims: list[int] | None = None,
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 64]

        self.points_embedding = nn.Sequential(
            nn.Linear(78, 128),
            activation(),
            nn.Linear(128, 128),
            activation(),
        )
        dim_input = 128
        if dim_input > 78:
            self.time_embedding = nn.Sequential(
                nn.Linear(78, 128),
                activation(),
                nn.Linear(128, 128),
                activation(),
            )
            dim_input += 128
            if dim_input > 156:
                self.energy_embedding = nn.Sequential(
                    nn.Linear(78, 128),
                    activation(),
                    nn.Linear(128, 128),
                    activation(),
                )
                dim_input += 128

        self.dim_input = dim_input
        layers = []
        prev_dim = dim_input + dim_condition + dim_time
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.LayerNorm(dim))  # only change from the original code
            layers.append(activation())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, dim_input))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_emb = self.points_embedding(x[:, :78])
        if x.shape[1] > 78:
            time_emb = self.time_embedding(x[:, 78:156])
            x_emb = torch.cat([x_emb, time_emb], dim=-1)
            if x.shape[1] > 156:
                energy_emb = self.energy_embedding(x[:, 156:])
                x_emb = torch.cat([x_emb, energy_emb], dim=-1)

        input = [t, x]
        if condition is not None:
            input.append(condition)
        input = torch.cat(input, dim=-1)
        return self.network(input)


class ConcatSquash(nn.Module):
    class __ConcatSquashLayer(nn.Module):
        def __init__(
            self,
            dim_input: int,
            dim_output: int,
            dim_cond: int,
            activation: type[nn.Module],
        ) -> None:
            super().__init__()
            self.cond_embed = nn.Sequential(
                nn.Linear(dim_cond, dim_input),
            )
            self.network = nn.Sequential(
                nn.Linear(dim_input, dim_output),
                activation(),
            )

        def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
            cond_embed = self.cond_embed(condition)
            return self.network(x + cond_embed)

    def __init__(
        self,
        dim_input: int,
        dim_condition: int = 0,
        dim_time: int = 0,
        hidden_dims: list[int] | None = None,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]
        layers = []
        prev_dim = dim_input
        dim_cond_time = dim_condition + dim_time
        for dim in hidden_dims:
            layers.append(
                self.__ConcatSquashLayer(prev_dim, dim, dim_cond_time, activation)
            )
            prev_dim = dim
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(prev_dim, dim_input)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        condition = t if condition is None else torch.cat([t, condition], dim=-1)
        for layer in self.layers:
            x = layer(x, condition)
        return self.output(x)
