import torch
from torch import nn

__all__ = [
    "FullyConnected",
    "ConcatSquash",
]


class _Block(nn.Module):
    """
    Pre-norm MLP block with optional context (time + condition) injection.
    Uses a residual connection when input and output dims are equal.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_dim: int,
        activation: type[nn.Module],
        dropout: float,
    ) -> None:
        super().__init__()
        self.is_residual = d_in == d_out
        self.norm = nn.LayerNorm(d_in)
        self.linear = nn.Linear(d_in, d_out)
        self.act = activation()
        # projects context into the hidden dimension and adds it before the activation
        self.context_proj = (
            nn.Linear(context_dim, d_out, bias=False) if context_dim > 0 else None
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.linear(self.norm(x))
        if self.context_proj is not None and context is not None:
            h = h + self.context_proj(context)
        h = self.drop(self.act(h))
        return (x + h) if self.is_residual else h


class FullyConnected(nn.Module):
    """
    Velocity network for conditional flow matching on point-count and
    min-time-per-layer histograms.

    The input x is split into up to three channels of n_layers features each:
        x = [ points_per_layer | (opt) min_time_per_layer | (opt) energy_per_layer ]

    Each channel is independently embedded, then a shared residual MLP backbone
    predicts the velocity, conditioned on the Fourier-embedded flow time t and
    the incident-energy condition.

    Parameters
    ----------
    dim_input     : total input dimension (1, 2, or 3 × n_layers)
    dim_condition : dimension of the external condition vector
    dim_time      : dimension of the (pre-embedded) flow time t
    dim_embedding : width of the per-channel embedding MLPs and context MLP
    n_layers      : number of calorimeter layers (= features per channel)
    hidden_dims   : width of each backbone block; same dim → residual connection
    activation    : pointwise activation class
    dropout       : dropout probability (0 = disabled)
    """

    def __init__(
        self,
        dim_input: int,
        dim_condition: int = 0,
        dim_time: int = 0,
        dim_embedding: int = 128,
        n_layers: int = 78,
        hidden_dims: list[int] | None = None,
        activation: type[nn.Module] = nn.SiLU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim_input = dim_input
        self.n_layers = n_layers
        if hidden_dims is None:
            hidden_dims = [256, 256, 256]

        # ── context MLP: fuses Fourier-time + condition ──────────────────────
        context_in = dim_time + dim_condition
        context_dim = dim_embedding * 2
        self.context_mlp = (
            nn.Sequential(
                nn.Linear(context_in, context_dim),
                activation(),
                nn.Linear(context_dim, context_dim),
                activation(),
            )
            if context_in > 0
            else None
        )
        eff_context_dim = context_dim if self.context_mlp is not None else 0

        # ── per-channel input embeddings ─────────────────────────────────────
        self.points_embedding = nn.Sequential(
            nn.Linear(n_layers, dim_embedding),
            activation(),
            nn.Linear(dim_embedding, dim_embedding),
        )
        total_embed_dim = dim_embedding

        self.time_channel_embedding = None
        if dim_input > n_layers:
            self.time_channel_embedding = nn.Sequential(
                nn.Linear(n_layers, dim_embedding),
                activation(),
                nn.Linear(dim_embedding, dim_embedding),
            )
            total_embed_dim += dim_embedding

        self.energy_channel_embedding = None
        if dim_input > 2 * n_layers:
            self.energy_channel_embedding = nn.Sequential(
                nn.Linear(n_layers, dim_embedding),
                activation(),
                nn.Linear(dim_embedding, dim_embedding),
            )
            total_embed_dim += dim_embedding

        # ── residual backbone ────────────────────────────────────────────────
        self.input_proj = nn.Linear(total_embed_dim, hidden_dims[0])

        self.blocks = nn.ModuleList(
            [
                _Block(d_in, d_out, eff_context_dim, activation, dropout)
                for d_in, d_out in [(hidden_dims[0], hidden_dims[0])]
                + list(zip(hidden_dims, hidden_dims[1:]))
            ]
        )

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dims[-1]),
            nn.Linear(hidden_dims[-1], total_embed_dim),
        )

        # ── output heads: one per input channel ─────────────────────────────
        self.points_head = nn.Linear(total_embed_dim, n_layers)
        self.time_head = (
            nn.Linear(total_embed_dim, n_layers) if dim_input > n_layers else None
        )
        self.energy_head = (
            nn.Linear(total_embed_dim, n_layers) if dim_input > 2 * n_layers else None
        )

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ── context ──────────────────────────────────────────────────────────
        ctx_parts = [t] if t.shape[-1] > 0 else []
        if condition is not None:
            ctx_parts.append(condition)
        context = (
            self.context_mlp(torch.cat(ctx_parts, dim=-1))
            if (self.context_mlp is not None and ctx_parts)
            else None
        )

        # ── embed input channels ─────────────────────────────────────────────
        x_emb = self.points_embedding(x[:, : self.n_layers])
        if self.time_channel_embedding is not None:
            x_emb = torch.cat(
                [
                    x_emb,
                    self.time_channel_embedding(
                        x[:, self.n_layers : 2 * self.n_layers]
                    ),
                ],
                dim=-1,
            )
        if self.energy_channel_embedding is not None:
            x_emb = torch.cat(
                [
                    x_emb,
                    self.energy_channel_embedding(x[:, 2 * self.n_layers :]),
                ],
                dim=-1,
            )

        # ── backbone ─────────────────────────────────────────────────────────
        h = self.input_proj(x_emb)
        for block in self.blocks:
            h = block(h, context)
        h = self.output_proj(h)

        # ── output heads ─────────────────────────────────────────────────────
        out = self.points_head(h)
        if self.time_head is not None:
            out = torch.cat([out, self.time_head(h)], dim=-1)
        if self.energy_head is not None:
            out = torch.cat([out, self.energy_head(h)], dim=-1)
        return out


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
