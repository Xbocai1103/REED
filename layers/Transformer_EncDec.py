import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, encoder_layers):
        super(Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList(
            encoder_layers
        )

    def forward(self, x):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        return x


class EncoderLayer(nn.Module):

    def __init__(self, configs):
        super().__init__()

        self.norm1 = nn.LayerNorm(
            configs.d_model
        )

        self.attn = CoTAR(
            configs.d_model,
            configs.d_model // 4,
        )

        self.norm2 = nn.LayerNorm(
            configs.d_model
        )

        self.mlp = nn.Sequential(
            nn.Linear(
                configs.d_model,
                int(
                    2
                    * configs.d_model
                ),
            ),
            nn.GELU(),
            nn.Dropout(
                configs.dropout
            ),
            nn.Linear(
                int(
                    2
                    * configs.d_model
                ),
                configs.d_model,
            ),
            nn.Dropout(
                configs.dropout
            ),
        )

    def forward(self, x):
        x_att = self.attn(
            x
        )

        x = self.norm1(
            x
            + x_att
        )

        x_ffn = self.mlp(
            x
        )

        x = self.norm2(
            x
            + x_ffn
        )

        return x


class CoTAR(nn.Module):

    def __init__(
        self,
        d_model,
        d_core=64,
    ):
        super(CoTAR, self).__init__()

        self.lin1 = nn.Linear(
            d_model,
            d_model,
        )

        self.lin2 = nn.Linear(
            d_model,
            d_core,
        )

        self.lin3 = nn.Linear(
            d_model + d_core,
            d_model,
        )

        self.lin4 = nn.Linear(
            d_model,
            d_model,
        )

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(
                "CoTAR expects [B,N,D], "
                f"got {tuple(x.shape)}."
            )

        _, n_tokens, _ = x.shape

        core = F.gelu(
            self.lin1(
                x
            )
        )

        core = self.lin2(
            core
        )

        weight = F.softmax(
            core,
            dim=1,
        )

        core = torch.sum(
            core
            * weight,
            dim=1,
            keepdim=True,
        ).repeat(
            1,
            n_tokens,
            1,
        )

        core_cat = torch.cat(
            [
                x,
                core,
            ],
            dim=-1,
        )

        core_cat = F.gelu(
            self.lin3(
                core_cat
            )
        )

        core_cat = self.lin4(
            core_cat
        )

        return core_cat


class Attention(nn.Module):

    def __init__(
        self,
        dim,
        n_heads,
        init=0,
    ):
        super().__init__()

        self.n_heads = int(
            n_heads
        )

        if dim % self.n_heads != 0:
            raise ValueError(
                f"dim={dim} must be divisible by "
                f"n_heads={self.n_heads}."
            )

        self.qkv = nn.Linear(
            dim,
            dim * 3,
        )

        self.out_proj = nn.Linear(
            dim,
            dim,
        )

        self.scale = (
            dim
            // self.n_heads
        ) ** -0.5

        if init:
            self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(
                module,
                nn.Linear,
            ):
                init.xavier_uniform_(
                    module.weight
                )

                if module.bias is not None:
                    init.zeros_(
                        module.bias
                    )

    def forward(self, x):
        batch_size, n_tokens, dim = x.shape

        qkv = (
            self.qkv(x)
            .reshape(
                batch_size,
                n_tokens,
                3,
                self.n_heads,
                dim // self.n_heads,
            )
            .permute(
                2,
                0,
                3,
                1,
                4,
            )
        )

        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (
            q
            @ k.transpose(-2, -1)
        ) * self.scale

        attn = attn.softmax(
            dim=-1
        )

        output = (
            attn
            @ v
        ).transpose(
            1,
            2,
        ).reshape(
            batch_size,
            n_tokens,
            dim,
        )

        output = self.out_proj(
            output
        )

        return output
