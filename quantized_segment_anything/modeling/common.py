import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat
from typing import Type


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class QuantMLPBlock(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            mlp_dim: int,
            act: Type[nn.Module] = nn.GELU,
            bit_width: int = 8,
            input_qt: bool = False,
            output_qt: bool = False,
    ) -> None:
        super().__init__()
        self.lin1 = qnn.QuantLinear(
            in_features=embedding_dim,
            out_features=mlp_dim,
            bias=True,
            input_quant=Int8ActPerTensorFloat if not input_qt else None,
            return_quant_tensor=False,
            weight_bit_width=bit_width,
            input_bit_width=bit_width if not input_qt else None,
        )  # float output
        self.lin2 = qnn.QuantLinear(
            in_features=mlp_dim,
            out_features=embedding_dim,
            bias=True,
            input_quant=Int8ActPerTensorFloat,
            output_quant=Int8ActPerTensorFloat if output_qt else None,
            return_quant_tensor=True,
            weight_bit_width=bit_width,
            input_bit_width=bit_width,
            output_bit_width=bit_width if output_qt else None,
        )
        self.act = act()

    def forward(self, x):
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
