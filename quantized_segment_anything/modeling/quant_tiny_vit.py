import torch
import torch.nn as nn
import brevitas.nn as qnn
import brevitas.quant as quant
from torch.utils.checkpoint import checkpoint
from timm.models.layers import trunc_normal_, DropPath
from brevitas.nn.utils import merge_bn
from .common import LayerNorm2d


class QuantConv2dBN(nn.Sequential):
    def __init__(
            self,
            ich: int,
            och: int,
            kernel_size: int = 1,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bn_weight_init: int = 1,
            bit_width: int = 8,
            quant_conv: bool = False,
    ):
        super().__init__()
        conv_kwargs = {
            'in_channels': ich,
            'out_channels': och,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'dilation': dilation,
            'groups': groups,
            'bias': False,
        }
        self.add_module(
            name='c',
            module=qnn.QuantConv2d(
                input_quant=quant.Int8ActPerTensorFloat,
                return_quant_tensor=False,
                weight_bit_width=bit_width,
                input_bit_width=bit_width,
                **conv_kwargs
            ) if quant_conv else nn.Conv2d(**conv_kwargs)
        )
        bn = nn.BatchNorm2d(och)
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module(name='bn', module=bn)


class QuantPatchEmbed(nn.Module):
    def __init__(
            self,
            in_chans: int,
            embed_dim: int,
            resolution: int,
            activation: nn.Module,
            bit_width: int = 8,
            quant_conv: bool = False,
    ):
        super().__init__()
        conv_kwargs = {
            'kernel_size': 1,
            'stride': 1,
            'padding': 0,
            'bit_width': bit_width,
            'quant_conv': quant_conv,
        }
        self.patches_resolution = resolution if hasattr(resolution, '__iter__') else (resolution,) * 2
        self.seq = nn.Sequential(
            QuantConv2dBN(in_chans, embed_dim // 2, **conv_kwargs),
            activation(),
            QuantConv2dBN(embed_dim // 2, embed_dim, **conv_kwargs),
        )

    def forward(self, x):
        return self.seq(x)


class QuantMBConv(nn.Module):
    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            expand_ratio: float,
            activation: nn.Module,
            drop_path: float,
            bit_width: int = 8,
            quant_conv: bool = False,
    ):
        super().__init__()
        hidden_chans = int(in_chans * expand_ratio)
        quant_kwargs = dict(bit_width=bit_width, quant_conv=quant_conv)
        self.conv1 = QuantConv2dBN(in_chans, hidden_chans, 1, **quant_kwargs)
        self.act1 = activation()
        self.conv2 = QuantConv2dBN(hidden_chans, hidden_chans, 3, 1, 1, groups=hidden_chans, **quant_kwargs)
        self.act2 = activation()
        self.conv3 = QuantConv2dBN(hidden_chans, out_chans, 1, bn_weight_init=0, **quant_kwargs)
        self.act3 = activation()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.drop_path(self.conv3(x)) + shortcut)
        return x


class QuantPatchMerging(nn.Module):
    def __init__(
            self,
            input_resolution: tuple | list,
            dim: int,
            out_dim: int,
            activation: nn.Module,
            bit_width: int = 8,
            quant_conv: bool = False,
    ):
        super().__init__()
        sc2 = 1 if out_dim in [320, 448, 576] else 2
        quant_kwargs = dict(bit_width=bit_width, quant_conv=quant_conv)
        self.input_resolution = input_resolution
        self.act = activation()
        self.conv1 = QuantConv2dBN(dim, out_dim, 1, 1, 0, **quant_kwargs)
        self.conv2 = QuantConv2dBN(out_dim, out_dim, 3, sc2, 1, groups=out_dim, **quant_kwargs)
        self.conv3 = QuantConv2dBN(out_dim, out_dim, 1, 1, 0, **quant_kwargs)

    def forward(self, x):
        if x.ndim == 3:
            h, w = self.input_resolution
            c = x.shape[-1]
            x = x.reshape(-1, h, w, c).permute(0, 3, 1, 2)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x).flatten(2).transpose(1, 2)
        return x


class QuantConvLayer(nn.Module):
    def __init__(
            self,
            dim: int,
            input_resolution: tuple | list,
            depth: int,
            activation: nn.Module,
            drop_path: float = 0,
            downsample: QuantPatchMerging = None,
            use_checkpoint: bool = False,
            out_dim: int = None,
            conv_expand_ratio: float = 4,
            bit_width: int = 8,
            quant_conv: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList()
        drop_path = drop_path if hasattr(drop_path, '__iter__') else [drop_path] * depth
        assert len(drop_path) == depth
        for dp in drop_path:
            self.blocks.append(QuantMBConv(dim, dim, conv_expand_ratio, activation, dp, bit_width, quant_conv))
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim, out_dim, activation, bit_width, quant_conv)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class QuantMlp(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: int = None,
            out_features: int = None,
            act_layer: nn.Module = nn.GELU,
            drop: float = 0,
            bit_width: int = 8,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = qnn.QuantLinear(
            in_features=in_features,
            out_features=hidden_features,
            input_quant=quant.Int8ActPerTensorFloat,
            return_quant_tensor=False,
            weight_bit_width=bit_width,
            input_bit_width=bit_width,
        )
        self.fc2 = qnn.QuantLinear(
            in_features=hidden_features,
            out_features=out_features,
            input_quant=quant.Int8ActPerTensorFloat,
            return_quant_tensor=False,
            weight_bit_width=bit_width,
            input_bit_width=bit_width,
        )
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class QuantAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            key_dim: int,
            num_heads: int = 8,
            attn_ratio: float = 4,
            resolution: tuple | list = (14, 14),
            bit_width: int = 8
    ):
        super().__init__()
        assert hasattr(resolution, '__iter__') and len(resolution) == 2
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.scale = key_dim ** -0.5
        self.nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        h = self.dh + self.nh_kd * 2
        self.norm = nn.LayerNorm(dim)
        self.qkv = qnn.QuantLinear(
            in_features=dim,
            out_features=h,
            input_quant=quant.Int8ActPerTensorFloat,
            return_quant_tensor=False,
            weight_bit_width=bit_width,
            input_bit_width=bit_width,
        )
        self.proj = qnn.QuantLinear(
            in_features=self.dh,
            out_features=dim,
            input_quant=quant.Int8ActPerTensorFloat,
            return_quant_tensor=False,
            weight_bit_width=bit_width,
            input_bit_width=bit_width,
        )
        self.act_q = qnn.QuantIdentity(return_quant_tensor=True, bit_width=bit_width)
        self.act_k = qnn.QuantIdentity(return_quant_tensor=True, bit_width=bit_width)
        self.act_v = qnn.QuantIdentity(return_quant_tensor=True, bit_width=bit_width)
        self.act_attn = qnn.QuantIdentity(return_quant_tensor=True, bit_width=bit_width)
        points = [(i, j) for i in range(resolution[0]) for j in range(resolution[1])]
        np = len(points)
        offsets = {}
        idx = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in offsets:
                    offsets[offset] = len(offsets)
                idx.append(offsets[offset])
        self.register_buffer('attention_bias_idxs', torch.LongTensor(idx).view(np, np), persistent=False)
        self.attention_biases = nn.Parameter(torch.zeros(num_heads, len(offsets)))

    def forward(self, x):
        n = x.shape[1]
        x = self.norm(x)
        qkv = self.qkv(x)
        qkv = qkv.reshape(-1, n, self.num_heads, self.key_dim * 2 + self.d)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.d], -1)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3).transpose(-2, -1)
        v = v.permute(0, 2, 1, 3)
        attn = torch.matmul(self.act_q(q), self.act_k(k)) * self.scale
        if self.training:
            attn = attn + self.attention_biases[:, self.attention_bias_idxs]
        else:
            attn = attn + self.ab
        attn = attn.softmax(dim=-1)
        x = torch.matmul(self.act_attn(attn), self.act_v(v))
        x = self.proj(x.transpose(1, 2).reshape(-1, n, self.dh))
        return x

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:  # eval() call self.train(False)
            self.register_buffer('ab', self.attention_biases[:, self.attention_bias_idxs], persistent=False)
        return self


class QuantTinyViTBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            input_resolution: tuple | list,
            num_heads: int,
            window_size: int = 7,
            mlp_ratio: float = 4,
            drop: float = 0,
            drop_path: float = 0,
            local_conv_size: int = 3,
            activation: nn.Module = nn.GELU,
            bit_width: int = 8,
            quant_conv: bool = False,
    ):
        super().__init__()
        assert window_size > 0 and dim % num_heads == 0
        head_dim = dim // num_heads
        window_resolution = (window_size,) * 2
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.attn = QuantAttention(dim, head_dim, num_heads, 1, window_resolution, bit_width)
        self.mlp = QuantMlp(dim, int(dim * mlp_ratio), act_layer=activation, drop=drop, bit_width=bit_width)
        self.local_conv = QuantConv2dBN(
            ich=dim,
            och=dim,
            kernel_size=local_conv_size,
            stride=1,
            padding=local_conv_size // 2,
            groups=dim,
            bit_width=bit_width,
            quant_conv=quant_conv
        )

    def forward(self, x):
        h, w = self.input_resolution
        _, l, c = x.shape
        assert l == h * w
        shortcut = x
        if h == self.window_size == w:
            x = self.attn(x)
        else:
            from torch.nn.functional import pad
            x = x.reshape(-1, h, w, c)
            pad_b = (self.window_size - h % self.window_size) % self.window_size
            pad_r = (self.window_size - w % self.window_size) % self.window_size
            x = pad(x, (0, 0, 0, pad_r, 0, pad_b))
            ph, pw = h + pad_b, w + pad_r
            nh, nw = ph // self.window_size, pw // self.window_size
            x = x.reshape(-1, nh, self.window_size, nw, self.window_size, c)
            x = x.transpose(2, 3)
            x = x.reshape(-1, self.window_size * self.window_size, c)
            x = self.attn(x)
            x = x.reshape(-1, nh, nw, self.window_size, self.window_size, c)
            x = x.transpose(2, 3)
            x = x.reshape(-1, ph, pw, c)
            x = x[:, :h, :w].contiguous().reshape(-1, l, c)
        x = shortcut + self.drop_path(x)
        x = x.transpose(1, 2).reshape(-1, c, h, w)
        x = self.local_conv(x)
        x = x.reshape(-1, c, l).transpose(1, 2)
        x = x + self.drop_path(self.mlp(x))
        return x


class QuantBasicLayer(nn.Module):
    def __init__(
            self,
            dim: int,
            input_resolution: tuple | list,
            depth: int,
            num_heads: int,
            window_size: int,
            mlp_ratio: float = 4,
            drop: float = 0,
            drop_path: float | tuple | list = 0,
            downsample: QuantPatchMerging = None,
            use_checkpoint: bool = False,
            local_conv_size: int = 3,
            activation: nn.Module = nn.GELU,
            out_dim: int = None,
            bit_width: int = 8,
            quant_conv: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList()
        drop_path = drop_path if hasattr(drop_path, '__iter__') else [drop_path] * depth
        assert len(drop_path) == depth
        for dp in drop_path:
            self.blocks.append(QuantTinyViTBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=dp,
                local_conv_size=local_conv_size,
                activation=activation,
                bit_width=bit_width,
                quant_conv=quant_conv,
            ))
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim, out_dim, activation, bit_width, quant_conv)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class QuantTinyViT(nn.Module):
    def __init__(
            self,
            img_size: int = 256,
            in_chans: int = 3,
            embed_dims: tuple | list = (64, 128, 160, 320),
            depths: tuple | list = (2, 2, 6, 2),
            num_heads: tuple | list = (2, 4, 5, 10),
            window_sizes: tuple | list = (7, 7, 14, 7),
            mlp_ratio: float = 4,
            drop_rate: float = 0,
            drop_path_rate: float = 0,
            use_checkpoint: bool = False,
            mbconv_expand_ratio: float = 4,
            local_conv_size: int = 3,
            bit_width: int = 8,
            quant_conv: bool = False,
            **kwargs
    ):
        super().__init__()
        num_layers = len(depths)
        activation = nn.GELU
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.img_size = img_size
        self.patch_embed = QuantPatchEmbed(in_chans, embed_dims[0], img_size, activation, bit_width, quant_conv)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            kwargs = {
                'dim': embed_dims[i],
                'input_resolution': (img_size // (2 ** (i - 1 if i == num_layers - 1 else i)),) * 2,
                'depth': depths[i],
                'drop_path': dpr[sum(depths[:i]): sum(depths[:i+1])],
                'downsample': QuantPatchMerging if i < num_layers - 1 else None,
                'use_checkpoint': use_checkpoint,
                'out_dim': embed_dims[min(i + 1, len(embed_dims) - 1)],
                'activation': activation,
                'bit_width': bit_width,
                'quant_conv': quant_conv,
            }
            if i == 0:
                layer = QuantConvLayer(conv_expand_ratio=mbconv_expand_ratio, **kwargs)
            else:
                layer = QuantBasicLayer(
                    num_heads=num_heads[i],
                    window_size=window_sizes[i],
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    local_conv_size=local_conv_size,
                    **kwargs
                )
            self.layers.append(layer)
        c1_kwargs = {
            'in_channels': embed_dims[-1],
            'out_channels': 256,
            'kernel_size': 1,
            'bias': False,
        }
        c2_kwargs = {
            'in_channels': 256,
            'out_channels': 256,
            'kernel_size': 3,
            'padding': 1,
            'bias': False,
        }
        self.neck = nn.Sequential(
            qnn.QuantConv2d(
                input_quant=quant.Int8ActPerTensorFloat,
                return_quant_tensor=False,
                weight_bit_width=bit_width,
                input_bit_width=bit_width,
                **c1_kwargs
            ) if quant_conv else nn.Conv2d(**c1_kwargs),
            LayerNorm2d(256),
            qnn.QuantConv2d(
                input_quant=quant.Int8ActPerTensorFloat,
                return_quant_tensor=False,
                weight_bit_width=bit_width,
                input_bit_width=bit_width,
                **c2_kwargs
            ) if quant_conv else nn.Conv2d(**c2_kwargs),
            LayerNorm2d(256),
        )
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        c = x.shape[-1]
        x = x.reshape(-1, 64, 64, c).permute(0, 3, 1, 2)
        x = self.neck(x)
        return x

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
