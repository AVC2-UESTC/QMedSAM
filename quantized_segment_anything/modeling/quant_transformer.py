import torch
import torch.nn as nn
import brevitas.nn as qnn
import brevitas.quant as quant
from .common import QuantMLPBlock


class QuantAttention(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            downsample_rate: float = 1,
            bit_width: int = 8,
    ):
        super().__init__()
        internal_dim = embedding_dim // downsample_rate
        assert internal_dim % num_heads == 0
        self.num_heads = num_heads
        self.act_q = qnn.QuantIdentity(return_quant_tensor=True, bit_width=bit_width)
        self.act_k = qnn.QuantIdentity(return_quant_tensor=True, bit_width=bit_width)
        self.act_v = qnn.QuantIdentity(return_quant_tensor=True, bit_width=bit_width)
        self.act_attn = qnn.QuantIdentity(return_quant_tensor=True, bit_width=bit_width)
        self.q_proj = qnn.QuantLinear(
            in_features=embedding_dim,
            out_features=internal_dim,
            input_quant=quant.Int8ActPerTensorFloat,
            return_quant_tensor=False,
            weight_bit_width=bit_width,
            input_bit_width=bit_width,
        )
        self.k_proj = qnn.QuantLinear(
            in_features=embedding_dim,
            out_features=internal_dim,
            input_quant=quant.Int8ActPerTensorFloat,
            return_quant_tensor=False,
            weight_bit_width=bit_width,
            input_bit_width=bit_width,
        )
        self.v_proj = qnn.QuantLinear(
            in_features=embedding_dim,
            out_features=internal_dim,
            input_quant=quant.Int8ActPerTensorFloat,
            return_quant_tensor=False,
            weight_bit_width=bit_width,
            input_bit_width=bit_width,
        )
        self.out_proj = qnn.QuantLinear(
            in_features=internal_dim,
            out_features=embedding_dim,
            input_quant=quant.Int8ActPerTensorFloat,
            return_quant_tensor=False,
            weight_bit_width=bit_width,
            input_bit_width=bit_width,
        )

    def forward(self, q, k, v):
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads).permute(0, 1, 3, 2)
        v = self._separate_heads(v, self.num_heads)
        c_per_head = q.shape[-1]
        attn = torch.matmul(self.act_q(q), self.act_k(k))
        if torch.onnx.is_in_onnx_export():
            attn = attn / torch.sqrt(c_per_head)
        else:
            from math import sqrt
            attn = attn / sqrt(c_per_head)
        attn = attn.softmax(dim=-1)
        out = torch.matmul(self.act_attn(attn), self.act_v(v))
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out

    @staticmethod
    def _separate_heads(x, num_heads):
        _, n, c = x.shape
        x = x.reshape(-1, n, num_heads, c // num_heads)
        x = x.transpose(1, 2)
        return x

    @staticmethod
    def _recombine_heads(x):
        _, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        x = x.reshape(-1, n_tokens, n_heads * c_per_head)
        return x


class QuantTwoWayAttentionBlock(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            mlp_dim: int = 2048,
            activation: nn.Module = nn.ReLU,
            attention_downsample_rate: float = 2,
            skip_first_layer_pe: bool = False,
            bit_width: int = 8,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.self_attn = QuantAttention(embedding_dim, num_heads, bit_width=bit_width)
        self.cross_attn_token_to_image = QuantAttention(embedding_dim, num_heads, attention_downsample_rate, bit_width)
        self.cross_attn_image_to_token = QuantAttention(embedding_dim, num_heads, attention_downsample_rate, bit_width)
        self.mlp = QuantMLPBlock(embedding_dim, mlp_dim, activation, bit_width=bit_width)
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries, keys, query_pe, key_pe):
        if self.skip_first_layer_pe:
            queries = self.self_attn(queries, queries, queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q, q, queries)
            queries = queries + attn_out
        queries = self.norm1(queries)
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q, k, keys)
        queries = queries + attn_out
        queries = self.norm2(queries)
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(k, q, queries)
        keys = keys + attn_out
        keys = self.norm4(keys)
        return queries, keys


class QuantTwoWayTransformer(nn.Module):
    def __init__(
            self,
            depth: int,
            embedding_dim: int,
            num_heads: int,
            mlp_dim: int,
            activation: nn.Module = nn.ReLU,
            attention_downsample_rate: float = 2,
            bit_width: int = 8,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(QuantTwoWayAttentionBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                activation=activation,
                attention_downsample_rate=attention_downsample_rate,
                skip_first_layer_pe=i == 0,
                bit_width=bit_width,
            ))
        self.final_attn_token_to_image = QuantAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            downsample_rate=attention_downsample_rate,
            bit_width=bit_width,
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(self, image_embedding, image_pe, point_embedding):
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)
        queries = point_embedding
        keys = image_embedding
        for layer in self.layers:
            queries, keys = layer(queries, keys, point_embedding, image_pe)
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q, k, keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)
        return queries, keys
