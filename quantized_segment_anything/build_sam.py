import torch
import brevitas
from functools import partial
from pathlib import Path
from .modeling import (
    Sam,
    QuantImageEncoderViT,
    ImageEncoderViT,
    PromptEncoder,
    MaskDecoder,
    TwoWayTransformer
)


def build_quant_sam(bit_width=8, checkpoint=None):
    encoder_embed_dim = 768
    encoder_depth = 12
    encoder_num_heads = 12
    encoder_global_attn_indexes = [2, 5, 8, 11]
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=QuantImageEncoderViT(
            img_size=image_size,
            patch_size=vit_patch_size,
            in_chans=3,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=4,
            out_chans=prompt_embed_dim,
            qkv_bias=True,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            act_layer=torch.nn.GELU,
            use_abs_pos=True,
            use_rel_pos=True,
            window_size=14,
            global_attn_indexes=encoder_global_attn_indexes,
            bit_width=bit_width,
            output_qt=False,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    checkpoint = Path(checkpoint)
    if checkpoint is not None:
        brevitas.config.IGNORE_MISSING_KEYS = True
        with open(checkpoint, 'rb') as f:
            state_dict = torch.load(f, map_location="cpu")
        sam.load_state_dict(state_dict)
    return sam


def build_sam(checkpoint=None):
    encoder_embed_dim = 768
    encoder_depth = 12
    encoder_num_heads = 12
    encoder_global_attn_indexes = [2, 5, 8, 11]
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    checkpoint = Path(checkpoint)
    if checkpoint is not None:
        with open(checkpoint, 'rb') as f:
            state_dict = torch.load(f, map_location="cpu")
        sam.load_state_dict(state_dict)
    return sam
