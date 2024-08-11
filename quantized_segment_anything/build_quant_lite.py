import torch
import torch.nn as nn
import brevitas
from .modeling import (
    QuantTwoWayTransformer, TwoWayTransformer,
    QuantMaskDecoder, MaskDecoder,
    QuantTinyViT, TinyViT,
    PromptEncoder
)


class QuantLiteMedSAM(nn.Module):
    def __init__(
            self,
            ckpt: str = None,
            quant_ie: bool = True,
            quant_md: bool = False,
            bits: int = 8,
            quant_conv: bool = True,
            strict_state_dict: bool = False,
            mode: str | int = 'default',
    ):
        super().__init__()
        mode_list = ['default', 'with_ie_output']
        if type(mode) is str:
            assert mode in mode_list
        elif type(mode) is int:
            assert mode < len(mode_list)
            mode = mode_list[mode]
        else:
            raise ValueError('mode must be either str or int')
        self.mode = mode
        ie_param = {
            'img_size': 256,
            'in_chans': 3,
            'embed_dims': (64, 128, 160, 320),
            'depths': (2, 2, 6, 2),
            'num_heads': (2, 4, 5, 10),
            'window_sizes': (7, 7, 14, 7),
            'mlp_ratio': 4,
            'drop_rate': 0,
            'drop_path_rate': 0,
            'use_checkpoint': False,
            'mbconv_expand_ratio': 4,
            'local_conv_size': 3,
            'layer_lr_decay': 0.8,
        }
        if quant_ie:
            ie_param['bit_width'] = bits
            ie_param['quant_conv'] = quant_conv
        pe_param = {
            'embed_dim': 256,
            'image_embedding_size': (64, 64),
            'input_image_size': (256, 256),
            'mask_in_chans': 16,
        }
        two_way_transformer = QuantTwoWayTransformer if quant_md else TwoWayTransformer
        transformer_param = {
            'depth': 2,
            'embedding_dim': 256,
            'mlp_dim': 2048,
            'num_heads': 8,
        }
        if quant_md:
            transformer_param['bit_width'] = bits
        md_param = {
            'num_multimask_outputs': 3,
            'transformer': two_way_transformer(**transformer_param),
            'transformer_dim': 256,
            'iou_head_depth': 3,
            'iou_head_hidden_dim': 256,
        }
        if quant_md:
            md_param['bit_width'] = bits
        image_encoder = QuantTinyViT if quant_ie else TinyViT
        mask_decoder = QuantMaskDecoder if quant_md else MaskDecoder
        self.image_encoder = image_encoder(**ie_param)
        self.prompt_encoder = PromptEncoder(**pe_param)
        self.mask_decoder = mask_decoder(**md_param)
        self.quant_ie = quant_ie
        self.quant_md = quant_md
        self.grad_ie = True
        self.grad_pe = False
        self.grad_md = False
        self.modify_grad(True, False, False)
        self.eval()
        if ckpt is not None:
            brevitas.config.IGNORE_MISSING_KEYS = not strict_state_dict
            state_dict = torch.load(ckpt, map_location='cpu')
            self.load_state_dict(state_dict)

    def modify_grad(self, grad_ie=True, grad_pe=False, grad_md=False):
        self.grad_ie = grad_ie
        self.grad_pe = grad_pe
        self.grad_md = grad_md
        for param in self.image_encoder.parameters():
            param.requires_grad = grad_ie
        for param in self.prompt_encoder.parameters():
            param.requires_grad = grad_pe
        for param in self.mask_decoder.parameters():
            param.requires_grad = grad_md

    def forward(self, image, boxes):
        assert boxes.ndim == 3  # box = box[:, None, :]  # (bs, 1, 4)
        image_embedding = self.image_encoder(image)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(boxes=boxes)
        low_res_masks = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        return_mask = low_res_masks if self.quant_md else low_res_masks[0]
        if self.mode is 'default':
            return return_mask
        elif self.mode is 'with_ie_output':
            return return_mask, image_embedding
        else:
            raise NotImplementedError

    @torch.no_grad()
    def fuse(self):
        if self.quant_ie:
            self.image_encoder.fuse()
