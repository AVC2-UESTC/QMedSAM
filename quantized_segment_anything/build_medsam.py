import torch
import torch.nn as nn
from torch.nn.functional import interpolate


class MedSAM(nn.Module):
    def __init__(
            self,
            sam_model,
            grad_ie: bool = True,
            grad_pe: bool = False,
            grad_md: bool = False,
            upscale_mask: bool = True,
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
        self.upscale_mask = upscale_mask
        self.image_encoder = sam_model.image_encoder
        self.prompt_encoder = sam_model.prompt_encoder
        self.mask_decoder = sam_model.mask_decoder
        self.grad_ie = grad_ie
        self.grad_pe = grad_pe
        self.grad_md = grad_md
        if not self.grad_ie:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        if not self.grad_pe:
            for param in self.prompt_encoder.parameters():
                param.requires_grad = False
        if not self.grad_pe:
            for param in self.mask_decoder.parameters():
                param.requires_grad = False

    def forward(self, image, box):
        if self.grad_ie:
            image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        else:
            with torch.no_grad():
                image_embedding = self.image_encoder(image)
        if self.grad_pe:
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)
            sparse_embeddings, dense_embeddings = self.prompt_encoder(boxes=box_torch)
        else:
            with torch.no_grad():
                box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
                if len(box_torch.shape) == 2:
                    box_torch = box_torch[:, None, :]
                sparse_embeddings, dense_embeddings = self.prompt_encoder(boxes=box_torch)
        if self.grad_ie or self.grad_pe or self.grad_ie:
            low_res_masks, _ = self.mask_decoder(
                image_embeddings=image_embedding,  # (B, 256, 64, 64)
                image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                multimask_output=False,
            )
        else:
            with torch.no_grad():
                low_res_masks, _ = self.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
        if self.upscale_mask:
            return_mask = interpolate(
                low_res_masks,
                size=(image.shape[2], image.shape[3]),
                mode="bilinear",
                align_corners=False,
            )
        else:
            return_mask = low_res_masks
        if self.mode == 'default':
            return return_mask
        elif self.mode == 'with_ie_output':
            return return_mask, image_embedding
        else:
            raise NotImplementedError
