import quantized_segment_anything as qsa
import torch
import os
import openvino as ov
from brevitas.export import export_onnx_qcdq


class SAMDecoder(torch.nn.Module):
    def __init__(self, sam_model):
        super(SAMDecoder, self).__init__()
        self.prompt_encoder = sam_model.prompt_encoder
        self.mask_decoder = sam_model.mask_decoder
        self.quant_md = sam_model.quant_md
        self.eval()

    def forward(self, image_embedding, boxes):
        sparse_embeddings, dense_embeddings = self.prompt_encoder(boxes=boxes)
        low_res_masks = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        if not self.quant_md:
            low_res_masks = low_res_masks[0]
        return low_res_masks[:, 0, :, :]


model = qsa.QuantLiteMedSAM('ckpt/quantized_dist_iemd_final.2.e16.pth', True, True)
# model = qsa.QuantLiteMedSAM('ckpt/lite_medsam.pth', False, False)
model = model.eval()
save_dir = './docker_python'
os.makedirs(save_dir, exist_ok=True)
sample_img = torch.randn(1, 3, 256, 256)
sample_boxes = torch.randn(1, 1, 4)
ie = model.image_encoder
sample_ie = ie(sample_img)
enc_fp = os.path.join(save_dir, 'encoder.onnx')
dec_fp = os.path.join(save_dir, 'decoder.onnx')
enc_ir_fp = os.path.join(save_dir, 'encoder.xml')
dec_ir_fp = os.path.join(save_dir, 'decoder.xml')

if not os.path.exists(enc_fp):
    export_onnx_qcdq(
        module=ie,
        args=sample_img,
        export_path=enc_fp,
        input_names=['image'],
        output_names=['image_embedding'],
    )

decoder = SAMDecoder(model)
if not os.path.exists(dec_fp):
    export_onnx_qcdq(
        module=decoder,
        args=(sample_ie, sample_boxes),
        export_path=dec_fp,
        input_names=['image_embedding', 'boxes'],
        output_names=['low_res_masks'],
        dynamic_axes={
            'boxes': {0: 'batch_size'},
            'low_res_masks': {0: 'batch_size'},
        }
    )

if not os.path.exists(enc_ir_fp):
    ir_encoder = ov.convert_model(enc_fp)
    ov.save_model(ir_encoder, enc_ir_fp, False)

if not os.path.exists(dec_ir_fp):
    ir_decoder = ov.convert_model(dec_fp)
    ov.save_model(ir_decoder, dec_ir_fp, False)
