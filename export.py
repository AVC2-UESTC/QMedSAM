import quantized_segment_anything as qsa
import torch
import os
from brevitas.export import export_onnx_qcdq


model = qsa.QuantLiteMedSAM('./ckpt/quantiemd_v5.pth', True, True).eval()
export_onnx_fp = 'QuantIEMDc900.onnx'
sample_img = torch.randn(1, 3, 256, 256)
sample_boxes = torch.randn(1, 1, 4)
if not os.path.exists(export_onnx_fp):
    export_onnx_qcdq(
        module=model,
        args=(sample_img, sample_boxes),
        export_path=export_onnx_fp,
        disable_warnings=False,
        input_names=['image', 'boxes'],
        output_names=['low_res_masks'],
        dynamic_axes={
            'boxes': {0: 'batch'},
            'low_res_masks': {0: 'batch'}
        }
    )

#
# model = qsa.QuantLiteMedSAM('../lite_medsam.pth', False, False)
# export_fp = 'fp_lite.onnx'
# if not os.path.exists(export_fp):
#     torch.onnx.export(
#         model=model,
#         args=(sample_img, sample_boxes),
#         f=export_fp,
#         input_names=['image', 'boxes'],
#         output_names=['low_res_masks'],
#         dynamic_axes={
#             'boxes': {0: 'batch'},
#             'low_res_masks': {0: 'batch'}
#         }
#     )
