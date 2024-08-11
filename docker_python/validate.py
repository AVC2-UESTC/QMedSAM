import os
import cv2
import numpy as np
import openvino as ov
import openvino.properties as props
import argparse
from functools import partial
from time import time


def apply_boxes(boxes, oh, ow, nh, nw):
    boxes = apply_coords(boxes.reshape(-1, 2, 2), oh, ow, nh, nw)
    return boxes.reshape(-1, 4)


def apply_coords(coords, oh, ow, nh, nw):
    from copy import deepcopy
    coords = deepcopy(coords).astype(float)
    coords[..., 0] = coords[..., 0] * (nw / ow)
    coords[..., 1] = coords[..., 1] * (nh / oh)
    return coords


def validator(args):
    dataset_fp = args.input_dir
    save_fp = args.output_dir
    img_size = args.img_size
    os.makedirs(save_fp, exist_ok=True)
    core = ov.Core()
    core.set_property({props.cache_dir: save_fp})
    encoder_onnx = core.read_model(model=os.path.join(args.model_dir, 'encoder.xml'))
    decoder_onnx = core.read_model(model=os.path.join(args.model_dir, 'decoder.xml'))
    compiled_encoder = core.compile_model(model=encoder_onnx, device_name='CPU')
    compiled_decoder = core.compile_model(model=decoder_onnx, device_name='CPU')
    infer = partial(inference, encoder=compiled_encoder, decoder=compiled_decoder, block_size=args.block_size)
    log = ''
    iterator = os.listdir(dataset_fp)
    if args.verbose:
        from tqdm import tqdm
        iterator = tqdm(iterator)
    for clip_fn in iterator:
        s = time()
        if os.path.isdir(clip_fn):
            continue
        if not clip_fn.endswith('.npz'):
            continue
        clip_fp = os.path.join(dataset_fp, clip_fn)
        clip_save_fp = os.path.join(save_fp, clip_fn)
        if clip_fn.startswith('2D'):
            segs = infer_2d(clip_fp, infer, img_size)
        else:
            segs = infer_3d(clip_fp, infer, img_size)
        np.savez_compressed(clip_save_fp, segs=segs)
        t = time()
        log += f'inference {clip_fn} within {t - s:.3f}s.\n'
    if args.verbose:
        print(log)


def infer_2d(npz_fp, infer_method, img_size):
    _npz = np.load(npz_fp)
    img_3c, bbox = _npz['imgs'], _npz['boxes']
    oh, ow = img_3c.shape[:2]
    scale = img_size * 1.0 / max(oh, ow)
    nh, nw = int(oh * scale + .5), int(ow * scale + .5)
    ph, pw = img_size - nh, img_size - nw
    segs = np.zeros(shape=(oh, ow), dtype=np.uint8)
    bbox = apply_boxes(bbox, oh, ow, nh, nw)
    img_3c = cv2.resize(img_3c, dsize=(nw, nh), interpolation=cv2.INTER_CUBIC)
    img_3c = np.pad(img_3c, pad_width=((0, ph), (0, pw), (0, 0)))
    img_01 = (img_3c - img_3c.min()) / np.clip(img_3c.max() - img_3c.min(), a_min=1e-8, a_max=None)
    img_c3 = np.expand_dims(np.transpose(img_01, axes=(2, 0, 1)), axis=0)
    low_res_masks: np.ndarray = infer_method(img_c3, bbox)
    for idx, pd in enumerate(low_res_masks, 1):
        pd = pd >= 0  # bool
        pd = cv2.resize(pd * 1.0, dsize=[max(oh, ow)]*2, interpolation=cv2.INTER_LINEAR)
        pd = pd[:oh, :ow]
        segs[pd >= .5] = idx
    return segs


def infer_3d(npz_fp, infer_method, img_size):
    _npz = np.load(npz_fp)
    img_3d, boxes = _npz['imgs'], _npz['boxes']
    len_z = img_3d.shape[0]
    box_z = [None for _ in range(len_z)]
    idx_z = [None for _ in range(len_z)]
    for idx, box in enumerate(boxes, 1):
        x_min, y_min, z_min, x_max, y_max, z_max = box
        box_2d = np.array([[x_min, y_min, x_max, y_max]])
        z_middle = int((z_max - z_min) / 2 + z_min)
        z_max = min(z_max + 1, len_z)
        z_min = max(z_min - 1, -1)  # fix to close interval
        for z in range(z_middle, z_max):
            if box_z[z] is None:
                box_z[z] = box_2d
                idx_z[z] = [idx]
            else:
                box_z[z] = np.concatenate((box_z[z], box_2d), axis=0)
                idx_z[z] += [idx]
        for z in range(z_middle - 1, z_min, -1):
            if box_z[z] is None:
                box_z[z] = box_2d
                idx_z[z] = [idx]
            else:
                box_z[z] = np.concatenate((box_z[z], box_2d), axis=0)
                idx_z[z] += [idx]
    segs = np.zeros_like(img_3d, dtype=np.uint8)
    for z in range(len_z):
        bbox = box_z[z]
        if bbox is None:
            continue
        img = img_3d[z]
        img_3c = np.repeat(np.expand_dims(img, axis=-1), repeats=3, axis=-1)
        oh, ow = img_3c.shape[:2]
        scale = img_size * 1.0 / max(oh, ow)
        nh, nw = int(oh * scale + .5), int(ow * scale + .5)
        ph, pw = img_size - nh, img_size - nw
        bbox = apply_boxes(bbox, oh, ow, nh, nw)
        img_3c = cv2.resize(img_3c, dsize=(nw, nh), interpolation=cv2.INTER_CUBIC)
        img_3c = np.pad(img_3c, pad_width=((0, ph), (0, pw), (0, 0)))
        img_01 = (img_3c - img_3c.min()) / np.clip(img_3c.max() - img_3c.min(), a_min=1e-8, a_max=None)
        img_c3 = np.expand_dims(np.transpose(img_01, axes=(2, 0, 1)), axis=0)
        low_res_masks: np.ndarray = infer_method(img_c3, bbox)
        for idx, pd in zip(idx_z[z], low_res_masks):
            pd = pd >= 0
            pd = cv2.resize(pd * 1.0, dsize=[max(oh, ow)]*2, interpolation=cv2.INTER_LINEAR)
            pd = pd[:oh, :ow]
            segs[z][pd >= .5] = idx
    return segs


def inference(img, box, encoder, decoder, block_size=64):
    batch_size = box.shape[0]
    image_embedding = encoder(inputs={'image': img})['image_embedding']
    low_res_mask = np.empty((0, 256, 256))
    num_blocks = batch_size // block_size
    if batch_size % block_size:
        num_blocks += 1
    for i in range(num_blocks):
        box_i = box[i * block_size: min((i + 1) * block_size, batch_size), None, :]
        low_res_masks_i = decoder(inputs={'image_embedding': image_embedding, 'boxes': box_i})['low_res_masks']
        low_res_mask = np.concatenate((low_res_mask, low_res_masks_i), axis=0)
    return low_res_mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str)
    parser.add_argument('-o', '--output_dir', type=str)
    parser.add_argument('-m', '--model_dir', type=str, default='.')
    parser.add_argument('-b', '--block_size', type=int, default=96)
    parser.add_argument('-s', '--img_size', type=int, default=256)
    parser.add_argument('-v', '--verbose', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    validator(args)
