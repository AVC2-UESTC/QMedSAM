import torch
import os
import pickle
import numpy as np
import random
import cv2
from torch.utils.data import Dataset
from typing import Any, Iterable, List
from glob import glob
from tqdm import tqdm
from collections import OrderedDict


def bsearch(li: List[int], x: int):
    """
    Find the leftest li[i] such that li[i] >= x
    :param li: list to search
    :param x: target value
    :return: i
    """
    left, right = 0, len(li) - 1
    answer = right
    assert li[right] >= x
    while left <= right:
        mid = (left + right) // 2
        if li[mid] >= x:
            answer = mid
            right = mid - 1
        else:
            left = mid + 1
    return answer


def preprocess(img_3c, gt, image_size, bbox_shift, rotate=False):
    oh, ow = img_3c.shape[:2]
    scale = image_size * 1.0 / max(oh, ow)
    nh, nw = int(oh * scale + .5), int(ow * scale + .5)
    img_3c = cv2.resize(img_3c, dsize=(nw, nh), interpolation=cv2.INTER_NEAREST)
    gt = cv2.resize(gt, dsize=(nw, nh), interpolation=cv2.INTER_NEAREST)
    ph, pw = image_size - nh, image_size - nw
    img_3c = np.pad(img_3c, ((0, ph), (0, pw), (0, 0)))
    gt = np.pad(gt, ((0, ph), (0, pw)))
    img_01 = (img_3c - img_3c.min()) / np.clip(img_3c.max() - img_3c.min(), a_min=1e-8, a_max=None)
    img_c3 = np.transpose(img_01, (2, 0, 1))
    gt = np.uint8(gt)
    label_ids = np.unique(gt).tolist()
    if label_ids[0] == 0 and len(label_ids) > 1:
        label_ids = label_ids[1:]
    gt = gt == random.choice(label_ids)
    if rotate:
        if random.random() > .5:
            img_c3 = np.ascontiguousarray(np.flip(img_c3, -1))
            gt = np.ascontiguousarray(np.flip(gt, -1))
        if random.random() > .5:
            img_c3 = np.ascontiguousarray(np.flip(img_c3, -2))
            gt = np.ascontiguousarray(np.flip(gt, -2))
    y_indices, x_indices = np.where(gt)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    gt = np.uint8(gt)
    y_min = max(0, y_min - random.randint(0, bbox_shift))
    x_min = max(0, x_min - random.randint(0, bbox_shift))
    y_max = min(image_size - 1, y_max + random.randint(0, bbox_shift))
    x_max = min(image_size - 1, x_max + random.randint(0, bbox_shift))
    bbox = torch.tensor([x_min, y_min, x_max, y_max]).float()
    return torch.from_numpy(img_c3).float(), torch.from_numpy(gt[None, :, :]), bbox


class ModalityDataset(Dataset):
    def __init__(self,
                 dataset_fp: str,
                 image_size: int = 1024,
                 bbox_shift: int = 20,
                 sample: int = None,  # None for all
                 shuffle: bool = True,
                 buffer_len: int = None,
                 augment: bool = True,
                 logger: Any = None,
                 ):
        self.image_size = image_size
        self.bbox_shift = bbox_shift
        self.augment = augment
        cache_fp = os.path.join(dataset_fp, 'cache.pkl')
        if os.path.exists(cache_fp):
            with open(cache_fp, 'rb') as cachefile:
                self.clips, self.sorts = pickle.load(cachefile)
        else:
            self.clips = glob(os.path.join(dataset_fp, '**', '*.npz'), recursive=True)
            self.sorts = [0]
            for clip in tqdm(self.clips):
                gt = np.load(clip)['gts']
                self.sorts.append((1 if len(gt.shape) == 2 else gt.shape[0]) + self.sorts[-1])
            self.sorts = self.sorts[1:]
            with open(cache_fp, 'wb') as cachefile:
                pickle.dump((self.clips, self.sorts), cachefile)
        self.num_sources = self.sorts[-1]
        self.num_samples = self.num_sources if sample is None else sample
        assert self.num_sources >= self.num_samples
        if logger is not None:
            logger.info(f'Modality {os.path.basename(dataset_fp)} has {self.num_sources} 2D clips.')
        self.is2D = self.num_sources == len(self.clips)
        self.name = os.path.basename(dataset_fp)
        self.pick = []
        self.use_buffer = None
        self.buffer_len = 1 if self.is2D else buffer_len
        self.buffer = OrderedDict()
        if shuffle:
            self.shuffle()
        else:
            assert buffer_len is not None
            self.ordered()

    def __getitem__(self, item):
        target, layer = self.pick[item]
        if not self.use_buffer:
            npz = np.load(self.clips[target])
            img, gt = npz['imgs'], npz['gts']
        elif target in self.buffer:  # use_buffer
            img, gt = self.buffer[target]
        else:
            npz = np.load(self.clips[target])
            img, gt = npz['imgs'], npz['gts']
            if len(self.buffer) >= self.buffer_len:
                _ = self.buffer.popitem(last=False)  # FIFO
            self.buffer[target] = (img, gt)
        if self.is2D:
            if len(img.shape) < 3:
                img_3c = np.repeat(img[:, :, None], repeats=3, axis=-1)
            else:
                img_3c = img
        else:
            img = img[layer, :, :]
            gt = gt[layer, :, :]
            img_3c = np.repeat(img[:, :, None], repeats=3, axis=-1)
        img_c3, gt, bbox = preprocess(img_3c, gt, self.image_size, self.bbox_shift, self.augment)
        return img_c3, gt, bbox

    def __len__(self):
        return self.num_samples

    def shuffle(self):
        self.use_buffer = False
        # it contains a total of self.sorts[i] 2D clips since clip[0] to clip[i]
        clip_choices = random.sample(range(self.num_sources), self.num_samples)
        if self.is2D:
            self.pick = [(idx, 0) for idx in clip_choices]
        else:
            self.pick = []
            for idx in clip_choices:
                target = bsearch(self.sorts, idx + 1)
                layer = self.sorts[target] - idx - 1
                self.pick.append((target, layer))

    def ordered(self):
        self.use_buffer = True
        clip_choices = list(range(self.num_sources))
        if self.is2D:
            self.pick = [(idx, 0) for idx in clip_choices]
        else:
            self.pick = []
            for idx in clip_choices:
                target = bsearch(self.sorts, idx + 1)
                layer = self.sorts[target] - idx - 1
                self.pick.append((target, layer))

    def clean_buffer(self):
        self.buffer.clear()

    def modify_sample(self, sample):
        if sample is None:
            self.num_samples = self.num_sources
        else:
            assert self.num_sources >= sample
            self.num_samples = sample


class GatherModalityDataset(Dataset):
    def __init__(self,
                 dataset_fp: str | Iterable[str],
                 image_size: int = 1024,
                 bbox_shift: int = 20,
                 mod_sample: int | Iterable[int] | None = 1000,
                 shuffle: bool = True,
                 buffer_len: int = None,
                 augment: bool = True,
                 logger: Any = None
                 ):
        if isinstance(dataset_fp, str):
            dataset_fp = [os.path.join(dataset_fp, fn) for fn in os.listdir(dataset_fp)]
        dataset_fp = [fp for fp in dataset_fp if os.path.isdir(fp)]
        if isinstance(mod_sample, int) or mod_sample is None:
            mod_sample = [mod_sample] * len(dataset_fp)
        assert len(dataset_fp) == len(mod_sample)
        self.num_mods = len(mod_sample)
        self.modalities = []
        for fp, s in zip(dataset_fp, mod_sample):
            mod = ModalityDataset(
                dataset_fp=fp,
                image_size=image_size,
                bbox_shift=bbox_shift,
                sample=s,
                shuffle=shuffle,
                buffer_len=buffer_len,
                augment=augment,
                logger=logger,
            )
            self.modalities.append(mod)
        self.sorts = [0]
        for mod in self.modalities:
            self.sorts.append(self.sorts[-1] + len(mod))
        self.sorts = self.sorts[1:]
        self.names = [mod.name for mod in self.modalities]
        self.sample_per_modality = mod_sample
        if logger:
            logger.info(f'{len(self.names)} modalities gathered.')

    def __getitem__(self, item):
        i = bsearch(self.sorts, item + 1)
        j = self.sorts[i] - item - 1
        return self.modalities[i][j]

    def __len__(self):
        return self.sorts[-1]

    def shuffle(self):
        for i in range(self.num_mods):
            self.modalities[i].shuffle()

    def modify_mod_sample(self, mod_sample: int | list | tuple):
        if isinstance(mod_sample, int) or mod_sample is None:
            mod_sample = [mod_sample] * self.num_mods
        assert len(mod_sample) == self.num_mods
        self.sorts = [0]
        for i in range(self.num_mods):
            self.modalities[i].modify_sample(mod_sample[i])
            self.sorts.append(self.sorts[-1] + len(self.modalities[i]))
        self.sorts = self.sorts[1:]

    def mod_names(self):
        return self.names

    def mod_samples(self):
        return [len(mod) for mod in self.modalities]


if __name__ == '__main__':
    from sdk import create_logger
    log = create_logger('brevitas.test.log')
    GatherModalityDataset(r"D:\GitHub\dataset\MedSAMLaptop\train", logger=log)
