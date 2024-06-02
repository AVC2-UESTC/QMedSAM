import glob
import os.path
import random
from os.path import join, dirname
import pickle
import numpy as np



def npzto2Dval(img, gts, bbox_shift):
    # img_name = os.path.basename(self.gt_path_files[index])
    img_1024 = img  # (1024, 1024, 3)
    # convert the shape to (3, H, W)
    # img_1024 = np.transpose(img_1024, (2, 0, 1))

    gt = gts  # multiple labels [0, 1,4,5...], (256,256)
    # assert img_name == os.path.basename(self.gt_path_files[index])
    label_ids = np.unique(gt)[1:]
    gt2D = np.uint8(
        gt == random.choice(label_ids.tolist())
    )  # only one label, (256, 256)
    # gt2D = np.uint8(gt > 0)
    assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"
    y_indices, x_indices = np.where(gt2D > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = gt2D.shape
    x_min = max(0, x_min - random.randint(0, bbox_shift))
    x_max = min(W-1, x_max + random.randint(0, bbox_shift))
    y_min = max(0, y_min - random.randint(0, bbox_shift))
    y_max = min(H-1, y_max + random.randint(0, bbox_shift))
    bboxes = np.array([[x_min, y_min, x_max, y_max]])
    return (
        img_1024,
        # torch.tensor(gt2D[None, :, :]).long(),
        bboxes,
        gt2D
    )

def npzto3Dval(img, gts, spacing, bbox_shift):
    # img_name = os.path.basename(self.gt_path_files[index])
    img_1024 = img  # (1024, 1024, 3)
    # convert the shape to (3, H, W)
    # img_1024 = np.transpose(img_1024, (1,2,0))
    gt = gts  # multiple labels [0, 1,4,5...], (256,256)
    # assert img_name == os.path.basename(self.gt_path_files[index])
    label_ids = np.unique(gt)[1:]
    label_id=random.choice(label_ids.tolist())
    gt3D = np.uint8(
        gt == label_id
    )
    assert np.max(gt3D) == 1 and np.min(gt3D) == 0.0, "ground truth should be 0, 1"
    z_indices, y_indices, x_indices = np.where(gt3D > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    z_min, z_max = np.min(z_indices), np.max(z_indices)
# add perturbation to bounding box coordinates
    C, H, W = gt3D.shape
    x_min = max(0, x_min - random.randint(0, bbox_shift))
    x_max = min(W-1, x_max + random.randint(0, bbox_shift))
    y_min = max(0, y_min - random.randint(0, bbox_shift))
    y_max = min(H-1, y_max + random.randint(0, bbox_shift))
    # z_min = max(0, z_min - random.randint(0, bbox_shift))
    # z_max = min(C, z_max + random.randint(0, bbox_shift))
    bboxes=np.array([[x_min, y_min,z_min, x_max, y_max,z_max]])

    return (
        img_1024,
        # torch.tensor(gt3D[None, :, :]).long(),
        bboxes,
        spacing,
        gt3D
    )

dataset_fp=r'/datadisk/luhaisheng/dataset/MedSAMLaptop/split_set/eval'
target_fp=r'/datadisk/luhaisheng/dataset/MedSAMLaptop/split_set/eval_verification'
if isinstance(dataset_fp, str):
    mod_names = [fn for fn in os.listdir(dataset_fp)]
mod_names = [fp for fp in mod_names if os.path.isdir(os.path.join(dataset_fp, fp))]
for mod_name in mod_names:
    count_id=1
    file = open(join(dataset_fp, mod_name, "cache.pkl"), 'rb')  # 以二进制读模式（rb）打开pkl文件
    data_cache = pickle.load(file)  # 读取存储的pickle文件
    file.close()
    tar_fps=data_cache[0]
    for tar_fp in tar_fps:
        npz = np.load(tar_fp)
        imgs=npz["imgs"]
        gts=npz["gts"]
        if len(gts.shape) > 2:  ## 3D image
            if not imgs.shape[1]==imgs.shape[2]:
                print(imgs.shape)
            imgs,box ,spacing,gts = npzto3Dval(imgs,gts,npz["spacing"],10)
            # print(tar_fp.replace('eval', 'eval_verification'))
            fp=join(target_fp, "imgs")
            if not os.path.exists(fp):
                os.makedirs((fp))
            np.savez(join(fp,"3DBox_" + mod_name +"_" +"{:0>4d}".format(count_id)), imgs=imgs, boxes=box, spacing=spacing)
            fp=join(target_fp, "gts")
            if not os.path.exists(fp):
                os.makedirs((fp))
            np.savez(join(fp,"3DBox_" + mod_name +"_" +"{:0>4d}".format(count_id)), gts =gts)
            count_id=count_id+1
        else:
            imgs,box,gts=npzto2Dval(imgs, gts, 10)
            fp=join(target_fp, "imgs")
            if not os.path.exists(fp):
                os.makedirs(fp)
            np.savez(join(fp,"2DBox_" + mod_name +"_" +"{:0>4d}".format(count_id)), imgs=imgs, boxes=box)
            fp=join(target_fp, "gts")
            if not os.path.exists(fp):
                os.makedirs(fp)
            np.savez(join(fp, "2DBox_" + mod_name + "_" + "{:0>4d}".format(count_id)), gts =gts)
            count_id = count_id + 1