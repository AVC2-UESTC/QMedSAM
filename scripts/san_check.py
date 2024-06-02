import os
import random
from os.path import join

import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
file_names=[[r"2DBox_Dermoscopy_0001.npz",r"2DBox_CT_0013.npz"],]
img_path = "MedSAMLaptop/split_set/eval_verification/imgs"
gts_path = "MedSAMLaptop/split_set/eval_verification/gts"
baseline_path = "MedSAMLaptop/split_set/eval_verification/lite_medsam"
result_path = "MedSAMLaptop/split_set/eval_verification/c900.iemd.v2"

def show_mask(mask, ax, random_color=False):
    mask_max=np.max(mask)
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.45])], axis=0)
    else:
        color = np.array([100 / 255 /mask_max, 240 / 255 /mask_max, 20 / 255 /mask_max])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    mask_image=np.concatenate([mask_image,(mask.reshape(h, w, 1)>0)*0.6],axis=2)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0, 0, 0, 0), lw=2))

for file_name in file_names:
    _, axs = plt.subplots(2, 4)
    idx = random.randint(0, 4)

    image = np.load((img_path+'/'+file_name[0]))["imgs"]
    gt = np.load((gts_path+'/'+file_name[0]))["gts"]
    bboxes = np.load((img_path+'/'+file_name[0]))["boxes"]
    result = np.load((result_path+'/'+file_name[0]))["segs"]
    baseline=np.load((result_path+'/'+file_name[0]))["segs"]
    try:
        axs[0,0].imshow(image)
    except:
        break

    axs[0,0].axis('off')
    axs[0,1].imshow(image)
    show_mask(gt, axs[0,1])
    axs[0,1].axis('off')


    axs[0,2].imshow(image)
    show_mask(baseline, axs[0,2])
    axs[0,2].axis('off')
    axs[0,3].imshow(image)
    show_mask(result, axs[0,3])
    axs[0,3].axis('off')

    image = np.load((img_path+'/'+ file_name[1]))["imgs"]
    gt = np.load((gts_path+'/'+ file_name[1]))["gts"]
    bboxes = np.load((img_path+'/'+ file_name[1]))["boxes"]
    result = np.load((result_path+'/'+ file_name[1]))["segs"]
    baseline = np.load((result_path+'/'+ file_name[1]))["segs"]
    try:
        axs[1, 0].imshow(image)
    except:
        break

    axs[1, 0].axis('off')
    axs[1, 0].set_title("(a)",y=-0.30)

    axs[1, 1].imshow(image)
    show_mask(gt, axs[1, 1])
    axs[1, 1].axis('off')
    axs[1, 1].set_title("(b)",y=-0.30)

    axs[1, 2].imshow(image)
    show_mask(baseline, axs[1, 2])
    axs[1, 2].axis('off')
    axs[1, 2].set_title("(c)",y=-0.30)

    axs[1, 3].imshow(image)
    show_mask(result, axs[1, 3])
    axs[1, 3].axis('off')
    axs[1, 3].set_title("(d)",y=-0.30)

    plt.subplots_adjust(wspace=0.01, hspace=-0.5)
    # plt.show()
    plt.savefig(
        join(r"MedSAMLaptop/result_fig", os.path.splitext(file_name[0])[0]+"_and_"+ os.path.splitext(file_name[1])[0]+".png"),
        bbox_inches='tight',
        dpi=300
    )
    plt.close()
