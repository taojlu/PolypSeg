import os
import pdb

from PIL import Image
import torch.utils.data as data
import cv2
import numpy as np
from PIL import ImageEnhance
import matplotlib.pyplot as plt




# 计算每一个连通区域的像素数
def cal_component(mask):
    num, labels = cv2.connectedComponents(mask)

    labels_dict = {i: [] for i in range(0, num)}
    height, width = mask.shape
    for h in range(height):
        for w in range(width):
            if labels[h][w] in labels_dict:
                labels_dict[labels[h][w]].append([h, w])
    # print(labels_dict.keys())
    components = []
    for i in range(1, len(labels_dict.keys())):
        num_pixel = labels_dict[i]
        components.append(len(num_pixel))
    return components


# 对肿瘤区域进行轮廓填充后处理
def contour_fill(mask):
    # 寻找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 创建空白图像
    filled_image = np.zeros_like(mask)
    # 填充轮廓
    cv2.drawContours(filled_image, contours, -1, 1, thickness=cv2.FILLED)
    # 光滑处理（这里使用闭运算来平滑图像）
    filled_image = cv2.morphologyEx(filled_image, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), np.uint8))
    return filled_image
def image_and_mask_plot(raw, mask, idx):
    fig, ax = plt.subplots(1, 2, figsize=(6, 6))
    ax[0].imshow(raw)
    ax[0].set_title("raw image_{}".format(idx))
    ax[1].imshow(mask)
    ax[1].set_title("mask_{}".format(idx))
    # fig_title = "patient_20_" + str(idx) + ".png"
    # fig.savefig(fig_title, bbox_inches="tight")
    # plt.title("get small object")
    plt.show()

def mask_ratio_plot(mask, idx, com_ratio):
    ratio, num = com_ratio[0], com_ratio[1]
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(mask)
    ax.set_title("mask_{}_{}".format(ratio, num))
    fig_title = "/sda1/wangtao/DataSets/PolypSeg/label_ratio/" + str(idx) + ".png"
    fig.savefig(fig_title, bbox_inches="tight")
    # plt.title("get small object")
    plt.show()

class PolypSmallRatio(data.Dataset):
    def __init__(self, gt_root):
        # get filenames
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]

        self.gts = sorted(self.gts)

        # get size of dataset
        self.size = len(self.gts)

    def __getitem__(self, index):
        # read imgs/gts/grads/depths
        gt = self.binary_loader(self.gts[index])

        gt_npy = np.array(gt)
        h, w = gt_npy.shape[0], gt_npy.shape[1]
        gt_npy[gt_npy > 0] = 1
        # img_plot(gt_npy)
        # 统计方式一
        label_num = np.count_nonzero(gt_npy == 1)
        polyp_ratio = label_num / (h * w)

        # 统计方式二
        # num, labels = cv2.connectedComponents(gt_npy)
        #
        # labels_dict = {i: [] for i in range(0, num)}
        # height, width = gt_npy.shape
        # for h in range(height):
        #     for w in range(width):
        #         if labels[h][w] in labels_dict:
        #             labels_dict[labels[h][w]].append([h, w])
        # # print(labels_dict.keys())
        # components = []
        # for i in range(1, len(labels_dict.keys())):
        #     num_pixel = labels_dict[i]
        #     components.append(len(num_pixel))

        fill_mask = contour_fill(gt_npy)

        # img_plot(fill_mask)
        coms_1 = cal_component(gt_npy)
        coms_2 = cal_component(fill_mask)
        # print(coms_2)
        # if len(coms_1) >= 25:
        #     print(coms_1)
        #     print(coms_2)
        #     image_and_mask_plot(gt_npy, fill_mask, index)
        #     print('======')
        # 将小于10个像素的轮廓区域删除
        coms = [i for i in coms_2 if i > 10]
        coms_r1 = [r/(h*w) for r in coms]
        coms_r2= [round(i, 3) for i in coms_r1]
        nums = [i for i in coms]
        for i,j in zip(coms_r2, nums):
            coms_r_num = [i, j]
        # mask_ratio_plot(fill_mask, index, coms_r_num)
        # if len(coms_ration) > 4:
        #     print(coms_ration)
        #     print(coms)
        #     img_plot(gt_npy)
        #     img_plot(fill_mask)

        return gt_npy, polyp_ratio, label_num, coms_r_num # , componet_pixels

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


def hist_plot(data):
    a = 0.1
    bins = int((max(data) - min(data)) / a)
    plt.hist(data, bins=bins)
    # plt.xticks(list(range(min(data), max(data)))[::2])
    plt.show()


if __name__ == '__main__':

    train_gt_dir = '/root/nfs/gaobin/wt/Datasets/Polyp/TrainDataset/mask/'
    polyp_info = PolypSmallRatio(train_gt_dir)
    statics_num = []
    pixel_count = []
    for step, info in enumerate(polyp_info):
        gt, polyp_ratio, count, coms_ration = info
        # pdb.set_trace()//
        statics_num.append(polyp_ratio)
        pixel_count.append(count)

        # img_plot(gt_small*255)
    print(statics_num)
    print(len(statics_num))
    hist_plot(statics_num)
    small = []
    for i in statics_num:
        if i <= 0.1:
            small.append(i)

    print(len(small) / len(statics_num))
