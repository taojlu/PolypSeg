import os
import pdb

from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def draw_histogram(data, bins=50, figsize=(10, 6), title='直方图', xlabel='值的范围', ylabel='频数', grid=True,
                   color='blue', xtick_interval=1000):
    """
    绘制列表数据的直方图，并设置x轴刻度间隔及最大刻度为数据最大值加500。

    参数:
    - data: 列表或数组，包含要绘制的数据。
    - bins: 直方图的柱状分组数，默认为50。
    - figsize: 图形的大小，以英寸为单位，默认为(10, 6)。
    - title: 图形的标题，默认为'直方图'。
    - xlabel: x轴的标签，默认为'值的范围'。
    - ylabel: y轴的标签，默认为'频数'。
    - grid: 是否显示网格，默认为True。
    - color: 柱状图的颜色，默认为'blue'。
    - xtick_interval: x轴刻度之间的间隔，默认为1000。
    """
    plt.figure(figsize=figsize)  # 设置图形的大小
    counts, bins, patches = plt.hist(data, bins=bins, alpha=0.7, color=color)  # 绘制直方图，并获取bins
    plt.title(title)  # 设置图形的标题
    plt.xlabel(xlabel)  # 设置x轴的标签
    plt.ylabel(ylabel)  # 设置y轴的标签
    if grid:
        plt.grid(True)  # 显示网格

    # 计算x轴的最大刻度，为data的最大值加500
    max_data_value = max(data) + 500
    # 设置x轴的刻度，考虑新的最大值
    plt.xticks(range(int(min(bins)), int(max_data_value) + 1, xtick_interval))

    plt.xlim(right=max_data_value)  # 设置x轴的最大限制为新的最大值

    plt.show()


def get_small(mask):
    mask = np.array(mask)
    mask_small = mask.copy()

    # mask = pixel_padding(mask)
    num, labels = cv2.connectedComponents(mask)

    # 创建字典存储每个连通组件的像素坐标
    labels_dict = {i: [] for i in range(1, num + 1)}
    height, width = mask.shape
    for h in range(height):
        for w in range(width):
            label = labels[h][w]
            if label in labels_dict:
                labels_dict[label].append([h, w])
    # 定义阈值
    threshold = 10

    # 使用字典推导式和条件判断筛选出值数量大于阈值的键值对，并重新构建字典
    # labels_dict = {key: value for key, value in labels_dict.items() if len(value) >= threshold}
    return labels_dict


class PolypTrainDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """

    def __init__(self, image_root, gt_root):
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)

    def __getitem__(self, index):
        # print(self.images[index])
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        lesion_component = get_small(gt)
        return image, gt, lesion_component

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def __len__(self):
        return self.size


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


if __name__ == '__main__':

    trainsize = 352

    image_root = '/home/wangtao/DataSet/Polyp/TrainDataset/image/'
    gt_root = '/home/wangtao/DataSet/Polyp/TrainDataset/mask/'
    train_path = '/home/wangtao/DataSet/Polyp/TrainDataset'
    test_path = '/home/wangtao/DataSet/Polyp/TestDataset/CVC-300'

    train_dataset = PolypTrainDataset(image_root, gt_root)

    train_component_nums = []
    # for img, gt, dict_com in tqdm(train_dataset):
    #     # print('====')
    #     # print(len(dict_com.keys()))
    #     for i in range(len(dict_com.keys())):
    #         # print(len(dict_com[i+1]))
    #         num = len(dict_com[i + 1])
    #         train_component_nums.append(num)
    for img, gt, dict_com in tqdm(train_dataset):
        # 如果dict_com的键是连续的，并且你只关心长度，可以直接使用列表推导
        train_component_nums.extend(len(dict_com[key]) for key in dict_com)

    print('息肉的个数：', len(train_component_nums))
    train_component_nums_new = list(filter(lambda x: x != 0, train_component_nums))
    data = np.array(train_component_nums_new)
    np.save('/sda1/wangtao/DataSets/Polyp_Project_2/com.npy', data)
    pdb.set_trace()

