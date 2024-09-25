import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import cv2
from plot.image_plot import img_mask_plot
import pdb

def pixel_padding(mask):
    # 连通区域标记
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)

    # 对每个连通区域进行处理
    for label in range(1, num_labels):  # 跳过背景标签0
        # 获取当前连通区域的像素值
        tumor_pixels = np.where(labels == label, mask, 0)

        # 获取当前连通区域的边界像素
        tumor_boundary = cv2.Canny((tumor_pixels > 0).astype(np.uint8), 0, 1)

        # 找到边界像素所在的位置
        boundary_y, boundary_x = np.nonzero(tumor_boundary)

        # 将边界像素的像素值填充到当前连通区域中
        for y, x in zip(boundary_y, boundary_x):
            mask[y, x] = tumor_pixels[y, x]
    return mask


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

    # print(labels_dict.keys())
    # 定义阈值
    threshold = 50

    # 使用字典推导式和条件判断筛选出值数量大于阈值的键值对，并重新构建字典
    labels_dict = {key: value for key, value in labels_dict.items() if len(value) >= threshold}

    # 设置一个像素阈值，删除小的组件
    pixel_threshold = 250000
    for label, pixel_list in labels_dict.items():
        # print(len(pixel_list))
        if len(pixel_list) > pixel_threshold:
            for coord in pixel_list:
                h, w = coord
                mask_small[h, w] = 0

    return mask_small


class PolypSmallDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """

    def __init__(self, image_root, gt_root, trainsize, augmentations):
        self.trainsize = trainsize
        self.augmentations = augmentations
        print(self.augmentations)
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        if self.augmentations == 'True':
            print('Using RandomRotation, RandomFlip')
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])

        else:
            print('no augmentation')
            self.img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

            self.gt_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])

    def __getitem__(self, index):

        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        gt_small = get_small(gt)
        gt_small = Image.fromarray(gt_small)
        # print(componet_pixels)

        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        if self.img_transform is not None:
            image = self.img_transform(image)

        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        if self.gt_transform is not None:
            gt_small = self.gt_transform(gt_small)
        return image, gt_small

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

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_small_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True,
                     augmentation=True):
    dataset = PolypSmallDataset(image_root, gt_root, trainsize, augmentation)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_small_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])

        gt_small = get_small(gt)
        gt_small = Image.fromarray(gt_small)


        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt_small, name

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
    test_path = '/root/nfs/gaobin/wt/Datasets/Polyp/TestDataset/ETIS-LaribPolypDB'
    batchsize = 1

    # =========== 训练集数据验证 ==================
    # trainsize = 352
    # train_image_root = '{}/image/'.format(train_path)
    # train_gt_root = '{}/mask/'.format(train_path)
    #
    # train_loader = get_small_loader(image_root, gt_root, batchsize=1,
    #                                 trainsize=trainsize, num_workers=12)
    #
    # for step, data_pack in enumerate(train_loader):
    #     images, mask = data_pack
    #     # gt_edge = gt_edge.numpy().squeeze()
    #     images = images.numpy().squeeze()
    #     mask = mask.numpy().squeeze()
    #     # pdb.set_trace()
    #     images = np.transpose(images, (1, 2, 0))
    #     count = np.count_nonzero(mask > 0)
    #
    #     print("大于0的元素个数：", count)
    #     img_mask_plot(images, mask)
        # print(np.unique(mask))
    # =========== 测试集数据验证 ==================
    test_image_root = '{}/images/'.format(test_path)
    test_gt_root = '{}/masks/'.format(test_path)
    test_loader = test_small_dataset(test_image_root, test_gt_root, 352)
    num1 = len(os.listdir(test_gt_root))
    for i in range(num1):
        # pdb.set_trace()
        image, gt, name = test_loader.load_data()
        if name == '9.png':
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.numpy().squeeze()
            image = np.transpose(image, (1, 2, 0))
            img_mask_plot(image, gt)
