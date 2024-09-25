import os
import pdb

from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
from plot.image_plot import img_mask_plot
import cv2


def get_edge(mask):
    mask = np.array(mask)
    lesion_coordinate = np.where(mask > 0)

    mask_edge = mask.copy()
    for i, j in zip(lesion_coordinate[0], lesion_coordinate[1]):
        if 1 <= i < mask.shape[0] - 1 and 1 <= j < mask.shape[1] - 1:
            if mask[i - 1, j] > 0 and mask[i, j - 1] > 0 and mask[i, j + 1] > 0 and mask[i + 1, j] > 0:
                mask_edge[i, j] = 0

    return mask_edge

def erode_mask(mask, erosion_size):
    kernel = np.ones((erosion_size, erosion_size), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    return eroded_mask


def get_internal_wide_edge(original_mask, erosion_size):
    original_mask = np.array(original_mask)
    eroded_mask = erode_mask(original_mask, erosion_size)

    # 得到内部边缘，原始肿瘤区域减去腐蚀后的肿瘤区域
    internal_wide_edge = original_mask - eroded_mask

    return internal_wide_edge
class PolypEdgeDataset(data.Dataset):
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

        # gt = get_edge(gt)
        gt = get_internal_wide_edge(gt, 100)   # 调整边缘像素宽度的参数
        gt = Image.fromarray(gt)

        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        if self.img_transform is not None:
            image = self.img_transform(image)

        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)
        return image, gt

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


def get_edge_erode_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True,
               augmentation=True):
    dataset = PolypEdgeDataset(image_root, gt_root, trainsize, augmentation)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_edge_erode_dataset:
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

        # gt = get_edge(gt)
        gt = get_internal_wide_edge(gt, 100)
        gt = Image.fromarray(gt)

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

    train_path = '/root/nfs/gaobin/wt/Datasets/Polyp/TrainDataset'
    test_path = '/root/nfs/gaobin/wt/Datasets/Polyp/TestDataset/CVC-300'
    batchsize = 1

    # =========== 训练集数据验证 ==================
    # trainsize = 352
    train_image_root = '{}/image/'.format(train_path)
    train_gt_root = '{}/mask/'.format(train_path)

    train_loader = get_edge_erode_loader(train_image_root, train_gt_root, batchsize=1,
                              trainsize=trainsize, num_workers=12)
    # #
    for step, data_pack in enumerate(train_loader):
        images, mask = data_pack
        # gt_edge = gt_edge.numpy().squeeze()
        images = images.numpy().squeeze()
        mask = mask.numpy().squeeze()
        # pdb.set_trace()
        images = np.transpose(images, (1, 2, 0))
        # img_mask_plot(images, mask)
    #     print(np.unique(mask))
    #
    # =========== 测试集数据验证 ==================
    # test_image_root = '{}/images/'.format(test_path)
    # test_gt_root = '{}/masks/'.format(test_path)
    # test_loader = test_edge_dataset(test_image_root, test_gt_root, 352)
    # num1 = len(os.listdir(test_gt_root))
    # for i in range(num1):
    #     # pdb.set_trace()
    #     image, gt, name = test_loader.load_data()
    #     gt = np.asarray(gt, np.float32)
    #     gt /= (gt.max() + 1e-8)
    #     image = image.numpy().squeeze()
    #     image = np.transpose(image, (1, 2, 0))
    #     # img_mask_plot(image, gt)
