import os
import pdb
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import cv2


def img_plot(img1, img2):
    # 显示原始图像和二值化后的图像
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img1, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("Binary Image")
    plt.imshow(img2, cmap='gray')
    plt.show()  # 添加 plt.show() 以显示图像


def img_read(main_img_dir, edge_img_dir, small_img_dir):
    main_img_info = Image.open(main_img_dir).convert('L')
    main_img = np.array(main_img_info) // 255
    main_img = (main_img > 0.5).astype(np.float64)

    edge_img_info = Image.open(edge_img_dir).convert('L')
    edge_img = np.array(edge_img_info) // 255
    edge_img = (edge_img > 0.5).astype(np.float64)

    small_img_info = Image.open(small_img_dir).convert('L')
    small_img = np.array(small_img_info) // 255
    small_img = (small_img > 0.5).astype(np.float64)
    return main_img, edge_img, small_img

# def edge_fill(edge_path):
#     img = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
#
#     # 使用形态学闭操作来封闭轮廓
#     kernel = np.ones((30, 30), np.uint8)  # 使用更大的核封闭所有缺口
#     closed_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
#
#     # 创建一个与原图像同样大小的全零（全黑）图像用于floodFill
#     h, w = closed_img.shape[:2]
#     mask = np.zeros((h + 2, w + 2), np.uint8)
#
#     # floodFill 从图像的内部某点开始填充（通常选择靠近中心的白点）
#     cv2.floodFill(closed_img, mask, (w // 2, h // 2), 255)
#
#     # 反转图像颜色
#     inverted_img = cv2.bitwise_not(closed_img)
#     return inverted_img

def edge_fill(edge_path):
    img = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)

    # 应用二值化
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 形态学闭操作以封闭间隙
    kernel = np.ones((100, 100), np.uint8)  # 核的大小根据需要调整
    closed_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

    # 寻找轮廓
    contours, _ = cv2.findContours(closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 填充轮廓
    filled_img = np.zeros_like(img)
    cv2.drawContours(filled_img, contours, -1, (255), thickness=cv2.FILLED)

    # 计算连通组件
    num_labels, labels = cv2.connectedComponents(filled_img)

    # 判断连通区域的数量，不包括背景
    if num_labels - 1 >= 10:  # 减1是因为第一个标签是背景
        filled_img = np.zeros_like(img)

    return filled_img

def fusion_1(main_img, edge_img, small_img):
    fusion_img = main_img + edge_img + small_img
    fusion_img[fusion_img >= 1] = 1
    return fusion_img

def fusion_2(main_img, edge_img, small_img):

    fusion_img = main_img + edge_img + small_img
    fusion_img[fusion_img >= 1] = 1
    return fusion_img

def fusion_3(main_img, edge_img, small_img):

    fusion_img = main_img + edge_img + small_img
    fusion_img[fusion_img >= 1] = 1
    return fusion_img

def fusion_4(main_img, edge_img, small_img):

    fusion_img = main_img + small_img
    fusion_img[fusion_img >= 1] = 1
    return fusion_img

def fusion_5(main_img, edge_img, small_img):

    fusion_img = main_img + edge_img
    fusion_img[fusion_img >= 1] = 1
    return fusion_img


if __name__ == '__main__':

    base_dir = '/root/nfs/gaobin/wt/Datasets/Polyp/Results/RESULT_MAP/PVT_UNet_NewAtt'
    fusion_save_dir = '/root/nfs/gaobin/wt/Datasets/Polyp/Results/FUSION/fusion_6'

    main_dir = os.path.join(base_dir, '2024-08-04_02-43-52and52_net_dice_0')
    edge_dir = os.path.join(base_dir, '2024-08-06_02-48-17and153_net_0')
    small_dir = os.path.join(base_dir, '2024-08-04_06-14-57and21_net_0')

    name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']

    # name = ['ETIS-LaribPolypDB']
    for ds in name:
        main_data_dir = os.path.join(main_dir, ds)
        edge_data_dir = os.path.join(edge_dir, ds)
        small_data_dir = os.path.join(small_dir, ds)

        image_names = os.listdir(main_data_dir)
        sorted(image_names)
        print(image_names)

        save_path = os.path.join(fusion_save_dir, ds)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for img in image_names:
            print(img)
            main_img_dir = os.path.join(main_data_dir, img)
            edge_img_dir = os.path.join(edge_data_dir, img)
            small_img_dir = os.path.join(small_data_dir, img)
            main_img, edge_img, small_img = img_read(main_img_dir, edge_img_dir, small_img_dir)

            # edge_img_fill= edge_fill(edge_img_dir)
            # img_plot(edge_img, edge_img_fill)
            fusion_img = fusion_5(main_img, edge_img, small_img)
            # img_plot(edge_img_fill, fusion_img)
            fusion_img_save = Image.fromarray(np.uint8(fusion_img*255))

            # 创建保存路径和保存图像
            fusion_img_path = os.path.join(save_path, img.split('.')[0] + '.png')
            fusion_img_save.save(fusion_img_path, 'png')
