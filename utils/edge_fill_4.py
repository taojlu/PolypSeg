import pdb

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from PIL import Image
from plot.image_plot import img_mask_plot

def get_edge_contour(edge_image):
    # 二值化
    _, binary = cv2.threshold(edge_image, 50, 255, cv2.THRESH_BINARY)

    # 形态学闭运算，连接断开的边缘
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("轮廓的数量： ", len(contours))

    # 创建一个空白图像用于绘制结果
    output_image = np.zeros_like(edge_image)
    filled_image = np.zeros_like(edge_image)

    if len(contours) > 0:
        # 寻找具有最大边界矩形面积的轮廓
        max_contour = max(contours, key=lambda cont: cv2.boundingRect(cont)[2] * cv2.boundingRect(cont)[3])

        # 绘制最大轮廓
        cv2.drawContours(output_image, [max_contour], -1, (255), thickness=1)

        # 填充最大轮廓
        cv2.drawContours(filled_image, [max_contour], -1, (255), thickness=cv2.FILLED)

    return output_image, filled_image



def apply_watershed(edge_image):
    # 转换为灰度图
    # gray = cv2.cvtColor(edge_image, cv2.COLOR_BGR2GRAY)
    # 二值化
    ret, thresh = cv2.threshold(edge_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 使用形态学运算去噪声和强化前景区域
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 确定背景区域
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 使用距离变换确定前景区域
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # 找到未知区域（前景和背景的边界）
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 标记标签区域
    ret, markers = cv2.connectedComponents(sure_fg)

    # 为所有的标记加1，保证背景是0而不是1
    markers = markers + 1

    # 未知区域标为0
    markers[unknown == 255] = 0

    # 应用Watershed
    markers = watershed(edge_image, markers)
    # inverted_image = cv2.bitwise_not(markers)
    output_image = np.zeros_like(edge_image)
    output_image[markers > 1] = 255  # Objects will have labels greater than 1
    # Inverting 0 and 255
    inverted_array = np.where(output_image == 0, 255, 0)
    return inverted_array


def img_plot(polyp_image, edge_image, image_counter, image_fill, image_fusion, gt_image):
    # 创建一个2行3列的子图布局，并定义整体图形的大小
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))

    # 显示原始图像
    ax[0, 0].imshow(polyp_image, cmap='gray')
    ax[0, 0].set_title('Original Image')
    ax[0, 0].axis('off')

    # 显示轮廓图像
    ax[0, 1].imshow(edge_image, cmap='gray')
    ax[0, 1].set_title('Edge Image')
    ax[0, 1].axis('off')

    # 显示填充后的图像
    ax[0, 2].imshow(image_counter, cmap='gray')
    ax[0, 2].set_title('Counter Image')
    ax[0, 2].axis('off')

    # 在第二行添加更多的图像或其他可视化
    # 第二行的第一个位置
    ax[1, 0].imshow(image_fill, cmap='gray')
    ax[1, 0].set_title('Filled Image')
    ax[1, 0].axis('off')

    # 第二行的第二个位置
    ax[1, 1].imshow(image_fusion, cmap='gray')
    ax[1, 1].set_title('Fusion Image')
    ax[1, 1].axis('off')

    # # 第二行的第三个位置
    ax[1, 2].imshow(gt_image, cmap='gray')
    ax[1, 2].set_title('GT Image')
    ax[1, 2].axis('off')

    plt.show()


def fusion_1(main_img, edge_img):
    fusion_img = main_img + edge_img
    fusion_img[fusion_img >= 1] = 1
    return fusion_img


def adjust_image_size_and_type(img1, img2):
    # 确保两个图像具有相同的尺寸
    # if img1.shape[:2] != img2.shape[:2]:
    #     # 调整img2的大小以匹配img1
    #     img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)

    # 确保两个图像具有相同的类型
    if img1.dtype != img2.dtype:
        img2 = img2.astype(img1.dtype)

    return img1, img2
def fusion_2(main_img, edge_img):
    # 调整图像尺寸和类型
    # main_img, edge_img = adjust_image_size_and_type(main_img, edge_img)
    # intersection = cv2.bitwise_and(main_img, edge_img)
    intersection = np.logical_and(main_img == 255, edge_img == 255)

    # Convert the result from boolean (True/False) to integer (255/0)
    intersection = np.where(intersection, 255, 0)

    return intersection


if __name__ == '__main__':

    num = 179

    # 加载真实图像掩膜
    gt_base_dir = '/root/nfs/gaobin/wt/Datasets/Polyp/TestDataset'
    gt_img_path = os.path.join(gt_base_dir, 'ETIS-LaribPolypDB/masks/{}.png'.format(num))

    # 加载模型预测的息肉图像
    polyp_base_dir = '/root/nfs/gaobin/wt/Datasets/Polyp/Results/RESULT_MAP/MaCbamSuperbisionFormer'
    polyp_image_path = os.path.join(polyp_base_dir, '2024-08-12_14-31-41and17_net_dice_0/ETIS-LaribPolypDB/{}.png'.format(num))

    # 加载模型预测的边缘图像
    edge_base_dir = '/root/nfs/gaobin/wt/Datasets/Polyp/Results/RESULT_MAP/MaCbamSuperbisionFormer'
    edge_img_path = os.path.join(edge_base_dir, '2024-09-01_15-26-54and10_net_0/ETIS-LaribPolypDB/{}.png'.format(num))

    gt_image = cv2.imread(gt_img_path, cv2.IMREAD_GRAYSCALE)
    polyp_image = cv2.imread(polyp_image_path, cv2.IMREAD_GRAYSCALE)
    edge_image = cv2.imread(edge_img_path, cv2.IMREAD_GRAYSCALE)

    #

    output_image = apply_watershed(edge_image)



    print("Polyp Image Shape:", polyp_image.shape, "Type:", polyp_image.dtype)
    print("Output Image Shape:", output_image.shape, "Type:", output_image.dtype)
    # image_fusion = fusion_1(polyp_image, filled_image)
    image_fusion = fusion_2(polyp_image, output_image)
    polyp_image[polyp_image>0]=255

    fusion_img_save = Image.fromarray(np.uint8(image_fusion * 255))
    print(fusion_img_save)
    # gray_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
    # output_image2 = apply_watershed(gray_image)

    img_mask_plot(polyp_image, output_image)
    img_mask_plot(image_fusion,gt_image)

    #+
    # img_plot(polyp_image, edge_image, output_image, filled_image, image_fusion, gt_image)
    #
    #
    # print(image_fusion)
