import os
import pdb
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import cv2
from skimage.segmentation import watershed
from plot.image_plot import img_mask_plot


def img_plot(img1, img2):
    # 显示原始图像和二值化后的图像
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img1, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("Binary Image")
    plt.imshow(img2, cmap='gray')
    plt.show()  # 添加 plt.show() 以显示图像


def img_read(main_img_dir, edge_img_dir):
    main_img_info = Image.open(main_img_dir).convert('L')
    main_img = np.array(main_img_info) // 255
    main_img = (main_img > 0.5).astype(np.float64)

    edge_img_info = Image.open(edge_img_dir).convert('L')
    edge_img = np.array(edge_img_info) // 255
    edge_img = (edge_img > 0.5).astype(np.float64)

    return main_img, edge_img


def apply_watershed(edge_image):
    # 转换为灰度图
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


def fusion_1(main_img, edge_img):
    # 调整图像尺寸和类型
    intersection = np.logical_and(main_img == 255, edge_img == 255)

    # Convert the result from boolean (True/False) to integer (255/0)
    intersection = np.where(intersection, 255, 0)

    return intersection


if __name__ == '__main__':

    base_dir = '/root/nfs/gaobin/wt/Datasets/Polyp/Results/RESULT_MAP/MaCbamSuperbisionFormer'
    fusion_save_dir = '/root/nfs/gaobin/wt/Datasets/Polyp/Results/FUSION/edge_correction_1'

    main_dir = os.path.join(base_dir, '2024-08-12_14-31-41and17_net_dice_0')
    edge_dir = os.path.join(base_dir, '2024-09-01_15-26-54and10_net_0')

    name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']

    # name = ['ETIS-LaribPolypDB']
    for ds in name:
        main_data_dir = os.path.join(main_dir, ds)
        edge_data_dir = os.path.join(edge_dir, ds)

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

            polyp_image = cv2.imread(main_img_dir, cv2.IMREAD_GRAYSCALE)
            edge_image = cv2.imread(edge_img_dir, cv2.IMREAD_GRAYSCALE)

            output_image = apply_watershed(edge_image)
            # img_mask_plot(polyp_image, output_image)

            polyp_image[polyp_image > 0] = 255
            edge_image[edge_image > 0] = 255

            count_1 = np.count_nonzero(polyp_image == 255)
            count_2 = np.count_nonzero(output_image == 255)

            # print(count_1)
            # print(count_2)

            rate = count_2 / count_1
            print(rate)

            if rate < 0.8:
                edge_final = polyp_image
            else:
                edge_final = output_image
            # img_mask_plot(polyp_image, edge_final)

            image_fusion = fusion_1(polyp_image, edge_final)
            # img_mask_plot(edge_image, image_fusion)
            # pdb.set_trace()

            #
            fusion_img_save = Image.fromarray(np.uint8(image_fusion))
            #
            # # 创建保存路径和保存图像
            fusion_img_path = os.path.join(save_path, img.split('.')[0] + '.png')
            fusion_img_save.save(fusion_img_path, 'png')
