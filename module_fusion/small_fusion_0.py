import pdb

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from PIL import Image
from plot.image_plot import img_mask_plot
import re


def get_contour_num(small_image):
    small_image = small_image.astype(np.uint8)
    # 二值化
    _, binary = cv2.threshold(small_image, 50, 255, cv2.THRESH_BINARY)

    # 形态学闭运算，连接断开的边缘
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return len(contours)


def get_small_contour(small_image):
    # 二值化
    _, binary = cv2.threshold(small_image, 50, 255, cv2.THRESH_BINARY)

    # 形态学闭运算，连接断开的边缘
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("轮廓的数量： ", len(contours))

    # 创建一个空白图像用于绘制结果
    output_image = np.zeros_like(small_image)
    filled_image = np.zeros_like(small_image)

    if len(contours) > 0:
        # 寻找具有最大边界矩形面积的轮廓
        max_contour = max(contours, key=lambda cont: cv2.boundingRect(cont)[2] * cv2.boundingRect(cont)[3])

        # 绘制最大轮廓
        cv2.drawContours(output_image, [max_contour], -1, (255), thickness=1)

        # 填充最大轮廓
        cv2.drawContours(filled_image, [max_contour], -1, (255), thickness=cv2.FILLED)

    return output_image, filled_image


def apply_watershed(small_image):
    # 转换为灰度图
    # gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
    # 二值化
    ret, thresh = cv2.threshold(small_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

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
    markers = watershed(small_image, markers)
    # inverted_image = cv2.bitwise_not(markers)
    output_image = np.zeros_like(small_image)
    output_image[markers > 1] = 255  # Objects will have labels greater than 1
    # Inverting 0 and 255
    inverted_array = np.where(output_image == 0, 255, 0)
    return inverted_array


def img_plot(polyp_image, small_image, image_counter, image_fill, image_fusion, gt_image):
    # 创建一个2行3列的子图布局，并定义整体图形的大小
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))

    # 显示原始图像
    ax[0, 0].imshow(polyp_image, cmap='gray')
    ax[0, 0].set_title('Original Image')
    ax[0, 0].axis('off')

    # 显示轮廓图像
    ax[0, 1].imshow(small_image, cmap='gray')
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


def fusion_1(main_img, small_img):
    fusion_img = main_img + small_img
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


def fusion_2(main_img, small_img):
    # 调整图像尺寸和类型
    # main_img, small_img = adjust_image_size_and_type(main_img, small_img)
    # intersection = cv2.bitwise_and(main_img, small_img)
    intersection = np.logical_and(main_img == 255, small_img == 255)

    # Convert the result from boolean (True/False) to integer (255/0)
    intersection = np.where(intersection, 255, 0)

    return intersection


def dice_score(pred, target, smooth=1e-6):
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    intersection = np.sum(pred_flat * target_flat)

    score = (2. * intersection + smooth) / (np.sum(pred_flat) + np.sum(target_flat) + smooth)
    return score


def calculate_smoothness(contour):
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if area == 0:
        return np.inf  # Return infinity if area is zero to indicate no smoothness score
    smoothness = perimeter ** 2 / (4 * np.pi * area)
    return smoothness


def find_contours(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def find_contours2(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def max_area_contour(input_image, contours):
    max_area = 0
    max_contour = None

    # 创建一个空白图像用于绘制结果
    output_image = np.zeros_like(input_image)
    filled_image = np.zeros_like(input_image)

    # 遍历所有轮廓
    for contour in contours:
        # 计算当前轮廓的面积
        area = cv2.contourArea(contour)

        # 如果当前轮廓面积大于之前的最大面积，则更新最大面积和最大轮廓
        if area > max_area:
            max_area = area
            max_contour = contour
        # 绘制最大轮廓
        cv2.drawContours(output_image, [max_contour], -1, (255), thickness=1)

        # 填充最大轮廓
        cv2.drawContours(filled_image, [max_contour], -1, (255), thickness=cv2.FILLED)
    return filled_image
def calculate_small_coherence(contour):
    # Calculate the curvature for each point and then find the variance
    curvature = []
    pts = contour[:, 0, :]  # Simplify contour array shape
    for i in range(1, len(pts) - 1):
        p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1]
        k = np.abs((p3[1] - p2[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p3[0] - p2[0]))
        curvature.append(k)
    variance = np.var(curvature) if curvature else np.inf
    return variance
if __name__ == '__main__':

    data_name = 'ETIS-LaribPolypDB'
    num = 47

    dataset = os.path.join('/root/nfs/gaobin/wt/Datasets/Polyp/TestDataset', data_name, "masks")
    img_list = os.listdir(dataset)
    # print(img_list)
    # img_list.sort()
    img_list.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    mian_dice_average = []
    small_dice_average = []
    # img_list = ['179.png']
    for num in img_list:
        num = num.split('.')[0]
        print("当前图片的名称：==============", num)
        # 加载真实图像掩膜
        gt_base_dir = '/root/nfs/gaobin/wt/Datasets/Polyp/TestDataset'
        gt_img_path = os.path.join(gt_base_dir, '{}/masks/{}.png'.format(data_name, num))

        # 加载模型预测的息肉图像
        polyp_base_dir = '/root/nfs/gaobin/wt/Datasets/Polyp/Results/RESULT_MAP/MaCbamSuperbisionFormer'
        polyp_image_path = os.path.join(polyp_base_dir,
                                        '2024-08-12_14-31-41and17_net_dice_0/{}/{}.png'.format(data_name, num))

        # 加载模型预测的边缘图像
        small_base_dir = '/root/nfs/gaobin/wt/Datasets/Polyp/Results/RESULT_MAP/MaCbamSuperbisionFormer'
        small_img_path = os.path.join(small_base_dir, '2024-09-05_00-34-02and32_net_0/{}/{}.png'.format(data_name, num))

        gt_image = cv2.imread(gt_img_path, cv2.IMREAD_GRAYSCALE)
        polyp_image = cv2.imread(polyp_image_path, cv2.IMREAD_GRAYSCALE)
        small_image = cv2.imread(small_img_path, cv2.IMREAD_GRAYSCALE)

        polyp_image[polyp_image > 0] = 1
        gt_image[gt_image > 0] = 1
        small_image[small_image>0]=1
        #

        # polyp_contours_num = get_contour_num(polyp_image)
        # small_contour_num = get_contour_num(small_image)
        # print("边缘分割轮廓数量：", polyp_contours_num)

        polyp_contours = find_contours(polyp_image)
        print("主任务息肉分割轮廓数量： ", len(polyp_contours))
        # pdb.set_trace()

        if len(polyp_contours) == 1:
            output_image = apply_watershed(small_image)
            # output_image[output_image == 1] = 1
            output_image = np.uint8(output_image)

            small_contours = find_contours(output_image)
            print("边缘息肉轮廓数量： ", len(small_contours))
            if len(small_contours) == 0:
                small_final = polyp_image
            elif len(small_contours) == 1:

                polyp_smoothness = calculate_smoothness(polyp_contours[0])
                polyp_perimeter = cv2.arcLength(polyp_contours[0], True)

                small_smoothness = calculate_smoothness(small_contours[0])
                small_perimeter = cv2.arcLength(small_contours[0], True)

                poly_coherence = calculate_small_coherence(polyp_contours[0])
                small_coherence = calculate_small_coherence(small_contours[0])

                print("polyp smoothness: ", polyp_smoothness)
                print("small smoothness: ", small_smoothness)
                print("polyp perimeter: ", polyp_perimeter)
                print("small perimeter: ", small_perimeter)
                print("polyp coherence: ", poly_coherence)
                print("small coherence: ", small_coherence)
                if small_coherence > 5000:
                    small_final = polyp_image
                else:
                    if polyp_perimeter < 500:
                        if abs(poly_coherence-small_coherence) / small_coherence < 0.5:
                            small_final = polyp_image
                        else:
                            small_final = small_image
                        # small_final = small_image
                    elif polyp_perimeter > 500:
                        small_final = polyp_image
            elif len(small_contours) > 1:
                small_final = polyp_image
        if len(polyp_contours) > 1:
            output_image = apply_watershed(small_image)
            # output_image[output_image == 1] = 1
            output_image = np.uint8(output_image)

            small_contours = find_contours(output_image)
            print("小息肉轮廓数量： ", len(small_contours))

            small_fill = max_area_contour(output_image, small_contours)
            small_fill[small_fill==255]=1
            small_fill_contour = find_contours(output_image)

            if len(small_contours) == 0:
                small_final = polyp_image
            if len(small_contours) > 0:

                polyp_smoothness = calculate_smoothness(polyp_contours[0])
                polyp_perimeter = cv2.arcLength(polyp_contours[0], True)

                small_smoothness = calculate_smoothness(small_fill_contour[0])
                small_perimeter = cv2.arcLength(small_fill_contour[0], True)

                poly_coherence = calculate_small_coherence(polyp_contours[0])
                small_coherence = calculate_small_coherence(small_fill_contour[0])

                print("polyp smoothness: ", polyp_smoothness)
                print("small smoothness: ", small_smoothness)
                print("polyp perimeter: ", polyp_perimeter)
                print("small perimeter: ", small_perimeter)
                print("polyp coherence: ", poly_coherence)
                print("small coherence: ", small_coherence)

                if polyp_perimeter < 200:
                    if abs(poly_coherence-small_coherence) / small_coherence < 0.5:
                        small_final = polyp_image
                    else:
                        small_final = small_fill
                else:
                    small_final = polyp_image
                # print(np.unique(small_final))

        #         small_final = small_image
        #     else:
        #         small_final = small_image
        #
        # elif len(polyp_contours) > 1:
        #     output_image = apply_watershed(small_image)
        #     output_image[output_image == 255] = 1
        #     output_image = np.uint8(output_image)
        #     small_final = small_image
            #
            # pdb.set_trace()

            # img_mask_plot(small_image, output_image)
            # small_image[small_image > 0] = 1
            # small_final =  small_image


            # count_1 = np.count_nonzero(polyp_image == 1)
            # count_2 = np.count_nonzero(output_image == 1)
            # # pdb.set_trace()
            # rate = count_2 / count_1

            # h, w = output_image.shape
            # tumor_rate = count_2 / (h * w)
            #
            # print("所占的比例",rate)
            # if rate < 0.1 or rate > 0.6:
            #     if tumor_rate < 0.01:
            #         small_final = polyp_image
            #     else:
            #         small_final = polyp_image + output_image
            # elif 0.1 < rate < 0.6:
            #     small_final = polyp_image
            # if rate < 0.1 or rate > 0.5:
            #     small_final = output_image
            # elif 0.1 < rate < 0.5:
            #     small_final = polyp_image
            # small_final = fusion_2(small_final, polyp_image)

            # small_final = fusion_2(output_image, polyp_image)
            # if rate < 0.5:
            #     small_final = polyp_image
            # elif rate > 0.5:
            #     small_final = output_image
            # small_contours = get_contour_num(small_final)
            # print(small_contours)
        # elif small_contours > 1:
        #     output_image = apply_watershed(small_image)
        #     small_contour, small_fill = get_small_contour(small_image)
        #     # small_final = small_fill
        #
        #     # small_fill[small_fill > 0] = 255
        #
        #     count_1 = np.count_nonzero(polyp_image == 1)
        #     count_2 = np.count_nonzero(small_fill == 1)
        #
        #     rate = count_2 / count_1
        #     if rate < 0.8:
        #         small_final = polyp_image
        #     else:
        #         small_final = output_image

        # img_mask_plot(polyp_image, small_image)
        # img_mask_plot(small_final, gt_image)
        small_final[small_final > 0] = 1
        # small_final = small_image + polyp_image
        mian_dice = dice_score(polyp_image, gt_image)
        small_dice = dice_score(small_final, gt_image)
        print("主任务息肉分割： ", mian_dice)
        print("边缘矫正分割： ", small_dice)
        mian_dice_average.append(mian_dice)
        small_dice_average.append(small_dice)

    main_a_dice = sum(mian_dice_average) / len(mian_dice_average)
    small_a_dice = sum(small_dice_average) / len(small_dice_average)

    print("主任务平均Dice: ", round(main_a_dice, 4))
    print("边缘任务平均Dice: ", round(small_a_dice, 4))
    fusion_img_save = Image.fromarray(np.uint8(small_final))
    #
    # # # 创建保存路径和保存图像
    # fusion_img_path = os.path.join(save_path, img.split('.')[0] + '.png')
    # fusion_img_save.save(fusion_img_path, 'png')
