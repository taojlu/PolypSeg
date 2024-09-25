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


def get_contour_num(edge_image):
    # 二值化
    _, binary = cv2.threshold(edge_image, 50, 255, cv2.THRESH_BINARY)

    # 形态学闭运算，连接断开的边缘
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)

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


def new_image(edge_contours, polyp_image, edge_image):
    if edge_contours == 1:
        output_image = apply_watershed(edge_image)
        polyp_image[polyp_image > 0] = 255
        edge_image[edge_image > 0] = 255

        count_1 = np.count_nonzero(polyp_image == 255)
        count_2 = np.count_nonzero(output_image == 255)

        rate = count_2 / count_1
        print(rate)
        if rate < 0.1 or rate > 0.6:
            edge_final = output_image
        elif 0.1 < rate < 0.6:
            edge_final = polyp_image

        # if rate < 0.5:
        #     edge_final = polyp_image
        # elif rate > 0.5:
        #     edge_final = output_image
        # edge_contours = get_contour_num(edge_final)
        # print(edge_contours)
    elif edge_contours > 1:
        output_image = apply_watershed(edge_image)
        edge_contour, edge_fill = get_edge_contour(edge_image)
        # edge_final = edge_fill

        polyp_image[polyp_image > 0] = 255
        # edge_fill[edge_fill > 0] = 255

        count_1 = np.count_nonzero(polyp_image == 255)
        count_2 = np.count_nonzero(edge_fill == 255)

        rate = count_2 / count_1
        if rate < 0.8:
            edge_final = polyp_image
        else:
            edge_final = output_image

    return edge_final

def find_contours(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours
def calculate_smoothness(contour):
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if area == 0:
        return np.inf  # Return infinity if area is zero to indicate no smoothness score
    smoothness = perimeter ** 2 / (4 * np.pi * area)
    return smoothness
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

    base_dir = '/root/nfs/gaobin/wt/Datasets/Polyp/Results/RESULT_MAP/MaCbamSuperbisionFormer'
    fusion_save_dir = '/root/nfs/gaobin/wt/Datasets/Polyp/Results/FUSION/small_fusion'

    main_dir = os.path.join(base_dir, '2024-08-12_14-31-41and17_net_dice_0')
    edge_dir = os.path.join(base_dir, '2024-09-05_00-34-02and32_net_0')

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
            polyp_image_path = os.path.join(main_data_dir, img)
            small_img_path = os.path.join(edge_data_dir, img)

            polyp_image = cv2.imread(polyp_image_path, cv2.IMREAD_GRAYSCALE)
            small_image = cv2.imread(small_img_path, cv2.IMREAD_GRAYSCALE)

            polyp_image[polyp_image > 0] = 1
            small_image[small_image > 0] = 1

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
                            if abs(poly_coherence - small_coherence) / small_coherence < 0.5:
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
                small_fill[small_fill == 255] = 1
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
                        if abs(poly_coherence - small_coherence) / small_coherence < 0.5:
                            small_final = polyp_image
                        else:
                            small_final = small_fill
                    else:
                        small_final = polyp_image

            small_final[small_final > 0] = 1
            small_final[small_final==1]=255


            #
            fusion_img_save = Image.fromarray(np.uint8(small_final))
            #
            # # 创建保存路径和保存图像
            fusion_img_path = os.path.join(save_path, img.split('.')[0] + '.png')
            fusion_img_save.save(fusion_img_path, 'png')
