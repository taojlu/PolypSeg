from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
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


def display_images(original_mask, edge, internal_wide_edge):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(original_mask, cmap='gray')
    axs[0].set_title('Original Mask')
    axs[1].imshow(edge, cmap='gray')
    axs[1].set_title('Edge Image')
    axs[2].imshow(internal_wide_edge, cmap='gray')
    axs[2].set_title('Internal Wide Edge Image')
    plt.show()


# 使用上传的图像
image_path = '/root/nfs/gaobin/wt/Datasets/Polyp/TestDataset/CVC-300/masks/170.png'
original_mask = Image.open(image_path).convert('L')
edge = get_edge(original_mask)
internal_wide_edge = get_internal_wide_edge(original_mask, 20)

# 显示原始掩膜、边缘图像和内部宽边缘图像
display_images(np.array(original_mask), edge, internal_wide_edge)
