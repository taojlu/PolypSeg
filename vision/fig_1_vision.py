import os
import pdb

import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

data_type = 'train'
data_name = 'ETIS-LaribPolypDB'

train_image_dir = '/root/nfs/gaobin/wt/Datasets/Polyp/TrainDataset/image'
train_mask_dir = '/root/nfs/gaobin/wt/Datasets/Polyp/TrainDataset/mask'

_data_name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
test_image_dir = '/root/nfs/gaobin/wt/Datasets/Polyp/TestDataset/{}/images'.format(data_name)
test_mask_dir = '/root/nfs/gaobin/wt/Datasets/Polyp/TestDataset/{}/masks'.format(data_name)

image_id1 = 'cju43gfosm63n08714rpih8pe'  # 大的且规则息肉 （来自train_dataset)
image_id2 = 'cju320gyvbch60801v2amdi2g'  # 不规则息肉 （来自train_dataset)
image_id3 = 44  # 低对比度息肉 （来自ETIS-LaribPolypDB)
image_id4 = 63  # 小息肉 （来自train_dataset)

img1_path = os.path.join(train_image_dir, str(image_id1) + '.png')
mask1_path = os.path.join(train_mask_dir, str(image_id1) + '.png')

img2_path = os.path.join(train_image_dir, str(image_id2) + '.png')
mask2_path = os.path.join(train_mask_dir, str(image_id2) + '.png')

img3_path = os.path.join(test_image_dir, str(image_id3) + '.png')
mask3_path = os.path.join(test_mask_dir, str(image_id3) + '.png')

img4_path = os.path.join(train_image_dir, str(image_id4) + '.png')
mask4_path = os.path.join(train_mask_dir, str(image_id4) + '.png')


def get_edge(mask):
    mask = np.array(mask)
    lesion_coordinate = np.where(mask > 0)

    mask_edge = mask.copy()
    for i, j in zip(lesion_coordinate[0], lesion_coordinate[1]):
        if 1 <= i < mask.shape[0] - 1 and 1 <= j < mask.shape[1] - 1:
            if mask[i - 1, j] > 0 and mask[i, j - 1] > 0 and mask[i, j + 1] > 0 and mask[i + 1, j] > 0:
                mask_edge[i, j] = 0

    return mask_edge


def extract_tumor_edges(image):
    # 确保图像是二值的（即只包含0和1）
    image = np.where(image > 0, 1, 0)

    # 获取图像的尺寸
    rows, cols = image.shape

    # 创建一个新的数组来存储边缘信息
    edges = np.zeros_like(image)

    # 遍历图像中的每个像素（除了边界）
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # 检查当前像素是否是肿瘤的一部分
            if image[i, j] == 1:
                # 检查当前像素的上下左右是否有非肿瘤
                if (image[i - 1, j] == 0 or image[i + 1, j] == 0 or
                        image[i, j - 1] == 0 or image[i, j + 1] == 0):
                    edges[i, j] = 1

    return edges


def erode_mask(mask, erosion_size):
    kernel = np.ones((erosion_size, erosion_size), np.uint8)
    eroded_mask = cv.erode(mask, kernel, iterations=1)
    return eroded_mask


def get_internal_wide_edge(original_mask, erosion_size):
    original_mask = np.array(original_mask)
    eroded_mask = erode_mask(original_mask, erosion_size)

    # 得到内部边缘，原始肿瘤区域减去腐蚀后的肿瘤区域
    internal_wide_edge = original_mask - eroded_mask

    return internal_wide_edge


# read image and mask
image1 = cv.imread(img1_path, cv.IMREAD_UNCHANGED)
image1_rgb = cv.cvtColor(image1, cv.COLOR_BGR2RGB)
mask1 = cv.imread(mask1_path, cv.IMREAD_GRAYSCALE)
# edge1 = extract_tumor_edges(mask1)

image2 = cv.imread(img2_path, cv.IMREAD_UNCHANGED)
image2_rgb = cv.cvtColor(image2, cv.COLOR_BGR2RGB)
mask2 = cv.imread(mask2_path, cv.IMREAD_GRAYSCALE)
# edge2 = extract_tumor_edges(mask2)

image3 = cv.imread(img3_path, cv.IMREAD_UNCHANGED)
image3_rgb = cv.cvtColor(image3, cv.COLOR_BGR2RGB)
mask3 = cv.imread(mask3_path, cv.IMREAD_GRAYSCALE)
# edge3 = extract_tumor_edges(mask3)

image4 = cv.imread(img4_path, cv.IMREAD_UNCHANGED)
image4_rgb = cv.cvtColor(image4, cv.COLOR_BGR2RGB)
mask4 = cv.imread(mask4_path, cv.IMREAD_GRAYSCALE)
# edge4 = extract_tumor_edges(mask4)

img_list = [image1_rgb, image2_rgb, image3_rgb, image4_rgb,
            mask1, mask2, mask3, mask4]

# pdb.set_trace()
height, width, c_ = image2_rgb.shape


def image_resize(image, img_width, img_height):
    resized_image = cv.resize(image, (img_width, img_height), interpolation=cv.INTER_LINEAR)
    return resized_image


def img_plot(image):
    fix, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(image, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_title("raw image")
    plt.show()


#
# resized_image = cv.resize(edge1, (width, height), interpolation=cv.INTER_LINEAR)
# img_plot(resized_image)


# 使用列表推导式进行图像尺寸调整
resized_images = [image_resize(img, width, height) for img in img_list]

# 如果你需要单独访问每个调整大小后的图像
image1_resize, image2_resize, image3_resize, image4_resize, mask1_resize, \
    mask2_resize, mask3_resize, mask4_resize = resized_images

# edge1_resize = extract_tumor_edges(mask1_resize)
# edge2_resize = extract_tumor_edges(mask2_resize)
# edge3_resize = extract_tumor_edges(mask3_resize)
# edge4_resize = extract_tumor_edges(mask4_resize)


edge1_resize = get_internal_wide_edge(mask1_resize, 10)
edge2_resize = get_internal_wide_edge(mask2_resize, 10)
edge3_resize = get_internal_wide_edge(mask3_resize, 10)
edge4_resize = get_internal_wide_edge(mask4_resize, 10)


small_mask = np.zeros_like(mask1_resize)
small1 = small_mask
small2 = small_mask
small3 = mask3_resize
small4 = mask4_resize


#
img_resize_list = [image1_resize, image2_resize, image3_resize, image4_resize, mask1_resize, \
                   mask2_resize, mask3_resize, mask4_resize, edge1_resize, edge2_resize, \
                   edge3_resize, edge4_resize, small1, small2, small3, small4]

#
# def fig1_plot(img_list):
#     img1, img2, img3, img4, mask1, mask2, mask3, mask4, edge1, edge2, edge3, \
#         edge4, small1, small2, small3, small4 = img_list
#     # fix, ax = plt.subplots(3, 4, figsize=(6, 4))
#     fig, ax = plt.subplots(4, 4, figsize=(6, 3.8), gridspec_kw={'wspace': 0.05, 'hspace': 0.0001})
#
#     ax[0, 0].imshow(img1)
#     ax[0, 0].set_xticks([])
#     ax[0, 0].set_yticks([])
#     ax[1, 0].imshow(mask1, cmap='gray')
#     ax[1, 0].set_xticks([])
#     ax[1, 0].set_yticks([])
#     ax[2, 0].imshow(edge1, cmap='gray')
#     ax[2, 0].set_xticks([])
#     ax[2, 0].set_yticks([])
#     ax[3, 0].imshow(small1, cmap='gray')
#     ax[3, 0].set_xticks([])
#     ax[3, 0].set_yticks([])
#
#     ax[0, 1].imshow(img2)
#     ax[0, 1].set_xticks([])
#     ax[0, 1].set_yticks([])
#     ax[1, 1].imshow(mask2, cmap='gray')
#     ax[1, 1].set_xticks([])
#     ax[1, 1].set_yticks([])
#     ax[2, 1].imshow(edge2, cmap='gray')
#     ax[2, 1].set_xticks([])
#     ax[2, 1].set_yticks([])
#     ax[3, 1].imshow(small2, cmap='gray')
#     ax[3, 1].set_xticks([])
#     ax[3, 1].set_yticks([])
#
#     ax[0, 2].imshow(img3)
#     ax[0, 2].set_xticks([])
#     ax[0, 2].set_yticks([])
#     ax[1, 2].imshow(mask3, cmap='gray')
#     ax[1, 2].set_xticks([])
#     ax[1, 2].set_yticks([])
#     ax[2, 2].imshow(edge3, cmap='gray')
#     ax[2, 2].set_xticks([])
#     ax[2, 2].set_yticks([])
#     ax[3, 2].imshow(small3, cmap='gray')
#     ax[3, 2].set_xticks([])
#     ax[3, 2].set_yticks([])
#
#     ax[0, 3].imshow(img4)
#     ax[0, 3].set_xticks([])
#     ax[0, 3].set_yticks([])
#     ax[1, 3].imshow(mask4, cmap='gray')
#     ax[1, 3].set_xticks([])
#     ax[1, 3].set_yticks([])
#     ax[2, 3].imshow(edge4, cmap='gray')
#     ax[2, 3].set_xticks([])
#     ax[2, 3].set_yticks([])
#     ax[3, 3].imshow(small4, cmap='gray')
#     ax[3, 3].set_xticks([])
#     ax[3, 3].set_yticks([])
#
#     # Adjust margins
#     fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.05, hspace=0.05)
#     # Titles for each column
#     # titles = ['(A)', '(B)', '(C)', '(D)']
#     # for i in range(4):
#     #     ax[2, i].set_xlabel(titles[i])
#     #     ax[2, i].xaxis.set_label_position('bottom')
#     plt.show()
#
#     # Save the figure without extra whitespace
#     fig.savefig('/root/nfs/gaobin/wt/Paper_Plot/Polyp/fig_2.png', bbox_inches='tight', pad_inches=0.01)



def fig1_plot(img_list):
    img1, img2, img3, img4, mask1, mask2, mask3, mask4, edge1, edge2, edge3, \
        edge4, small1, small2, small3, small4 = img_list
    fig, ax = plt.subplots(4, 4, figsize=(9.8, 8), gridspec_kw={'wspace': 0.02, 'hspace': 0.02})

    # 设置图片显示属性
    for i in range(4):
        for j in range(4):
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].set_aspect('equal')  # 确保宽高比一致

    # 显示图片
    ax[0, 0].imshow(img1)
    ax[1, 0].imshow(mask1, cmap='gray')
    ax[2, 0].imshow(edge1, cmap='gray')
    ax[3, 0].imshow(small1, cmap='gray')

    ax[0, 1].imshow(img2)
    ax[1, 1].imshow(mask2, cmap='gray')
    ax[2, 1].imshow(edge2, cmap='gray')
    ax[3, 1].imshow(small2, cmap='gray')

    ax[0, 2].imshow(img3)
    ax[1, 2].imshow(mask3, cmap='gray')
    ax[2, 2].imshow(edge3, cmap='gray')
    ax[3, 2].imshow(small3, cmap='gray')

    ax[0, 3].imshow(img4)
    ax[1, 3].imshow(mask4, cmap='gray')
    ax[2, 3].imshow(edge4, cmap='gray')
    ax[3, 3].imshow(small4, cmap='gray')

    # 调整子图间距
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.05, hspace=0.05)

    plt.show()

    # 保存图像时也要确保间距一致
    fig.savefig('/root/nfs/gaobin/wt/Paper_Plot/Polyp/fig_2.png', bbox_inches='tight', pad_inches=0.01)

# 可视化图像并保存
fig1_plot(img_resize_list)

# for img in img_resize_list:
#
#     img_plot(img)
