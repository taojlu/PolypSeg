import os
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

def img_plot(image, mask):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(image)
    ax[0].set_title("Raw Image")
    ax[0].set_axis_off()  # 移除坐标轴

    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title("Mask")
    ax[1].set_axis_off()  # 移除坐标轴

    plt.subplots_adjust(wspace=0.05, hspace=0)  # 调整子图间距
    plt.show()


# img_plot(image_rgb, mask)


def img_plot_fitted(image):
    height, width, _ = image.shape
    dpi = 600  # 屏幕每英寸点数，用于将像素转换为英寸
    fig_width = width / dpi  # 宽度（英寸）
    fig_height = height / dpi  # 高度（英寸）

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.imshow(image)
    ax.axis('off')

    # 无边距填充整个画布
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    plt.show()

    fig.savefig('/root/nfs/gaobin/wt/Paper_Plot/Polyp/fig3_3.png', bbox_inches='tight', pad_inches=0.01)


def mask_plot_fitted(image):
    height, width = image.shape
    dpi = 600  # 屏幕每英寸点数，用于将像素转换为英寸
    fig_width = width / dpi  # 宽度（英寸）
    fig_height = height / dpi  # 高度（英寸）

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.imshow(image, cmap='gray')  # 确保使用灰度颜色映射
    ax.axis('off')

    # 无边距填充整个画布
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    plt.show()

    # 保存图像
    fig.savefig('/root/nfs/gaobin/wt/Paper_Plot/Polyp/fig3_2.png', bbox_inches='tight', pad_inches=0.01)


def create_tiled_image(image, tiles_per_row, tiles_per_col, margin):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # 读取图像尺寸
    height, width = image.shape[:2]

    # 计算子图的高度和宽度
    sub_height = height // tiles_per_row
    sub_width = width // tiles_per_col

    # 计算新画布的尺寸
    canvas_height = height + (tiles_per_row - 1) * margin
    canvas_width = width + (tiles_per_col - 1) * margin

    # 创建一个足够大的画布，以容纳所有子图和间隔
    new_image = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Split and rearrange the image
    for i in range(tiles_per_row):
        for j in range(tiles_per_col):
            # Calculate the start point for current sub-image in the original image
            start_row = i * sub_height
            start_col = j * sub_width

            # Calculate the start point for current sub-image in the new canvas
            target_row = i * (sub_height + margin)
            target_col = j * (sub_width + margin)

            # Extract sub-image
            sub_image = image[start_row:start_row + sub_height, start_col:start_col + sub_width]

            # Place the sub-image in the corresponding position in the new canvas
            new_image[target_row:target_row + sub_height, target_col:target_col + sub_width] = sub_image

    return new_image


if __name__ == '__main__':
    data_type = 'test'
    data_name = 'ETIS-LaribPolypDB'

    if data_type == 'train':
        image_dir = '/root/nfs/gaobin/wt/Datasets/Polyp/TrainDataset/image'
        mask_dir = '/root/nfs/gaobin/wt/Datasets/Polyp/TrainDataset/mask'
    elif data_type == 'test':
        _data_name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
        image_dir = '/root/nfs/gaobin/wt/Datasets/Polyp/TestDataset/{}/images'.format(data_name)
        mask_dir = '/root/nfs/gaobin/wt/Datasets/Polyp/TestDataset/{}/masks'.format(data_name)

    image_id = 101

    img_path = os.path.join(image_dir, str(image_id) + '.png')
    mask_path = os.path.join(mask_dir, str(image_id) + '.png')

    # read image and mask
    image = cv.imread(img_path, cv.IMREAD_UNCHANGED)
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

    # 显示图像
    img_plot_fitted(image_rgb)
    # mask_plot_fitted(mask)
    tiles_per_row = 4  # 每行块数
    tiles_per_col = 4  # 每列块数

    # 创建图像阵列
    tiled_image = create_tiled_image(image, tiles_per_row, tiles_per_col, 20)
    # img_plot_fitted(tiled_image)