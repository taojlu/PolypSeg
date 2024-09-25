import os
import matplotlib.pyplot as plt
import cv2 as cv

data_type = 'train'
data_name = 'CVC-300'

if data_type == 'train':
    image_dir = '/home/wangtao/DataSet/Polyp/TrainDataset/image'
    mask_dir = '/home/wangtao/DataSet/Polyp/TrainDataset/mask'
elif data_type == 'test':
    _data_name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
    image_dir = '/home/wangtao/DataSet/Polyp/TestDataset/{}/images'.format(data_name)
    mask_dir = '/home/wangtao/DataSet/Polyp/TestDataset/{}/masks'.format(data_name)

image_id = 101

img_path = os.path.join(image_dir, str(image_id) + '.png')
mask_path = os.path.join(mask_dir, str(image_id) + '.png')

# read image and mask
image = cv.imread(img_path, cv.IMREAD_UNCHANGED)
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)


def img_plot(image, mask):
    fix, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(image)
    ax[0].set_title("raw image")
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title("raw image")
    plt.show()


img_plot(image_rgb, mask)
