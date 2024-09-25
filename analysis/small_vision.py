
import os
import cv2
import numpy as np

base_dir = '/root/nfs/gaobin/wt/Datasets/Polyp/TestDataset'
image_path = os.path.join(base_dir, 'ETIS-LaribPolypDB/masks/186.png')


image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

print(np.count_nonzero(image))

print(image.shape)

rate = np.count_nonzero(image) / (image.shape[0] * image.shape[1])
print(rate)

