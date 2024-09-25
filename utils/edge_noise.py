import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 读取图像
base_dir = '/root/nfs/gaobin/wt/Datasets/Polyp/Results/RESULT_MAP/MaCbamSuperbisionFormer'
image_path = os.path.join(base_dir, '2024-09-01_15-26-54and10_net_0/ETIS-LaribPolypDB/90.png')
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 应用开运算去除小的白点
kernel = np.ones((40,40), np.uint8)  # 核大小可以根据需要调整
opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# 显示原始图像和处理后的图像
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(opened, cmap='gray')
plt.title('Image after Removing Noise')
plt.axis('off')

plt.show()
