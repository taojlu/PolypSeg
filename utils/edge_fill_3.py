import pdb

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 加载图像
base_dir = '/root/nfs/gaobin/wt/Datasets/Polyp/Results/RESULT_MAP/MaCbamSuperbisionFormer'
img_path = os.path.join(base_dir, '2024-09-01_15-26-54and10_net_0/ETIS-LaribPolypDB/177.png')
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
print(np.unique(image))
print(image)


# 二值化
_, binary = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)

# 形态学闭运算，连接断开的边缘
kernel = np.ones((5, 5), np.uint8)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# 查找轮廓
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print("轮廓的数量： ", len(contours))

# 筛选出面积最大的三个轮廓
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

# 创建一个空白图像用于绘制结果
output_image = np.zeros_like(image)

filled_image = np.zeros_like(image)  # 新增一个图像用于展示填充效果

# 处理每一个轮廓
for contour in sorted_contours:
    # 多边形逼近
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    # 绘制逼近后的轮廓
    cv2.drawContours(output_image, [approx], -1, (255), thickness=1)
    # 在新图像上填充逼近后的轮廓
    cv2.drawContours(filled_image, [approx], -1, (255), thickness=-1)

# 使用matplotlib显示结果
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(output_image, cmap='gray')
plt.title('Smoothed Contour Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(filled_image, cmap='gray')
plt.title('Filled Contour Image')
plt.axis('off')

plt.show()