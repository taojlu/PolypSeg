import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 加载图像
base_dir = '/root/nfs/gaobin/wt/Datasets/Polyp/Results/RESULT_MAP/MaCbamFormer'
img_path = os.path.join(base_dir,'2024-08-10_18-24-59and124_net_0/ETIS-LaribPolypDB/16.png')
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# 应用二值化
_, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# 形态学闭操作以封闭间隙
kernel = np.ones((50, 50), np.uint8)  # 核的大小根据需要调整
closed_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

# 寻找轮廓
contours, _ = cv2.findContours(closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 填充轮廓
filled_img = np.zeros_like(img)
cv2.drawContours(filled_img, contours, -1, (255), thickness=cv2.FILLED)

# 计算连通组件
num_labels, labels = cv2.connectedComponents(filled_img)

# 判断连通区域的数量，不包括背景
# if num_labels - 1 >= 10:  # 减1是因为第一个标签是背景
#     filled_img = np.zeros_like(img)

# 显示结果
plt.figure(figsize=(15, 7))
plt.subplot(131)
plt.title('Original Image')
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(132)
plt.title('Closed Image')
plt.imshow(closed_img, cmap='gray')
plt.axis('off')

plt.subplot(133)
plt.title('Filled Image')
plt.imshow(filled_img, cmap='gray')
plt.axis('off')

plt.show()
