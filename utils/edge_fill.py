# import os.path
#
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# # 加载图像
# base_dir = '/root/nfs/gaobin/wt/Datasets/Polyp/Results/RESULT_MAP/PVT_UNet_NewAtt'
# img_path = os.path.join(base_dir,'2024-08-06_02-48-17and153_net_0/ETIS-LaribPolypDB/119.png')
# img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#
# # 使用形态学闭操作来封闭轮廓
# kernel = np.ones((15,15), np.uint8)  # 使用更大的核封闭所有缺口
# closed_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
#
# # 创建一个与原图像同样大小的全零（全黑）图像用于floodFill
# h, w = closed_img.shape[:2]
# mask = np.zeros((h+2, w+2), np.uint8)
#
# # floodFill 从图像的内部某点开始填充（通常选择靠近中心的白点）
# cv2.floodFill(closed_img, mask, (w//2, h//2), 255)
#
# # 反转图像颜色
# # inverted_img = cv2.bitwise_not(closed_img)
# inverted_img = closed_img
# # 使用 matplotlib 显示图像
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(img, cmap='gray')
# plt.title("Original Image")
# plt.axis('off')
#
# plt.subplot(1, 2, 2)
# plt.imshow(inverted_img, cmap='gray')
# plt.title("Inverted Filled Image")
# plt.axis('off')
#
# plt.show()

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi


def find_internal_seed_point(contour, image_shape):
    """ Find a safe seed point inside the given contour. """
    if contour is not None and len(contour) > 0:
        # Calculate moments of the first contour to find the centroid
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return (cx, cy)
        else:
            return None
    return None


# def fill_contour(img):
#     # Convert the image to gray scale and find contours
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     contours, hierarchy = cv2.findContours(gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Find the largest contour assuming it's the object to fill
#     if contours:
#         largest_contour = max(contours, key=cv2.contourArea)
#         seed_point = find_internal_seed_point(largest_contour, img.shape)
#
#         if seed_point:
#             # Create a mask for floodFill
#             h, w = img.shape[:2]
#             mask = np.zeros((h + 2, w + 2), np.uint8)
#
#             # Perform floodFill
#             flood_filled_image = img.copy()
#             cv2.floodFill(flood_filled_image, mask, seed_point, (255, 255, 255))
#
#             return flood_filled_image
#     return img
# def fill_contour(img):
#     # Convert to grayscale if not already
#     if len(img.shape) > 2:
#         gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     else:
#         gray_img = img.copy()
#
#     # Threshold the image to make sure it is binary
#     _, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
#
#     # Use morphological closing to close small holes in the contour
#     kernel = np.ones((10, 10), np.uint8)
#     closed_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
#
#     # Find contours from the closed image
#     contours, _ = cv2.findContours(closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Create an image to use for the flood fill
#     flood_fill_img = closed_img.copy()
#     h, w = flood_fill_img.shape
#     mask = np.zeros((h+2, w+2), np.uint8)
#
#     # Find the largest contour and use it to find a seed point
#     if contours:
#         largest_contour = max(contours, key=cv2.contourArea)
#         moments = cv2.moments(largest_contour)
#         if moments['m00'] != 0:
#             cx = int(moments['m10'] / moments['m00'])
#             cy = int(moments['m01'] / moments['m00'])
#             # Perform the flood fill from the seed point
#             cv2.floodFill(flood_fill_img, mask, (cx, cy), 255)
#
#     # Invert the image if the background is white
#     if np.mean(flood_fill_img[0:5, 0:5]) > 127:
#         flood_fill_img = cv2.bitwise_not(flood_fill_img)
#
#     return flood_fill_img

# Load image
base_dir = '/root/nfs/gaobin/wt/Datasets/Polyp/Results/RESULT_MAP/PVT_UNet_NewAtt'
img_path = os.path.join(base_dir,'2024-08-06_02-48-17and153_net_0/ETIS-LaribPolypDB/9.png')
# img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.imread(img_path)

# Fill the contour
# filled_image = fill_contour(img)
_, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

filled_image = fill_coins = ndi.binary_fill_holes(binary_img)

filled_image = (filled_image * 255).astype(np.uint8)
# Display the images
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(122)
plt.title('Filled Image')
plt.imshow(cv2.cvtColor(filled_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
