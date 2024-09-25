import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def classify_shape(contour, epsilon=0.01):
    # Approximate contour and check circularity
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    circularity = 4 * np.pi * (area / (perimeter ** 2))
    print(circularity)

    if 0.8 < circularity <= 1.05:
        return 'Circle', circularity

    # Check if the contour is suitable for ellipse fitting
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        (major_axis, minor_axis) = max(ellipse[1]) / 2, min(ellipse[1]) / 2
        aspect_ratio = major_axis / minor_axis
        print(aspect_ratio)
        if aspect_ratio < 1.2:
            return 'Ellipse', aspect_ratio
    # Default to irregular polygon if not enough points for ellipse or not a circle
    return 'Irregular Polygon', None

def load_and_preprocess_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return image, contours

def plot_shapes(image, contours):
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='gray')

    for contour in contours:
        shape_type, _ = classify_shape(contour)
        if shape_type == 'Circle':
            color = 'r'
        elif shape_type == 'Ellipse':
            color = 'b'
        else:
            color = 'g'

        # Draw contour
        plt.plot(contour[:, 0, 0], contour[:, 0, 1], label=shape_type, color=color)

    plt.legend()
    plt.show()

if __name__ == '__main__':
    base_dir = '/root/nfs/gaobin/wt/Datasets/Polyp/Results/RESULT_MAP/MaCbamSuperbisionFormer'
    image_path = os.path.join(base_dir, '2024-08-12_14-31-41and17_net_dice_0/CVC-300/149.png')
    image, contours = load_and_preprocess_image(image_path)
    plot_shapes(image, contours)
