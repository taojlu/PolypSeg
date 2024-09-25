import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return binary

def find_contours(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def calculate_smoothness(contour):
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if area == 0:
        return np.inf  # Return infinity if area is zero to indicate no smoothness score
    smoothness = perimeter**2 / (4 * np.pi * area)
    return smoothness

def calculate_edge_coherence(contour):
    # Calculate the curvature for each point and then find the variance
    curvature = []
    pts = contour[:, 0, :]  # Simplify contour array shape
    for i in range(1, len(pts) - 1):
        p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1]
        k = np.abs((p3[1] - p2[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p3[0] - p2[0]))
        curvature.append(k)
    variance = np.var(curvature) if curvature else np.inf
    return variance

def analyze_contours(contours):
    results = []
    for contour in contours:
        smoothness = calculate_smoothness(contour)
        coherence = calculate_edge_coherence(contour)
        perimeter = cv2.arcLength(contour, True)  # Calculate the contour's perimeter
        results.append((smoothness, coherence, perimeter))
    return results


def plot_results(image, contours, results):
    plt.figure(figsize=(10, 8))
    plt.imshow(image, cmap='gray')
    for i, contour in enumerate(contours):
        smoothness, coherence, perimeter = results[i]  # Update unpacking to include perimeter
        label = f'Contour {i+1}: Smoothness = {smoothness:.2f}, Coherence = {coherence:.2f}, Perimeter = {perimeter:.2f}'
        x, y, w, h = cv2.boundingRect(contour)
        plt.text(x, y - 10, label, color='red')
        plt.plot(contour[:, 0, 0], contour[:, 0, 1], linewidth=2)
    plt.title('Contour Analysis')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    base_dir = '/root/nfs/gaobin/wt/Datasets/Polyp/Results/RESULT_MAP/MaCbamSuperbisionFormer'
    image_path = os.path.join(base_dir, '2024-08-12_14-31-41and17_net_dice_0/ETIS-LaribPolypDB/179.png')
    binary_image = load_image(image_path)
    contours = find_contours(binary_image)
    results = analyze_contours(contours)
    print(results)
    plot_results(binary_image, contours, results)
