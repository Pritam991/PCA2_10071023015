import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    hist_flat = hist.flatten()
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    axs[0, 0].imshow(image, cmap='gray')
    axs[0, 0].set_title('Image')
    axs[0, 0].axis('off')
    
    axs[0, 1].hist(image.ravel(), bins=256, range=[0, 256], color='gray')
    axs[0, 1].set_title('Histogram (imhist)')
    axs[0, 1].set_xlabel('Intensity value')
    axs[0, 1].set_ylabel('Frequency')
    
    axs[1, 0].bar(np.arange(256), hist_flat, color='gray')
    axs[1, 0].set_title('Histogram (bar)')
    axs[1, 0].set_xlabel('Intensity value')
    axs[1, 0].set_ylabel('Frequency')
    
    axs[1, 1].stem(hist_flat)
    axs[1, 1].set_title('Histogram (stem)')
    axs[1, 1].set_xlabel('Intensity value')
    axs[1, 1].set_ylabel('Frequency')
    
    plt.figure(figsize=(8, 6))
    plt.plot(hist_flat, color='gray')
    plt.title('Histogram (plot)')
    plt.xlabel('Intensity value')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

image_path = "P:\\PCA2_10071023015\\nature.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"Error: Unable to load image from {image_path}")
else:
    plot_histogram(image)
