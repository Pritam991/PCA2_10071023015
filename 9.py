import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_equalization(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return
    
    # Perform histogram equalization
    equalized_image = cv2.equalizeHist(image)
    
    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original Image
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    # Equalized Image
    axs[1].imshow(equalized_image, cmap='gray')
    axs[1].set_title('Histogram Equalized Image')
    axs[1].axis('off')
    
    # Show plot
    plt.tight_layout()
    plt.show()

# Path to your image
image_path = "P:\\PCA2_10071023015\\nature.jpg"

# Perform histogram equalization
histogram_equalization(image_path)
