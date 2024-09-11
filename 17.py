import cv2
import numpy as np
import matplotlib.pyplot as plt

def equalize_histogram(image):

    return cv2.equalizeHist(image)

def plot_histogram(image, title):
    
    plt.hist(image.ravel(), 256, [0, 256])
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

def main():
    # Load the image
    image_path = 'P:\\PCA2_10071023015\\nature.jpg'  # Change this to your image file path
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Error loading image from path: {image_path}")
        return

    # Perform histogram equalization
    equalized_image = equalize_histogram(image)

    # Display original and equalized images
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(equalized_image, cmap='gray')
    plt.title('Equalized Image')
    plt.axis('off')

    # Plot histograms
    plt.subplot(2, 2, 3)
    plot_histogram(image, 'Original Histogram')
    
    plt.subplot(2, 2, 4)
    plot_histogram(equalized_image, 'Equalized Histogram')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
