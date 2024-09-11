import cv2
import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt

def apply_convolution(image, mask):
    """Apply convolution to the image using the given mask.
    
    Args:
        image (numpy.ndarray): The input grayscale image.
        mask (numpy.ndarray): The convolution mask.
        
    Returns:
        numpy.ndarray: The image after convolution.
    """
    return convolve(image, mask, mode='reflect')

def display_images(original, convolved, title):
    """Display the original and convolved images.
    
    Args:
        original (numpy.ndarray): The original grayscale image.
        convolved (numpy.ndarray): The image after convolution.
        title (str): The title for the subplot.
    """
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(convolved, cmap='gray')
    plt.title(title)
    plt.axis('off')
    
    plt.show()

def main():
    # Load the grayscale image
    image_path = "P:\\PCA2_10071023015\\nature.jpg"  # Change this to your image file path
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Error loading image from path: {image_path}")
        return

    # Define the 3x3 averaging mask
    mask_3x3 = np.ones((3, 3), dtype=np.float32) / 9.0
    
    # Define the 5x5 averaging mask
    mask_5x5 = np.ones((5, 5), dtype=np.float32) / 25.0
    
    # Apply convolution with the 3x3 mask
    convolved_3x3 = apply_convolution(image, mask_3x3)
    
    # Apply convolution with the 5x5 mask
    convolved_5x5 = apply_convolution(image, mask_5x5)
    
    # Display the results
    display_images(image, convolved_3x3, 'Convolved with 3x3 Mask')
    display_images(image, convolved_5x5, 'Convolved with 5x5 Mask')

if __name__ == "__main__":
    main()
