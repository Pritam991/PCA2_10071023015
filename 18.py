import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_threshold(image, threshold):
    
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

def plot_results(original_image, binary_image, threshold):
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(binary_image, cmap='gray')
    plt.title(f'Binary Image (Threshold={threshold})')
    plt.axis('off')
    
    plt.show()

def main():
    # Load the grayscale image
    image_path = "P:\\PCA2_10071023015\\nature.jpg"  # Change this to your image file path
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Error loading image from path: {image_path}")
        return
    
    # Define the threshold value (you can change this to test different values)
    threshold = 128  # Example threshold value; adjust as needed
    
    # Apply thresholding
    binary_image = apply_threshold(image, threshold)
    
    # Display results
    plot_results(image, binary_image, threshold)

if __name__ == "__main__":
    main()
