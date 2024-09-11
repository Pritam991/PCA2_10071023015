import cv2
import numpy as np
from scipy.ndimage import convolve, median_filter
import matplotlib.pyplot as plt

def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    
    noisy_image = image.copy()
    total_pixels = image.size
    
    # Salt noise
    num_salt = np.ceil(salt_prob * total_pixels)
    salt_coords = [np.random.randint(0, i-1, int(num_salt)) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 255
    
    # Pepper noise
    num_pepper = np.ceil(pepper_prob * total_pixels)
    pepper_coords = [np.random.randint(0, i-1, int(num_pepper)) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0
    
    return noisy_image

def apply_box_filter(image, size):
    
    mask = np.ones((size, size), dtype=np.float32) / (size * size)
    return convolve(image, mask, mode='reflect')

def apply_median_filter(image, size):
    
    return median_filter(image, size=size)

def display_results(original, noisy, box_filtered_3x3, box_filtered_5x5, median_filtered, title):
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(noisy, cmap='gray')
    plt.title('Noisy Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(box_filtered_3x3, cmap='gray')
    plt.title('Box Filter 3x3')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(box_filtered_5x5, cmap='gray')
    plt.title('Box Filter 5x5')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(median_filtered, cmap='gray')
    plt.title('Median Filter')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Load the grayscale image
    image_path = 'P:\\PCA2_10071023015\\nature.jpg'  # Change this to your image file path
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Error loading image from path: {image_path}")
        return
    
    # Add salt and pepper noise
    noisy_image = add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01)
    
    # Apply filters
    box_filtered_3x3 = apply_box_filter(noisy_image, size=3)
    box_filtered_5x5 = apply_box_filter(noisy_image, size=5)
    median_filtered = apply_median_filter(noisy_image, size=3)
    
    # Display results
    display_results(image, noisy_image, box_filtered_3x3, box_filtered_5x5, median_filtered, 'Salt & Pepper Noise')

if __name__ == "__main__":
    main()
