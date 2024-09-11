import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

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

def add_gaussian_noise(image, mean=0, var=0.01):
    
    noisy_image = image.copy()
    sigma = var ** 0.5
    gaussian_noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = np.clip(noisy_image + gaussian_noise, 0, 255)
    return noisy_image.astype(np.uint8)

def apply_convolution(image, mask):
   
    return convolve(image, mask, mode='reflect')

def display_results(original, noisy, filtered_3x3, filtered_5x5, title):
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(noisy, cmap='gray')
    plt.title(f'Noisy Image ({title})')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(filtered_3x3, cmap='gray')
    plt.title('Filtered (3x3)')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(filtered_5x5, cmap='gray')
    plt.title('Filtered (5x5)')
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
    salt_pepper_noisy_image = add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01)
    
    # Add Gaussian noise
    gaussian_noisy_image = add_gaussian_noise(image, mean=0, var=0.01)
    
    # Define the 3x3 and 5x5 averaging masks
    mask_3x3 = np.ones((3, 3), dtype=np.float32) / 9.0
    mask_5x5 = np.ones((5, 5), dtype=np.float32) / 25.0
    
    # Apply convolution with the 3x3 and 5x5 masks
    filtered_3x3_salt_pepper = apply_convolution(salt_pepper_noisy_image, mask_3x3)
    filtered_5x5_salt_pepper = apply_convolution(salt_pepper_noisy_image, mask_5x5)
    
    filtered_3x3_gaussian = apply_convolution(gaussian_noisy_image, mask_3x3)
    filtered_5x5_gaussian = apply_convolution(gaussian_noisy_image, mask_5x5)
    
    # Display results for salt and pepper noise
    display_results(image, salt_pepper_noisy_image, filtered_3x3_salt_pepper, filtered_5x5_salt_pepper, 'Salt & Pepper Noise')
    
    # Display results for Gaussian noise
    display_results(image, gaussian_noisy_image, filtered_3x3_gaussian, filtered_5x5_gaussian, 'Gaussian Noise')

if __name__ == "__main__":
    main()
