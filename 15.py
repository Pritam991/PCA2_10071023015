import cv2
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

def arithmetic_mean_filter(image, kernel_size=3):
    """Apply Arithmetic Mean Filter."""
    return cv2.blur(image, (kernel_size, kernel_size))

def geometric_mean_filter(image, kernel_size=3):
    """Apply Geometric Mean Filter."""
    image = image.astype(np.float64) + 1  # Avoid log(0)
    image_log = np.log(image)
    geometric_mean = scipy.ndimage.generic_filter(image_log, np.mean, size=kernel_size)
    return np.exp(geometric_mean) - 1

def harmonic_mean_filter(image, kernel_size=3):
    """Apply Harmonic Mean Filter."""
    image = image.astype(np.float64) + 1e-5  # To avoid division by zero
    inverse_image = 1.0 / image
    harmonic_mean = scipy.ndimage.generic_filter(inverse_image, np.mean, size=kernel_size)
    return 1.0 / harmonic_mean

def median_filter(image, kernel_size=3):
    """Apply Median Filter."""
    return cv2.medianBlur(image, kernel_size)

def max_filter(image, kernel_size=3):
    """Apply Max Filter."""
    return scipy.ndimage.maximum_filter(image, size=kernel_size)

def min_filter(image, kernel_size=3):
    """Apply Min Filter."""
    return scipy.ndimage.minimum_filter(image, size=kernel_size)

def display_images(original, filtered_images, titles):
    """Display images using Matplotlib."""
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 4, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    for i in range(len(filtered_images)):
        plt.subplot(2, 4, i + 2)
        plt.imshow(filtered_images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Load a grayscale image
    image = cv2.imread('P:\\PCA2_10071023015\\nature.jpg', cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("Error: Unable to load image. Check the file path.")
        return

    # Apply various filters
    arithmetic_mean = arithmetic_mean_filter(image)
    geometric_mean = geometric_mean_filter(image)
    harmonic_mean = harmonic_mean_filter(image)
    median = median_filter(image)
    max_filt = max_filter(image)
    min_filt = min_filter(image)

    # Display images
    filtered_images = [
        arithmetic_mean, geometric_mean, harmonic_mean,
        median, max_filt, min_filt
    ]
    titles = [
        'Arithmetic Mean', 'Geometric Mean', 'Harmonic Mean',
        'Median Filter', 'Max Filter', 'Min Filter'
    ]
    display_images(image, filtered_images, titles)

if __name__ == "__main__":
    main()
