import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift

def create_filter(shape, filter_type, cutoff1, cutoff2=None, order=2):
    
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    
    row, col = np.indices((rows, cols))
    distance = np.sqrt((row - center_row) ** 2 + (col - center_col) ** 2)
    
    if filter_type == 'low':
        filter_ = 1 / (1 + (distance / cutoff1) ** (2 * order))
    elif filter_type == 'high':
        filter_ = 1 / (1 + (cutoff1 / distance) ** (2 * order))
    elif filter_type == 'band':
        filter_ = (1 / (1 + (distance / cutoff1) ** (2 * order))) * \
                  (1 - 1 / (1 + (distance / cutoff2) ** (2 * order)))
    elif filter_type == 'bandstop':
        filter_ = 1 - (1 / (1 + (distance / cutoff1) ** (2 * order))) * \
                    (1 - 1 / (1 + (distance / cutoff2) ** (2 * order)))
    
    return filter_

def apply_filter(image, filter_type, cutoff1, cutoff2=None, order=2):
    """Apply a frequency domain filter to an image."""
    # Convert image to frequency domain
    image_freq = fft2(image)
    image_freq_shifted = fftshift(image_freq)
    
    # Create filter
    filter_ = create_filter(image.shape, filter_type, cutoff1, cutoff2, order)
    
    # Apply filter in frequency domain
    filtered_freq = image_freq_shifted * filter_
    
    # Convert back to spatial domain
    filtered_freq_shifted = ifftshift(filtered_freq)
    filtered_image = np.abs(ifft2(filtered_freq_shifted))
    
    return filtered_image

def display_results(original, low_pass, high_pass, band_pass, band_stop):
    """Display the original and filtered images."""
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(low_pass, cmap='gray')
    plt.title('Low-pass Filter')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(high_pass, cmap='gray')
    plt.title('High-pass Filter')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(band_pass, cmap='gray')
    plt.title('Band-pass Filter')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(band_stop, cmap='gray')
    plt.title('Band-stop Filter')
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
    
    # Define cutoff frequencies
    cutoff_low = 30
    cutoff_high = 60
    cutoff_band_low = 20
    cutoff_band_high = 50
    
    # Apply filters
    low_pass = apply_filter(image, 'low', cutoff_low)
    high_pass = apply_filter(image, 'high', cutoff_low)
    band_pass = apply_filter(image, 'band', cutoff_band_low, cutoff_band_high)
    band_stop = apply_filter(image, 'bandstop', cutoff_band_low, cutoff_band_high)
    
    # Display results
    display_results(image, low_pass, high_pass, band_pass, band_stop)

if __name__ == "__main__":
    main()
