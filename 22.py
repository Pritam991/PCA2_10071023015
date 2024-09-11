import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift

def butterworth_filter(shape, cutoff, filter_type='low', order=2):
    
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    
    row, col = np.indices((rows, cols))
    distance = np.sqrt((row - center_row) ** 2 + (col - center_col) ** 2)
    
    if filter_type == 'low':
        filter_ = 1 / (1 + (distance / cutoff) ** (2 * order))
    elif filter_type == 'high':
        filter_ = 1 / (1 + (cutoff / distance) ** (2 * order))
    
    return filter_

def apply_filter(image, filter_type, cutoff, order=2):
    
    # Convert image to frequency domain
    image_freq = fft2(image)
    image_freq_shifted = fftshift(image_freq)
    
    # Create Butterworth filter
    filter_ = butterworth_filter(image.shape, cutoff, filter_type, order)
    
    # Apply filter in frequency domain
    filtered_freq = image_freq_shifted * filter_
    
    # Convert back to spatial domain
    filtered_freq_shifted = ifftshift(filtered_freq)
    filtered_image = np.abs(ifft2(filtered_freq_shifted))
    
    return filtered_image

def display_images(original, low_pass, high_pass, cutoff1, cutoff2):
  
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(low_pass, cmap='gray')
    plt.title(f'Low-pass Filter (Cutoff={cutoff1})')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(high_pass, cmap='gray')
    plt.title(f'High-pass Filter (Cutoff={cutoff1})')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(low_pass, cmap='gray')
    plt.title(f'Low-pass Filter (Cutoff={cutoff2})')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(high_pass, cmap='gray')
    plt.title(f'High-pass Filter (Cutoff={cutoff2})')
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

    # Define cut-off frequencies
    cutoff1 = 30
    cutoff2 = 60
    
    # Apply filters
    low_pass1 = apply_filter(image, 'low', cutoff1)
    high_pass1 = apply_filter(image, 'high', cutoff1)
    low_pass2 = apply_filter(image, 'low', cutoff2)
    high_pass2 = apply_filter(image, 'high', cutoff2)
    
    # Display results
    display_images(image, low_pass1, high_pass1, cutoff1, cutoff2)
    display_images(image, low_pass2, high_pass2, cutoff1, cutoff2)

if __name__ == "__main__":
    main()
