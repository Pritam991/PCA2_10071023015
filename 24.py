import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

def create_motion_blur_kernel(length, angle):
    
    kernel = np.zeros((length, length))
    
    # Convert angle to radians
    angle = np.deg2rad(angle)
    
    # Calculate the center of the kernel
    center = length // 2
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    for i in range(length):
        for j in range(length):
            # Calculate distance from the center
            x = i - center
            y = j - center
            
            # Rotate the coordinates
            x_rot = cos_angle * x - sin_angle * y
            y_rot = sin_angle * x + cos_angle * y
            
            # If the rotated coordinate is close to the center line, set value
            if np.abs(x_rot) < 1 and np.abs(y_rot) < 1:
                kernel[i, j] = 1
    
    # Normalize the kernel
    kernel /= np.sum(kernel)
    
    return kernel

def apply_motion_blur(image, kernel):
    
    return convolve(image, kernel, mode='reflect')

def display_images(original, blurred):
    """Display the original and blurred images."""
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(blurred, cmap='gray')
    plt.title('Motion Blurred Image')
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
    
    # Create motion blur kernel
    length = 15  # Length of the blur effect
    angle = 45   # Angle of the motion blur in degrees
    kernel = create_motion_blur_kernel(length, angle)
    
    # Apply motion blur
    blurred_image = apply_motion_blur(image, kernel)
    
    # Display results
    display_images(image, blurred_image)

if __name__ == "__main__":
    main()
