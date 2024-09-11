import cv2
import numpy as np
import matplotlib.pyplot as plt

def image_subtraction(image1_path, image2_path):
    # Load images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    
    if image1 is None or image2 is None:
        print(f"Error: Unable to load images from {image1_path} or {image2_path}")
        return
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Check if dimensions match
    if gray1.shape != gray2.shape:
        # Resize gray1 to match gray2 dimensions
        gray1 = cv2.resize(gray1, (gray2.shape[1], gray2.shape[0]))
    
    # Perform subtraction
    subtracted_image = cv2.subtract(gray1, gray2)
    
    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original Images
    axs[0].imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Image 1')
    axs[0].axis('off')
    
    axs[1].imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Image 2')
    axs[1].axis('off')
    
    # Subtracted Image
    axs[2].imshow(subtracted_image, cmap='gray')
    axs[2].set_title('Subtracted Image')
    axs[2].axis('off')
    
    # Show plot
    plt.tight_layout()
    plt.show()

# Paths to your images
image1_path = "P:\\PCA2_10071023015\\nature.jpg"  # Replace with your image path
image2_path = "P:\\PCA2_10071023015\\image.jpg"  # Replace with your image path

# Perform image subtraction
image_subtraction(image1_path, image2_path)
