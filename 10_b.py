import cv2
import numpy as np
import matplotlib.pyplot as plt

def image_averaging(image1_path, image2_path):
    
    image1 = cv2.imread(image1_path).astype(np.float32)
    image2 = cv2.imread(image2_path).astype(np.float32)
    
    if image1 is None or image2 is None:
        print(f"Error: Unable to load images from {image1_path} or {image2_path}")
        return
    

    if image1.shape != image2.shape:
        print("Error: Images must have the same dimensions.")
        return
    
    
    averaged_image = (image1 + image2) / 2
    
    
    averaged_image = averaged_image.astype(np.uint8)
    
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    

    axs[0].imshow(cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_BGR2RGB))
    axs[0].set_title('Image 1')
    axs[0].axis('off')
    
    axs[1].imshow(cv2.cvtColor(image2.astype(np.uint8), cv2.COLOR_BGR2RGB))
    axs[1].set_title('Image 2')
    axs[1].axis('off')
    

    axs[2].imshow(cv2.cvtColor(averaged_image, cv2.COLOR_BGR2RGB))
    axs[2].set_title('Averaged Image')
    axs[2].axis('off')
    
    
    plt.tight_layout()
    plt.show()

image1_path = "P:\\PCA2_10071023015\\nature.jpg" 
image2_path = "P:\\PCA2_10071023015\\image.jpg"  

# Perform image averaging
image_averaging(image1_path, image2_path)
