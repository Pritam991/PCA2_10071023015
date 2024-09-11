import cv2
import numpy as np

def log_transform(image):
    
    image = np.where(image == 0, 1, image)
    image = image.astype(np.float32)
    c = 255 / np.log(1 + np.max(image))
    log_image = c * (np.log(image + 1))
    
    log_image = np.array(log_image, dtype=np.uint8)
    return log_image


image_path = "P:\\PCA2_10071023015\\nature.jpg" 
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error loading image.")
else:
    
    log_image = log_transform(image)
    cv2.imshow('Original Image', image)
    cv2.imshow('Log Transformed Image', log_image)
    
    cv2.imwrite('log_transformed_image.jpg', log_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
