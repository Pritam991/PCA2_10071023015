import cv2
import numpy as np

def power_law_transform(image, gamma):

    normalized_img = image / 255.0
    power_law_img = np.power(normalized_img, gamma)
    power_law_img = np.uint8(power_law_img * 255)
    return power_law_img
image_path = "P:\\PCA2_10071023015\\nature.jpg" 
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error loading image.")
else:
    gamma = 2.0  
    power_law_image = power_law_transform(image, gamma)
    cv2.imshow('Original Image', image)
    cv2.imshow('Power Law Transformed Image', power_law_image)

    cv2.imwrite('power_law_transformed_image.jpg', power_law_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
