import cv2
import numpy as np

def piecewise_linear_transform(image, low, high):
    normalized_img = image / 255.0
    low = max(0, low)
    high = min(255, high)
    
    def piecewise_linear(x):
        return np.piecewise(x, [x < low, (low <= x) & (x <= high), x > high], [0, lambda x: ((x - low) / (high - low)) * 255, 255])
    
    piecewise_img = piecewise_linear(normalized_img)
    piecewise_img = np.uint8(piecewise_img)
    
    return piecewise_img

image_path = "P:\\PCA2_10071023015\\nature.jpg" 
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error loading image.")
else:

    low = 50
    high = 200
    
    piecewise_image = piecewise_linear_transform(image, low, high)

    cv2.imshow('Original Image', image)
    cv2.imshow('Piecewise Linear Transformed Image', piecewise_image)

    cv2.imwrite('piecewise_linear_transformed_image.jpg', piecewise_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
