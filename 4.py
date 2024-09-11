import cv2
import numpy as np

image = cv2.imread("P:\\PCA2_10071023015\\nature.jpg")
if image is None:
    print("No file exists")
    exit(1)

original_height, original_width = image.shape[:2]

new_width = original_width * 2
new_height = original_height * 2

enlarged_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
cv2.imshow('Original Image', image)
cv2.imshow('Enlarged Image', enlarged_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('enlarged_image.jpg', enlarged_image)
