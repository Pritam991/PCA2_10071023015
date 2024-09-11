import cv2 as cv 

img = cv.imread("P:\\PCA2_10071023015\\nature.jpg")

shape_image = img.shape

if (len(shape_image) == 3):
    height = shape_image[0]
    width = shape_image[1]
    chann = shape_image[2]

print(f"Image size: {(height * width * chann) // 1024} KB")
print(f"Height: {height}")
print(f"Width: {width}")
