import cv2

image = cv2.imread("P:\\PCA2_10071023015\\nature.jpg")

if image is None:
    print("Error loading image.")
else:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Original Image', image)
    cv2.imshow('Grayscale Image', gray_image)
    cv2.imwrite('grayscale_image.jpg', gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
