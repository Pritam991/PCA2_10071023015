import cv2
image = cv2.imread("P:\\PCA2_10071023015\\nature.jpg")

if image is None:
    print("Error loading image.")
else:

    rotated_clockwise = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    rotated_anticlockwise = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    cv2.imshow('Original Image', image)
    cv2.imshow('Rotated Clockwise', rotated_clockwise)
    cv2.imshow('Rotated Anticlockwise', rotated_anticlockwise)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
