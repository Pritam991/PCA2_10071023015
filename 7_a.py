import cv2

def invert_image(image):
    return 255 - image

image = cv2.imread("P:\\PCA2_10071023015\\nature.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error loading image.")
else:
    
    inverted_image = invert_image(image)
        
    cv2.imshow('Original Image', image)
    cv2.imshow('Inverted Image', inverted_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
