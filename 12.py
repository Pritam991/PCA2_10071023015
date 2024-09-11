import cv2
import numpy as np

def apply_laplacian_filter(image):
    """Apply Laplacian sharpening filter to the image."""
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    sharpened_image = cv2.convertScaleAbs(laplacian)
    return sharpened_image

def main():
    # Load a grayscale image
    image = cv2.imread('P:\\PCA2_10071023015\\nature.jpg', cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("Error: Unable to load image. Check the file path.")
        return

    # Apply Laplacian filter
    laplacian_sharpened_image = apply_laplacian_filter(image)

    # Display results
    cv2.imshow('Original Image', image)
    cv2.imshow('Laplacian Sharpened Image', laplacian_sharpened_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
