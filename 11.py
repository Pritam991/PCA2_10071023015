import cv2
import numpy as np
from scipy.ndimage import median_filter

def apply_standard_average_filter(image, kernel_size=3):
    """Apply a Standard Average/Box filter."""
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    return cv2.filter2D(image, -1, kernel)

def apply_weighted_average_filter(image):
    """Apply a Weighted Average filter (e.g., Gaussian blur)."""
    # A simple weighted average filter can be a Gaussian filter
    return cv2.GaussianBlur(image, (3, 3), 0)

def apply_median_filter(image, kernel_size=3):
    """Apply a Median filter."""
    return median_filter(image, size=kernel_size)

def main():
    # Load a grayscale image
    image = cv2.imread('P:\\PCA2_10071023015\\nature.jpg', cv2.IMREAD_GRAYSCALE)

    # Apply Standard Average/Box filter
    standard_avg_filtered_image = apply_standard_average_filter(image, kernel_size=3)

    # Apply Weighted Average filter
    weighted_avg_filtered_image = apply_weighted_average_filter(image)

    # Apply Median filter
    median_filtered_image = apply_median_filter(image, kernel_size=3)

    # Display results
    cv2.imshow('Original Image', image)
    cv2.imshow('Standard Average Filtered', standard_avg_filtered_image)
    cv2.imshow('Weighted Average Filtered', weighted_avg_filtered_image)
    cv2.imshow('Median Filtered', median_filtered_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
