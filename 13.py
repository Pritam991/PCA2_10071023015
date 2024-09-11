import cv2
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

def homomorphic_filter(image, d0=30, gamma_l=0.5, gamma_h=2.0, c=1.0):
   
    rows, cols = image.shape
    image_log = np.log1p(np.array(image, dtype="float"))

    # Fourier Transform
    image_fft = fft2(image_log)
    image_fft_shift = fftshift(image_fft)

    # Create a high-pass filter
    U, V = np.meshgrid(np.arange(cols), np.arange(rows))
    D_uv = np.sqrt((U - cols/2)**2 + (V - rows/2)**2)
    H_uv = (gamma_h - gamma_l) * (1 - np.exp(-c * (D_uv**2 / (d0**2)))) + gamma_l

    # Apply filter on the Fourier transformed image
    result_filter = H_uv * image_fft_shift

    # Inverse Fourier Transform
    result_ifft_shift = ifftshift(result_filter)
    result_ifft = ifft2(result_ifft_shift)
    result_exp = np.expm1(np.real(result_ifft))

    # Normalize the image to 0-255
    result_exp = np.uint8(cv2.normalize(result_exp, None, 0, 255, cv2.NORM_MINMAX))

    return result_exp

def main():
    # Load a grayscale image
    image = cv2.imread('P:\\PCA2_10071023015\\nature.jpg', cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Unable to load image. Check the file path.")
        return

    # Apply Homomorphic filter
    filtered_image = homomorphic_filter(image, d0=30, gamma_l=0.5, gamma_h=2.0, c=1.0)

    # Display results
    cv2.imshow('Original Image', image)
    cv2.imshow('Homomorphic Filtered Image', filtered_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
