import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_uniform_noise(image, low=0, high=255):
    """Add uniform noise to an image."""
    uniform_noise = np.random.uniform(low, high, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, uniform_noise)
    return noisy_image

def add_gaussian_noise(image, mean=0, sigma=25):
    """Add Gaussian noise to an image."""
    gaussian_noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy_image = cv2.add(image.astype(np.float32), gaussian_noise)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def add_salt_and_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05):
    """Add salt and pepper noise to an image."""
    noisy_image = np.copy(image)
    total_pixels = image.size
    num_salt = int(np.ceil(salt_prob * total_pixels))
    num_pepper = int(np.ceil(pepper_prob * total_pixels))

    # Salt noise
    salt_coords = [np.random.randint(0, i, num_salt) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 255

    # Pepper noise
    pepper_coords = [np.random.randint(0, i, num_pepper) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0

    return noisy_image


def add_rayleigh_noise(image, scale=25):
    """Add Rayleigh noise to an image."""
    rayleigh_noise = np.random.rayleigh(scale, image.shape).astype(np.float32)
    noisy_image = cv2.add(image.astype(np.float32), rayleigh_noise)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def add_exponential_noise(image, scale=25):
    """Add Exponential noise to an image."""
    exponential_noise = np.random.exponential(scale, image.shape).astype(np.float32)
    noisy_image = cv2.add(image.astype(np.float32), exponential_noise)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def add_erlang_noise(image, shape=2, scale=15):
    """Add Erlang (Gamma) noise to an image."""
    erlang_noise = np.random.gamma(shape, scale, image.shape).astype(np.float32)
    noisy_image = cv2.add(image.astype(np.float32), erlang_noise)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def display_images(original, noisy_images, titles):
    """Display images using Matplotlib."""
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 4, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    for i in range(len(noisy_images)):
        plt.subplot(2, 4, i + 2)
        plt.imshow(noisy_images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Load a grayscale image
    image = cv2.imread('P:\\PCA2_10071023015\\nature.jpg', cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("Error: Unable to load image. Check the file path.")
        return

    # Generate noisy images using different noise models
    uniform_noisy_image = add_uniform_noise(image)
    gaussian_noisy_image = add_gaussian_noise(image)
    salt_and_pepper_noisy_image = add_salt_and_pepper_noise(image)
    rayleigh_noisy_image = add_rayleigh_noise(image)
    exponential_noisy_image = add_exponential_noise(image)
    erlang_noisy_image = add_erlang_noise(image)

    # Display images
    noisy_images = [
        uniform_noisy_image, gaussian_noisy_image, salt_and_pepper_noisy_image,
        rayleigh_noisy_image, exponential_noisy_image, erlang_noisy_image
    ]
    titles = [
        'Uniform Noise', 'Gaussian Noise', 'Salt & Pepper Noise',
        'Rayleigh Noise', 'Exponential Noise', 'Erlang Noise'
    ]
    display_images(image, noisy_images, titles)

if __name__ == "__main__":
    main()
