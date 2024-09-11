from PIL import Image, ImageEnhance, ImageOps

def adjust_brightness(image, factor):
    
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def adjust_contrast(image, factor):
   
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def get_negative(image):
    
    return ImageOps.invert(image)

def main():
    # Load the image
    image_path = "P:\\PCA2_10071023015\\nature.jpg"  # Change this to your image file path
    image = Image.open(image_path)
    
    # Increase brightness
    bright_image = adjust_brightness(image, 1.5)  # Increase brightness by 50%
    bright_image.save('bright_image.jpg')
    
    # Decrease brightness
    dark_image = adjust_brightness(image, 0.5)  # Decrease brightness by 50%
    dark_image.save('dark_image.jpg')
    
    # Increase contrast
    high_contrast_image = adjust_contrast(image, 1.5)  # Increase contrast by 50%
    high_contrast_image.save('high_contrast_image.jpg')
    
    # Decrease contrast
    low_contrast_image = adjust_contrast(image, 0.5)  # Decrease contrast by 50%
    low_contrast_image.save('low_contrast_image.jpg')
    
    # Get negative of the image
    negative_image = get_negative(image.convert('RGB'))  # Ensure image is in RGB mode
    negative_image.save('negative_image.jpg')

if __name__ == "__main__":
    main()
