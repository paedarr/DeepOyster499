'''Overview: image_augmentation_pipeline.py is a Python script designed for performing bulk image augmentation on a collection of images. This script applies multiple image transformation techniques, including rotation, flipping, resizing, cropping, brightness adjustment, contrast adjustment, affine transformations (shearing), and adding Gaussian noise. The script is can be used for enhancing a dataset by increasing its size through the augmentation of existing images.'''

from PIL import Image, ImageEnhance, ImageTransform
import numpy as np
import os

# Function to create output directory if it doesn't exist
def create_output_dir(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

# Rotation
def rotate_image(image, degrees, output_folder, filename):
    rotated_image = image.rotate(degrees)
    rotated_image.save(os.path.join(output_folder, f"{filename}_rotated_{degrees}.jpg"))

# Flipping
def flip_image(image, output_folder, filename, mode='horizontal'):
    if mode == 'horizontal':
        flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    elif mode == 'vertical':
        flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)
    flipped_image.save(os.path.join(output_folder, f"{filename}_flipped_{mode}.jpg"))

# # Scaling (Resizing)
# def resize_image(image, scale_factor, output_folder, filename):
#     width, height = image.size
#     resized_image = image.resize((int(width * scale_factor), int(height * scale_factor)))
#     resized_image.save(os.path.join(output_folder, f"{filename}_resized_{scale_factor}.jpg"))

# # Cropping
# def crop_image(image, output_folder, filename, crop_area=(50, 50, 200, 200)):
#     cropped_image = image.crop(crop_area)
#     cropped_image.save(os.path.join(output_folder, f"{filename}_cropped.jpg"))

# Brightness Adjustment
def adjust_brightness(image, factor, output_folder, filename):
    enhancer = ImageEnhance.Brightness(image)
    brightened_image = enhancer.enhance(factor)
    brightened_image.save(os.path.join(output_folder, f"{filename}_brightness_{factor}.jpg"))

# Contrast Adjustment
def adjust_contrast(image, factor, output_folder, filename):
    enhancer = ImageEnhance.Contrast(image)
    contrasted_image = enhancer.enhance(factor)
    contrasted_image.save(os.path.join(output_folder, f"{filename}_contrast_{factor}.jpg"))

# Affine Transformation (Shearing)
def shear_image(image, shear_factor, output_folder, filename):
    width, height = image.size
    shear_transform = ImageTransform.AffineTransform((1, shear_factor, 0, shear_factor, 1, 0))
    sheared_image = image.transform((width, height), Image.AFFINE, shear_transform.data)
    sheared_image.save(os.path.join(output_folder, f"{filename}_sheared_{shear_factor}.jpg"))

# Gaussian Noise
def add_gaussian_noise(image, output_folder, filename, mean=0, std=25):
    image_np = np.array(image)
    noise = np.random.normal(mean, std, image_np.shape).astype(np.uint8)
    noisy_image = Image.fromarray(np.clip(image_np + noise, 0, 255).astype(np.uint8))
    noisy_image.save(os.path.join(output_folder, f"{filename}_noisy.jpg"))

# Main pipeline function to apply all transformations
def augment_image(image_path, output_folder):
    # Create the output directory if it doesn't exist
    create_output_dir(output_folder)
    
    # Load the image
    image = Image.open(image_path)
    
    # Get the image filename (without extension)
    filename = os.path.splitext(os.path.basename(image_path))[0]

    # Apply transformations
    rotate_image(image, np.random.randint(0, 360), output_folder, filename)
    flip_image(image, output_folder, filename, mode='horizontal')
    flip_image(image, output_folder, filename, mode='vertical')
    # resize_image(image, 0.8, output_folder, filename)
    # crop_image(image, output_folder, filename, crop_area=(50, 50, 200, 200))
    adjust_brightness(image, np.random.uniform(0.5, 1.5), output_folder, filename)
    adjust_contrast(image, np.random.uniform(0.5, 1.5), output_folder, filename)
    shear_image(image, np.random.uniform(-0.3, 0.3), output_folder, filename)
    add_gaussian_noise(image, output_folder, filename, mean=0, std=np.random.uniform(10, 50))

# Function to process multiple images
def augment_images_in_folder(input_folder, output_folder):
    # Get list of all image files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_folder, filename)
            augment_image(image_path, output_folder)


script_dir = os.path.dirname(os.path.abspath(__file__))

# Define base folders for original and augmented images
input_folder_base = os.path.join(script_dir, "..", "dataset")  # base folder for original images
output_folder_base = os.path.join(script_dir, "..", "aug_dataset")  # base folder for augmented images

# Iterate through 'train' and 'validation' folders
for dataset_type in ['train', 'validation']:
    for i in range(5):
        input_folder = os.path.join(input_folder_base, dataset_type, str(i))
        output_folder = os.path.join(output_folder_base, dataset_type, str(i))
        augment_images_in_folder(input_folder, output_folder)
        print(f"Augmented images in folder {input_folder} and saved to {output_folder}")

print("Image augmentation complete.")

