# Preprocessing function

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def preprocess_image(image_path):
  
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
   
    # Original image
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   
    # Grayscale image
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_image = cv2.resize(grayscale_image, (64, 64))
   
    # Preprocessed image
    preprocessed_image = grayscale_image / 255.0  # Normalize
    preprocessed_image = np.expand_dims(preprocessed_image, axis=-1)  # Add channel dimension

    return original_image, grayscale_image, preprocessed_image



# Data augmentation function
def augment_image(image):
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    # Preprocess the image by adding a batch dimension
    image = np.expand_dims(image, axis=0)  

    # Generate augmented images
    augmented_images = [next(datagen.flow(image, batch_size=1))[0] for _ in range(3)]

    return augmented_images


# Display images from the dataset
def display_dataset_images(dataset_path):
    """
    Display one image from each folder in the dataset with all preprocessing steps.
    """
    folders = os.listdir(dataset_path)
    total_images = len(folders) * 5  # 1 Original + 1 Grayscale + 1 Preprocessed + 3 Augmented
    plt.figure(figsize=(15, len(folders) * 5))  # Adjust the figure size for responsiveness
    print("Processing images...\n")
    
    for idx, folder in enumerate(folders):
        folder_path = os.path.join(dataset_path, folder)
        image_files = os.listdir(folder_path)
        
        if len(image_files) == 0:
            print(f"No images found in folder: {folder}")
            continue

        image_file = image_files[0]  # Take the first image from each folder
        image_path = os.path.join(folder_path, image_file)
        
        # Log the start of processing
        print(f"\nProcessing folder: {folder}")
        print(f"Loading image: {image_file}")
        
        # Preprocess the image
        original_image, grayscale_image, preprocessed_image = preprocess_image(image_path)
        
        # Debug: Check preprocessing output
        print(f"Original image shape: {original_image.shape}, dtype: {original_image.dtype}")
        print(f"Grayscale image shape: {grayscale_image.shape}, dtype: {grayscale_image.dtype}")
        print(f"Preprocessed image shape: {preprocessed_image.shape}, dtype: {preprocessed_image.dtype}")
        
        # Augment preprocessed image
        augmented_images = augment_image(preprocessed_image)
        
        # Debug: Check augmented images
        for aug_idx, aug_img in enumerate(augmented_images):
            print(f"Augmented Image {aug_idx+1}: shape {aug_img.shape}, dtype: {aug_img.dtype}, "
                  f"min {aug_img.min()}, max {aug_img.max()}")
            if aug_img.max() <= 1:  # If normalized, scale to 0-255
                augmented_images[aug_idx] = (aug_img * 255).astype('uint8')

        # Display original image
        plt.subplot(len(folders), 5, idx * 5 + 1)
        plt.imshow(original_image)
        plt.title(f"{folder}: Original")
        plt.axis("off")
        
        # Display grayscale image
        plt.subplot(len(folders), 5, idx * 5 + 2)
        plt.imshow(grayscale_image, cmap="gray")
        plt.title(f"{folder}: Grayscale")
        plt.axis("off")
        
        # Display preprocessed image
        plt.subplot(len(folders), 5, idx * 5 + 3)
        plt.imshow(preprocessed_image.squeeze(), cmap="gray")
        plt.title(f"{folder}: Preprocessed")
        plt.axis("off")
        
        # Display augmented images
        for aug_idx, aug_img in enumerate(augmented_images):
            subplot_index = idx * 5 + 4 + aug_idx
            if subplot_index <= total_images:  # Ensure subplot index is valid
                plt.subplot(len(folders), 5, subplot_index)
                plt.imshow(aug_img.squeeze(), cmap="gray")
                plt.title(f"{folder}: Aug {aug_idx+1}")
                plt.axis("off")
                
    plt.tight_layout()
    plt.show()

