"""
This script is used to augment the images in the dataset
"""
import cv2
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from remove_bg import BackgroundRemover

# Define the BackgroundRemover instance
bg_remover = BackgroundRemover()


# All your augmentation functions remain the same, no change there
def rotate(image):
    """
    Rotate image by angle 15

    Parameters
    ----------
    image : numpy.ndarray
        Image to be rotated

    Returns
    -------
    numpy.ndarray
        Rotated image
    """
    angle = 15
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))


def horizontal_flip(image):
    """
    Flip image horizontally

    Parameters
    ----------
    image : numpy.ndarray
        Image to be flipped

    Returns
    -------
    numpy.ndarray
        Flipped image
    """
    return cv2.flip(image, 1)


def increase_brightness(image, alpha=1.5, beta=30):
    """
    Increase the brightness of the image.

    Parameters
    ----------
    image : numpy.ndarray
        The image to be adjusted.
    alpha : float
        Contrast control (1.0-3.0).
    beta : int
        Brightness control (0-100).

    Returns
    -------
    numpy.ndarray
        The adjusted image.
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def decrease_brightness(image, alpha=0.8, beta=-30):
    """
    Decrease the brightness of the image.

    Parameters
    ----------
    image : numpy.ndarray
        The image to be adjusted.
    alpha : float
        Contrast control (0.1-1.0).
    beta : int
        Brightness control (-100-0).

    Returns
    -------
    numpy.ndarray
        The adjusted image.
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def add_gaussian_noise(image):
    """
    Add gaussian noise to image

    Parameters
    ----------
    image : numpy.ndarray
        Image to be added noise

    Returns
    -------
    numpy.ndarray
        Image with noise
    """
    row, col, ch = image.shape
    mean = 0
    var = 2500
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    return image + gauss


def remove_background(image):
    foreground_cv, background_image_cv = bg_remover.separate_background_single(image)
    return foreground_cv


def remove_foreground(image):
    foreground_cv, background_image_cv = bg_remover.separate_background_single(image)
    return background_image_cv


# Dictionary of available augmentations
available_augmentations = {
    "rotate": rotate,
    "horizontal_flip": horizontal_flip,
    "increase_brightness": increase_brightness,
    "decrease_brightness": decrease_brightness,
    "gaussian_noise": add_gaussian_noise,
    "remove_background": remove_background,
    "remove_foreground": remove_foreground,
}


def data_augmentation(image):
    augmented_images = []

    for aug_name, aug_function in available_augmentations.items():
        augmented = image.copy()
        augmented = aug_function(augmented)
        augmented_images.append(augmented)

    return augmented_images


def main():
    # List of directories
    folders = [
        "../data/raw/train/Divers",
        "../data/raw/train/IdealisiertNormschön",
    ]

    # Loop through each directory
    for folder in folders:
        # Get the list of filenames in the directory
        filenames = [f for f in os.listdir(folder) if f.endswith(".jpg")]

        # Loop through each file in the directory with tqdm for a progress bar
        for filename in tqdm(filenames, desc=f"Processing images in {folder}"):
            image_path = os.path.join(folder, filename)
            image = cv2.imread(image_path)

            augmented_images = data_augmentation(image)

            # Save the augmented images
            for i, img in enumerate(augmented_images):
                new_filename = f"{filename.split('.')[0]}_augmented_{i}.jpg"
                new_path = os.path.join(folder, new_filename)
                cv2.imwrite(new_path, img)


if __name__ == "__main__":
    main()
