import cv2
import os

def save_image(img, output_path):
    """
    Saves an image to the specified path, creating directories if necessary.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)

def list_raw_images(directory='data/raw'):
    """
    Lists all images in the raw data directory.
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    return [os.path.join(directory, f) for f in os.listdir(directory) 
            if f.lower().endswith(valid_extensions)]
