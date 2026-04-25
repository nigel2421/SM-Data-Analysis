import cv2
import numpy as np

def remove_and_restore(original_img, mask):
    """
    Removes handwriting identified by the mask and restores
    underlying structural lines using OpenCV Inpainting.
    """
    # If the original image is grayscale, convert to BGR for inpainting if needed
    # (cv2.inpaint works on both, but pipeline might prefer color)
    if len(original_img.shape) == 2:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)

    # Dilate the mask slightly to ensure we catch the edges of the pen strokes
    kernel = np.ones((3,3), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Restore the area under the handwriting
    # Using INPAINT_TELEA as suggested, but INPAINT_NS is also an option
    result = cv2.inpaint(original_img, dilated_mask, 3, cv2.INPAINT_TELEA)
    return result
