import cv2
import numpy as np

def clean_background(image_path):
    """
    Advanced background cleaning for survey maps.
    Uses background division to remove shadows and normalize lighting.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Isolate background by heavy blurring
    # This captures the "shading" of the paper
    dilated = cv2.dilate(gray, np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated, 21)
    
    # Divide the original image by the background to "flatten" it
    # Result is essentially (pixel / background) * 255
    diff = cv2.absdiff(gray, bg_img)
    diff = 255 - diff
    
    # Increase contrast
    norm_img = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Final sharpening and thresholding to make it "pure" black and white if needed
    # (Leaving it as grayscale for better inpainting results)
    return norm_img

def enhance_text(image):
    """
    Sharpens and boldens text in the image.
    Accepts an image array and returns an enhanced version.
    """
    # If image is a path, read it
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    elif len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # Sharpening kernel
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    
    # Adaptive threshold to isolate text
    thresh = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Morphological thickening
    # (Erode on white-on-black = thickening the black characters)
    # Since we have black-on-white, we invert, dilate, then invert back
    inv = cv2.bitwise_not(thresh)
    dilated = cv2.dilate(inv, np.ones((2,2), np.uint8), iterations=1)
    result = cv2.bitwise_not(dilated)
    
    return result

def remove_noise(image):
    """
    Removes small black dots (noise/speckles) from the image.
    Uses morphological opening and contour filtering.
    """
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    elif len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # Morphological opening (Erode then Dilate) to remove small dots
    # Since we have black dots on white background, 
    # we need to consider black as "foreground" for erosion/dilation
    # or just use median blur
    denoised = cv2.medianBlur(image, 3)
    
    # Advanced: Remove small isolated components
    # Threshold to binary
    _, thresh = cv2.threshold(denoised, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Filter by area
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    
    mask = np.zeros_like(thresh)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 4: # Only keep components larger than 4 pixels
            mask[labels == i] = 255
            
    result = cv2.bitwise_not(mask)
    return result
