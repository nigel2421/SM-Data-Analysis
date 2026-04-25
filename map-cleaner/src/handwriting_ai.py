import requests
import os
import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def get_handwriting_mask(image_path, model_id=None):
    """
    Orchestrates handwriting detection based on DETECTION_MODE.
    """
    mode = os.getenv('DETECTION_MODE', 'nanonets').lower()
    
    if mode == 'local':
        print("Using Local CV-based detection...")
        return get_handwriting_mask_local(image_path)
    else:
        print("Using NanoNets API detection...")
        return get_handwriting_mask_nanonets(image_path, model_id)

def get_handwriting_mask_local(image_path):
    """
    FREE LOCAL ALTERNATIVE:
    Uses adaptive thresholding and contour analysis to find handwriting.
    Identifies dark, irregular shapes that are likely handwritten annotations.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Pre-process for better contour detection
    # Adaptive thresholding to handle lighting variations
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Noise removal
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros_like(gray)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect_ratio = float(w)/h
        
        # Heuristics for handwriting on maps:
        # 1. Not too small (noise)
        # 2. Not too huge (property borders)
        # 3. Irregular shapes (not perfect rectangles or straight lines)
        if 50 < area < 5000:
            if 0.2 < aspect_ratio < 5.0:
                # This approximates handwritten numbers/text regions
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                # Dilate slightly to cover the full stroke
                cv2.rectangle(mask, (x-2, y-2), (x+w+2, y+h+2), 255, -1)
                
    return mask

def get_handwriting_mask_nanonets(image_path, model_id=None):
    """
    Sends the image to NanoNets to get bounding boxes of handwriting.
    """
    api_key = os.getenv('NANONETS_API_KEY')
    model_id = model_id or os.getenv('MODEL_ID')
    
    if not api_key:
        raise ValueError("NANONETS_API_KEY not found in environment. Please add it to .env or switch DETECTION_MODE=local")
    if not model_id:
        raise ValueError("MODEL_ID not found in environment")

    url = f'https://app.nanonets.com/api/v2/OCR/Model/{model_id}/LabelFile/'
    
    with open(image_path, 'rb') as f:
        response = requests.post(url, auth=(api_key, ''), files={'file': f})
    
    if response.status_code != 200:
        raise Exception(f"NanoNets API Error: {response.text}")
        
    data = response.json()
    
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for result in data.get('result', []):
        for prediction in result.get('prediction', []):
            if prediction.get('label') == 'handwritten':
                xmin = int(prediction.get('xmin'))
                ymin = int(prediction.get('ymin'))
                xmax = int(prediction.get('xmax'))
                ymax = int(prediction.get('ymax'))
                cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), 255, -1)
                
    return mask
