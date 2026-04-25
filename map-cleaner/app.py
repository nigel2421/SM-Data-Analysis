import os
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, send_from_directory, jsonify
from flask_cors import CORS
from src.preprocess import clean_background, enhance_text, remove_noise
from src.handwriting_ai import get_handwriting_mask
from src.inpainter import remove_and_restore
from src.utils import save_image, list_raw_images

app = Flask(__name__)
CORS(app)

RAW_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'
MASK_DIR = 'data/masks'

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/images')
def get_images():
    images = list_raw_images(RAW_DIR)
    return jsonify([os.path.basename(img) for img in images])

@app.route('/data/raw/<filename>')
def serve_raw(filename):
    return send_from_directory(RAW_DIR, filename)

@app.route('/data/processed/<filename>')
def serve_processed(filename):
    return send_from_directory(PROCESSED_DIR, filename)

@app.route('/api/process', methods=['POST'])
def process_image():
    data = request.json
    filename = data.get('filename')
    image_path = os.path.join(RAW_DIR, filename)
    norm_img = clean_background(image_path)
    output_filename = f"cleaned_{filename}"
    output_path = os.path.join(PROCESSED_DIR, output_filename)
    save_image(norm_img, output_path)
    return jsonify({"success": True, "output": output_filename})

@app.route('/api/enhance', methods=['POST'])
def enhance():
    data = request.json
    filename = data.get('filename')
    image_path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(image_path):
        image_path = os.path.join(RAW_DIR, filename)
    img = cv2.imread(image_path)
    enhanced = enhance_text(img)
    output_filename = f"enhanced_{filename.replace('cleaned_', '').replace('enhanced_', '')}"
    output_path = os.path.join(PROCESSED_DIR, output_filename)
    save_image(enhanced, output_path)
    return jsonify({"success": True, "output": output_filename})

@app.route('/api/despeckle', methods=['POST'])
def despeckle():
    data = request.json
    filename = data.get('filename')
    image_path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(image_path):
        image_path = os.path.join(RAW_DIR, filename)
    img = cv2.imread(image_path)
    denoised = remove_noise(img)
    output_filename = f"denoised_{filename.replace('cleaned_', '').replace('enhanced_', '').replace('denoised_', '')}"
    output_path = os.path.join(PROCESSED_DIR, output_filename)
    save_image(denoised, output_path)
    return jsonify({"success": True, "output": output_filename})

@app.route('/api/save_manual', methods=['POST'])
def save_manual():
    data = request.json
    filename = data.get('filename')
    mask_data = data.get('mask_base64')
    rotation = data.get('rotation', 0)
    
    # Find the current version of the image (raw or already processed)
    # We should use the one currently shown in the UI if possible
    # For now, we take from PROCESSED if it exists, else RAW
    # A better way is to send the 'current_processed_filename' from front
    current_image_name = data.get('current_image', filename)
    image_path = os.path.join(PROCESSED_DIR, current_image_name)
    if not os.path.exists(image_path):
        image_path = os.path.join(RAW_DIR, filename)
        
    original_img = cv2.imread(image_path)
    if rotation != 0:
        if rotation == 90: original_img = cv2.rotate(original_img, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180: original_img = cv2.rotate(original_img, cv2.ROTATE_180)
        elif rotation == 270: original_img = cv2.rotate(original_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    encoded_data = mask_data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    mask = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    result = remove_and_restore(original_img, mask)
    output_filename = f"manual_{filename}"
    output_path = os.path.join(PROCESSED_DIR, output_filename)
    save_image(result, output_path)
    return jsonify({"success": True, "output": output_filename})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
