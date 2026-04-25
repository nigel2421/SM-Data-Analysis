import os
import argparse
from src.preprocess import clean_background
from src.handwriting_ai import get_handwriting_mask
from src.inpainter import remove_and_restore
from src.utils import save_image, list_raw_images

def process_pipeline(image_path):
    print(f"Processing: {image_path}")
    
    # 1. Image Normalization
    print("Step 1: Cleaning background...")
    norm_img = clean_background(image_path)
    
    # Save intermediate normalized image for debugging
    filename = os.path.basename(image_path)
    temp_norm_path = os.path.join('data', 'processed', f"norm_{filename}")
    save_image(norm_img, temp_norm_path)
    
    # 2. Handwriting Identification
    print("Step 2: Detecting handwriting via NanoNets...")
    try:
        mask = get_handwriting_mask(image_path)
        mask_path = os.path.join('data', 'masks', f"mask_{filename}")
        save_image(mask, mask_path)
    except Exception as e:
        print(f"Skipping Smart Removal: {e}")
        return
        
    # 3. Smart Removal & Inpainting
    print("Step 3: Inpainting and restoring...")
    final_img = remove_and_restore(norm_img, mask)
    
    # 4. Save Final Output
    output_path = os.path.join('data', 'processed', f"cleaned_{filename}")
    save_image(final_img, output_path)
    print(f"Done! Saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map Cleaner Pipeline")
    parser.add_argument("--image", help="Path to a single image to process")
    parser.add_argument("--batch", action="store_true", help="Process all images in data/raw")
    
    args = parser.parse_args()
    
    if args.image:
        process_pipeline(args.image)
    elif args.batch:
        raw_images = list_raw_images()
        if not raw_images:
            print("No images found in data/raw")
        for img_path in raw_images:
            process_pipeline(img_path)
    else:
        parser.print_help()
