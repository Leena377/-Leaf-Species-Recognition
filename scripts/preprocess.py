import os
import cv2
import numpy as np
from pathlib import Path

# Define directories
RAW_DIR = Path('../data/raw')
PROCESSED_DIR = Path('../data/processed')

def extract_silhouette(image_path):
    """
    Extracts a binary silhouette of a leaf from a white background.
    """
    # 1. Read image and convert to grayscale
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: Could not read {image_path.name}")
        return None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Blur slightly to remove minor noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Apply Otsu's thresholding
    # We use THRESH_BINARY_INV so the dark leaf becomes white (255) and white bg becomes black (0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 4. Find contours to isolate the main leaf and remove background artifacts
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Assume the largest contour is the leaf
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create a blank mask and draw the solid silhouette
        mask = np.zeros_like(binary)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        return mask
        
    return binary

def main():
    # Ensure processed directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get all jpg images in the raw directory
    image_paths = list(RAW_DIR.glob('*.jpg'))
    print(f"Found {len(image_paths)} images to process. Starting...")
    
    processed_count = 0
    for img_path in image_paths:
        silhouette = extract_silhouette(img_path)
        
        if silhouette is not None:
            # Save to processed folder with the same filename
            save_path = PROCESSED_DIR / img_path.name
            cv2.imwrite(str(save_path), silhouette)
            processed_count += 1
            
    print(f"Successfully processed {processed_count}/{len(image_paths)} images.")

if __name__ == "__main__":
    main()