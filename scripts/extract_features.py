import cv2
import numpy as np
import csv
from pathlib import Path

# Directories
PROCESSED_DIR = Path('../data/processed')
FEATURES_FILE = Path('../data/processed/classical_features.csv')

# Configuration
NUM_FOURIER_DESCRIPTORS = 16  # Keep the first 16 low-frequency components

def extract_hu_moments(binary_image):
    """Calculates the 7 Hu Moments and applies a log transform to handle large scale variations."""
    moments = cv2.moments(binary_image)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # Log transform to compress the vast range of values
    # We use a small epsilon to avoid log(0) errors
    epsilon = 1e-10
    log_hu = -1 * np.sign(hu_moments) * np.log10(np.abs(hu_moments) + epsilon)
    return list(log_hu)

def extract_fourier_descriptors(binary_image):
    """Calculates translation, rotation, and scale-invariant Fourier Descriptors."""
    # 1. Find the outer contour
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return [0.0] * NUM_FOURIER_DESCRIPTORS
        
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 2. Convert contour to a sequence of complex numbers: z(t) = x(t) + j*y(t)
    # The contour array shape is (N, 1, 2)
    contour_complex = np.empty(largest_contour.shape[0], dtype=complex)
    contour_complex.real = largest_contour[:, 0, 0]
    contour_complex.imag = largest_contour[:, 0, 1]
    
    # 3. Apply Discrete Fourier Transform
    fourier_result = np.fft.fft(contour_complex)
    
    # 4. Truncate to keep only the lowest frequencies (general shape, discarding noise)
    # The lowest frequencies are at the start of the array
    fourier_truncated = fourier_result[:NUM_FOURIER_DESCRIPTORS + 1]
    
    # 5. Achieve Invariance
    # Magnitude removes starting-point dependency (rotation invariance)
    magnitudes = np.abs(fourier_truncated)
    
    # Drop the DC component (magnitudes[0]) because it only represents position (translation invariance)
    # Divide the rest by the magnitude of the second component (scale invariance)
    if magnitudes[1] != 0:
        descriptors = magnitudes[1:] / magnitudes[1]
    else:
        descriptors = np.zeros(NUM_FOURIER_DESCRIPTORS)
        
    return list(descriptors)

def main():
    image_paths = list(PROCESSED_DIR.glob('*.jpg'))
    print(f"Found {len(image_paths)} processed silhouettes. Extracting features...")
    
    # Prepare CSV headers
    headers = ['filename'] 
    headers += [f'hu_{i}' for i in range(1, 8)]
    headers += [f'fd_{i}' for i in range(1, NUM_FOURIER_DESCRIPTORS + 1)]
    
    processed_count = 0
    
    with open(FEATURES_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for img_path in image_paths:
            # Load as grayscale since it's already a binary mask
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                hu_feats = extract_hu_moments(img)
                fd_feats = extract_fourier_descriptors(img)
                
                # Combine filename and features into one row
                row = [img_path.name] + hu_feats + fd_feats
                writer.writerow(row)
                processed_count += 1
                
    print(f"Feature extraction complete! Saved {processed_count} rows to {FEATURES_FILE.name}")

if __name__ == "__main__":
    main()