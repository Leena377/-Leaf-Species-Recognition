"""Modern pipeline: frozen MobileNetV2 features for the linear SVM classifiers.

Produces two feature tables so we can contrast "shape only, but learned"
against "full appearance, learned":
  - deep_features_silhouette.csv : from data/processed/*.jpg (binary masks,
    replicated to 3 channels) -> isolates shape, same information budget as
    the classical pipeline, just fed through a CNN instead of hand-crafted
    descriptors.
  - deep_features_raw.csv        : from data/raw/*.jpg (actual color leaf
    photos) -> adds color, texture and venation on top of shape.
"""
import csv
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

RAW_DIR = Path('../data/raw')
PROCESSED_DIR = Path('../data/processed')
IMG_SIZE = 224
BATCH_SIZE = 32


def load_batch_raw(paths):
    batch = []
    for p in paths:
        img = cv2.imread(str(p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        batch.append(img)
    return preprocess_input(np.stack(batch).astype('float32'))


def load_batch_silhouette(paths):
    batch = []
    for p in paths:
        mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        batch.append(rgb)
    return preprocess_input(np.stack(batch).astype('float32'))


def extract_features(model, image_paths, loader):
    features = []
    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i:i + BATCH_SIZE]
        batch = loader(batch_paths)
        feats = model.predict(batch, verbose=0)
        features.append(feats)
        print(f"  {min(i + BATCH_SIZE, len(image_paths))}/{len(image_paths)}", end='\r')
    print()
    return np.concatenate(features, axis=0)


def save_features(out_path, filenames, features):
    headers = ['filename'] + [f'deep_{i}' for i in range(features.shape[1])]
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for name, row in zip(filenames, features):
            writer.writerow([name] + list(row))
    print(f"Saved {len(filenames)} rows to {out_path}")


def main():
    model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

    raw_paths = sorted(RAW_DIR.glob('*.jpg'))
    silhouette_paths = sorted(PROCESSED_DIR.glob('*.jpg'))

    print(f"Extracting deep features from {len(raw_paths)} raw color images...")
    raw_features = extract_features(model, raw_paths, load_batch_raw)
    save_features(PROCESSED_DIR / 'deep_features_raw.csv',
                   [p.name for p in raw_paths], raw_features)

    print(f"Extracting deep features from {len(silhouette_paths)} silhouettes...")
    silhouette_features = extract_features(model, silhouette_paths, load_batch_silhouette)
    save_features(PROCESSED_DIR / 'deep_features_silhouette.csv',
                   [p.name for p in silhouette_paths], silhouette_features)


if __name__ == "__main__":
    main()
