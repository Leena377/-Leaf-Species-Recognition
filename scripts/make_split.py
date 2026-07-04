"""Builds one stratified train/test split shared by every downstream pipeline.

Writes data/processed/split.json = {"train": [filenames...], "test": [filenames...]}
so classical features, deep features (raw + silhouette), and all classifiers
are compared on exactly the same held-out images.
"""
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from species import get_label_from_filename

RAW_DIR = Path('../data/raw')
SPLIT_FILE = Path('../data/processed/split.json')


def main():
    filenames = sorted(p.name for p in RAW_DIR.glob('*.jpg'))
    labels = [get_label_from_filename(f) for f in filenames]

    train_files, test_files = train_test_split(
        filenames, test_size=0.2, random_state=42, stratify=labels
    )

    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_FILE, 'w') as f:
        json.dump({"train": sorted(train_files), "test": sorted(test_files)}, f, indent=2)

    print(f"Total images: {len(filenames)}")
    print(f"Train: {len(train_files)}  Test: {len(test_files)}")
    print(f"Saved split to {SPLIT_FILE}")


if __name__ == "__main__":
    main()
