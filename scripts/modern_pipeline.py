"""modern_pipeline.py

Modern pipeline: frozen MobileNetV2 features + linear SVM.

Loads raw colour images from data/raw/, resizes to 224x224, runs them
through a frozen MobileNetV2 (ImageNet weights, GAP output = 1280-dim),
then trains a LinearSVC on those embeddings.

Same stratified 80/20 split as classify.py (random_state=42, stratify=y).

Outputs saved to results/figures/ and results/tables/:
  - mobilenet_svm_cm.png          confusion matrix
  - mobilenet_svm_per_class.csv   per-class accuracy table
  - mobilenet_svm_per_class.png   per-class accuracy bar chart
  - mobilenet_svm_easy_hard.png   4 easy + 4 hard examples
"""

import csv
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent.parent
RAW_DIR    = BASE_DIR / "data" / "raw"
FIGURES_DIR = BASE_DIR / "results" / "figures"
TABLES_DIR  = BASE_DIR / "results" / "tables"

IMG_SIZE = 224  # MobileNetV2 default input

# ---------------------------------------------------------------------------
# Flavia label map  (same as classify.py)
# ---------------------------------------------------------------------------
FLAVIA_RANGES = [
    (1001, 1059,  0),  # Phyllostachys edulis
    (1060, 1122,  1),  # Aesculus chinensis
    (1552, 1616,  2),  # Berberis anhweiensis
    (1123, 1194,  3),  # Cercis chinensis
    (1195, 1267,  4),  # Indigofera tinctoria
    (1268, 1323,  5),  # Acer palmatum
    (1324, 1385,  6),  # Phoebe nanmu
    (1386, 1437,  7),  # Kalopanax septemlobus
    (1497, 1551,  8),  # Cinnamomum japonicum
    (1438, 1496,  9),  # Koelreuteria paniculata
    (2001, 2050, 10),  # Ilex macrocarpa
    (2051, 2113, 11),  # Pittosporum tobira
    (2114, 2165, 12),  # Chimonanthus praecox
    (2166, 2230, 13),  # Cinnamomum camphora
    (2231, 2290, 14),  # Viburnum awabuki
    (2291, 2346, 15),  # Osmanthus fragrans
    (2347, 2423, 16),  # Cedrus deodara
    (2424, 2485, 17),  # Ginkgo biloba
    (2486, 2546, 18),  # Lagerstroemia indica
    (2547, 2612, 19),  # Nerium oleander
    (2616, 2675, 20),  # Podocarpus macrophyllus
    (3001, 3055, 21),  # Prunus serrulata
    (3056, 3110, 22),  # Ligustrum lucidum
    (3111, 3175, 23),  # Toona sinensis
    (3176, 3229, 24),  # Prunus persica
    (3230, 3281, 25),  # Manglietia fordiana
    (3282, 3334, 26),  # Acer buergerianum
    (3335, 3389, 27),  # Mahonia bealei
    (3390, 3446, 28),  # Magnolia grandiflora
    (3447, 3510, 29),  # Populus x canadensis
    (3511, 3563, 30),  # Liriodendron chinense
    (3566, 3621, 31),  # Citrus reticulata
]

SPECIES_NAMES = [
    "Phyllostachys edulis",   # 0
    "Aesculus chinensis",     # 1
    "Berberis anhweiensis",   # 2
    "Cercis chinensis",       # 3
    "Indigofera tinctoria",   # 4
    "Acer palmatum",          # 5
    "Phoebe nanmu",           # 6
    "Kalopanax septemlobus",  # 7
    "Cinnamomum japonicum",   # 8
    "Koelreuteria paniculata",# 9
    "Ilex macrocarpa",        # 10
    "Pittosporum tobira",     # 11
    "Chimonanthus praecox",   # 12
    "Cinnamomum camphora",    # 13
    "Viburnum awabuki",       # 14
    "Osmanthus fragrans",     # 15
    "Cedrus deodara",         # 16
    "Ginkgo biloba",          # 17
    "Lagerstroemia indica",   # 18
    "Nerium oleander",        # 19
    "Podocarpus macrophyllus",# 20
    "Prunus serrulata",       # 21
    "Ligustrum lucidum",      # 22
    "Toona sinensis",         # 23
    "Prunus persica",         # 24
    "Manglietia fordiana",    # 25
    "Acer buergerianum",      # 26
    "Mahonia bealei",         # 27
    "Magnolia grandiflora",   # 28
    "Populus x canadensis",   # 29
    "Liriodendron chinense",  # 30
    "Citrus reticulata",      # 31
]


def get_label(filename: str) -> int:
    image_id = int("".join(filter(str.isdigit, Path(filename).stem)))
    for start, end, label in FLAVIA_RANGES:
        if start <= image_id <= end:
            return label
    print(f"  Warning: {filename} outside known ranges, skipping.")
    return -1


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_images(image_paths):
    """Load, resize to 224x224, convert BGR->RGB. Returns float32 array."""
    images, kept_paths = [], []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"  Could not read {p.name}, skipping.")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        images.append(img.astype(np.float32))
        kept_paths.append(p)
    return np.stack(images, axis=0), kept_paths


# ---------------------------------------------------------------------------
# MobileNetV2 feature extraction
# ---------------------------------------------------------------------------

def build_feature_extractor():
    """Frozen MobileNetV2 with global average pooling, output 1280-dim."""
    import tensorflow as tf
    model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    model.trainable = False
    return model


def extract_features(images_f32, model, batch_size=32):
    """Run images through frozen MobileNetV2. images_f32 in [0, 255] range."""
    import tensorflow as tf
    # MobileNetV2 expects input in [-1, 1]
    preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(images_f32.copy())
    features = model.predict(preprocessed, batch_size=batch_size, verbose=1)
    return features  # shape (N, 1280)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 14))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="viridis", colorbar=False)
    ax.set_title(title, fontsize=13, pad=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def per_class_accuracy(y_true, y_pred, n_classes):
    accs, counts = [], []
    for c in range(n_classes):
        mask = y_true == c
        if mask.sum() == 0:
            accs.append(None)
            counts.append(0)
        else:
            accs.append(accuracy_score(y_true[mask], y_pred[mask]))
            counts.append(int(mask.sum()))
    return accs, counts


def plot_per_class_bar(accs, save_path):
    valid = [(i, a) for i, a in enumerate(accs) if a is not None]
    indices, values = zip(*valid)
    short_names = [SPECIES_NAMES[i].split()[-1] for i in indices]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(len(values)), values, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(short_names, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class accuracy — MobileNetV2 + LinearSVM")
    ax.axhline(np.mean(values), color="tomato", linestyle="--", linewidth=1.2, label=f"mean={np.mean(values):.2f}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def save_per_class_csv(accs, counts, save_path):
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "species", "accuracy", "n_test"])
        for i, (acc, n) in enumerate(zip(accs, counts)):
            acc_str = f"{acc:.4f}" if acc is not None else "N/A"
            writer.writerow([i, SPECIES_NAMES[i], acc_str, n])
    print(f"  Saved: {save_path.name}")


def plot_easy_hard(
    test_image_paths, y_test, y_pred, confidence, raw_dir, save_path, n=4
):
    """Show n easy (correct, high margin) and n hard (wrong) examples."""
    correct = y_test == y_pred
    wrong   = ~correct

    # Easy: correct predictions sorted by descending confidence
    easy_candidates = np.where(correct)[0]
    easy_candidates = easy_candidates[np.argsort(confidence[easy_candidates])[::-1]][:n]

    # Hard: wrong predictions sorted by ascending confidence (least certain)
    hard_candidates = np.where(wrong)[0]
    if len(hard_candidates) == 0:
        hard_candidates = np.argsort(confidence)[:n]  # fallback: lowest confidence overall
    else:
        hard_candidates = hard_candidates[np.argsort(confidence[hard_candidates])][:n]

    rows = 2
    cols = max(len(easy_candidates), len(hard_candidates), 1)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 4))
    if cols == 1:
        axes = axes.reshape(rows, 1)

    def _plot_row(row_idx, candidates, row_label):
        for j, idx in enumerate(candidates):
            img = cv2.imread(str(raw_dir / test_image_paths[idx]))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            ax = axes[row_idx, j]
            ax.imshow(img)
            true_name  = SPECIES_NAMES[y_test[idx]].split()[-1]
            pred_name  = SPECIES_NAMES[y_pred[idx]].split()[-1]
            color = "green" if y_test[idx] == y_pred[idx] else "red"
            ax.set_title(
                f"{row_label}\nTrue: {true_name}\nPred: {pred_name}\nConf: {confidence[idx]:.2f}",
                fontsize=7.5, color=color
            )
            ax.axis("off")
        for j in range(len(candidates), cols):
            axes[row_idx, j].axis("off")

    _plot_row(0, easy_candidates, "EASY")
    _plot_row(1, hard_candidates, "HARD")

    plt.suptitle("Easy vs Hard examples — MobileNetV2 + LinearSVM", fontsize=11)
    plt.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Collect image paths & labels -----------------------------------
    print("Loading image list...")
    all_paths = sorted(RAW_DIR.glob("*.jpg"))
    print(f"  Found {len(all_paths)} images in {RAW_DIR}")

    filenames = [p.name for p in all_paths]
    labels_raw = np.array([get_label(f) for f in filenames])

    valid_mask = labels_raw != -1
    all_paths  = [p for p, v in zip(all_paths, valid_mask) if v]
    labels_raw = labels_raw[valid_mask]
    filenames  = [p.name for p in all_paths]
    print(f"  {len(all_paths)} images with known labels across {len(set(labels_raw))} classes")

    # --- 2. Load images ----------------------------------------------------
    print("\nLoading images (resize to 224x224)...")
    images, kept_paths = load_images(all_paths)
    kept_names = [p.name for p in kept_paths]
    labels = np.array([get_label(n) for n in kept_names])
    print(f"  Loaded {len(images)} images, shape {images.shape}")

    # --- 3. Stratified split (same seed as classical pipeline) -------------
    idx = np.arange(len(images))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"\nSplit: {len(train_idx)} train / {len(test_idx)} test")

    # --- 4. MobileNetV2 feature extraction ---------------------------------
    print("\nBuilding frozen MobileNetV2 feature extractor...")
    model = build_feature_extractor()
    model.summary(line_length=80, print_fn=lambda x: print("  " + x))

    print("\nExtracting features (this may take a minute)...")
    features = extract_features(images, model, batch_size=32)
    print(f"  Features shape: {features.shape}")  # (N, 1280)

    X_train = features[train_idx]
    X_test  = features[test_idx]
    y_train = labels[train_idx]
    y_test  = labels[test_idx]

    test_names = [kept_names[i] for i in test_idx]

    # --- 5. Standardise features (helps LinearSVC convergence) -------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # --- 6. Train LinearSVC ------------------------------------------------
    print("\nTraining LinearSVC (C=1.0)...")
    svm = LinearSVC(C=1.0, max_iter=5000, random_state=42)
    svm.fit(X_train, y_train)

    # --- 7. Evaluate -------------------------------------------------------
    y_pred = svm.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n  LinearSVC + MobileNetV2 test accuracy: {acc * 100:.2f}%")

    # Confusion matrix
    print("\nSaving confusion matrix...")
    plot_confusion_matrix(
        y_test, y_pred,
        title="MobileNetV2 + LinearSVM — Confusion Matrix",
        save_path=FIGURES_DIR / "mobilenet_svm_cm.png",
    )

    # Per-class accuracy
    print("Computing per-class accuracy...")
    n_classes = len(SPECIES_NAMES)
    accs, counts = per_class_accuracy(y_test, y_pred, n_classes)

    save_per_class_csv(accs, counts, TABLES_DIR / "mobilenet_svm_per_class.csv")
    plot_per_class_bar(accs, FIGURES_DIR / "mobilenet_svm_per_class.png")

    # Easiest / hardest examples
    print("Saving easy/hard example plot...")
    # confidence = max OVR decision score (higher = more certain)
    decision    = svm.decision_function(X_test)       # (n_test, n_classes)
    confidence  = decision[np.arange(len(y_pred)), y_pred]

    plot_easy_hard(
        test_names, y_test, y_pred, confidence,
        raw_dir=RAW_DIR,
        save_path=FIGURES_DIR / "mobilenet_svm_easy_hard.png",
        n=4,
    )

    # --- 8. Summary --------------------------------------------------------
    valid_accs = [a for a in accs if a is not None]
    print("\n" + "=" * 55)
    print("  MODERN PIPELINE SUMMARY")
    print("=" * 55)
    print(f"  Backbone       : MobileNetV2 (frozen, ImageNet weights)")
    print(f"  Feature dim    : {features.shape[1]}")
    print(f"  Classifier     : LinearSVC  C=1.0")
    print(f"  Test accuracy  : {acc * 100:.2f}%")
    print(f"  Mean per-class : {np.mean(valid_accs) * 100:.2f}%")
    best_idx  = int(np.argmax([a if a is not None else -1 for a in accs]))
    worst_idx = int(np.argmin([a if a is not None else 2  for a in accs]))
    print(f"  Best class     : {SPECIES_NAMES[best_idx]} ({accs[best_idx]*100:.1f}%)")
    print(f"  Worst class    : {SPECIES_NAMES[worst_idx]} ({accs[worst_idx]*100:.1f}%)")
    print("=" * 55)
    print("\nDone. Results in results/figures/ and results/tables/")


if __name__ == "__main__":
    main()
