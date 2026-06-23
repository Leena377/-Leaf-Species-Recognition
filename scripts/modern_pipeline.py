# Modern pipeline: MobileNetV2 features + linear SVM
# Uses raw colour images (not the silhouettes), same split as classify.py

import csv
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

BASE_DIR    = Path(__file__).resolve().parent.parent
RAW_DIR     = BASE_DIR / "data" / "raw"
FIGURES_DIR = BASE_DIR / "results" / "figures"
TABLES_DIR  = BASE_DIR / "results" / "tables"

IMG_SIZE = 224

# same label map as classify.py
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
    "Phyllostachys edulis", "Aesculus chinensis", "Berberis anhweiensis",
    "Cercis chinensis", "Indigofera tinctoria", "Acer palmatum",
    "Phoebe nanmu", "Kalopanax septemlobus", "Cinnamomum japonicum",
    "Koelreuteria paniculata", "Ilex macrocarpa", "Pittosporum tobira",
    "Chimonanthus praecox", "Cinnamomum camphora", "Viburnum awabuki",
    "Osmanthus fragrans", "Cedrus deodara", "Ginkgo biloba",
    "Lagerstroemia indica", "Nerium oleander", "Podocarpus macrophyllus",
    "Prunus serrulata", "Ligustrum lucidum", "Toona sinensis",
    "Prunus persica", "Manglietia fordiana", "Acer buergerianum",
    "Mahonia bealei", "Magnolia grandiflora", "Populus x canadensis",
    "Liriodendron chinense", "Citrus reticulata",
]


def get_label(filename):
    image_id = int("".join(filter(str.isdigit, Path(filename).stem)))
    for start, end, label in FLAVIA_RANGES:
        if start <= image_id <= end:
            return label
    return -1


def load_images(paths):
    images = []
    kept = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"Warning: could not read {p.name}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img.astype(np.float32))
        kept.append(p)
    return np.stack(images), kept


def main():
    import tensorflow as tf

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # load all raw images (sorted so the split is reproducible)
    all_paths = sorted(RAW_DIR.glob("*.jpg"))
    print(f"Found {len(all_paths)} images")

    labels_all = np.array([get_label(p.name) for p in all_paths])
    all_paths  = [p for p, l in zip(all_paths, labels_all) if l != -1]
    labels_all = labels_all[labels_all != -1]

    print("Loading images...")
    images, kept_paths = load_images(all_paths)
    names  = [p.name for p in kept_paths]
    labels = np.array([get_label(n) for n in names])
    print(f"Loaded {len(images)} images")

    # same split as classify.py
    idx = np.arange(len(images))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=labels)

    # frozen MobileNetV2 as feature extractor
    print("Building MobileNetV2 feature extractor...")
    base = tf.keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.trainable = False

    print("Extracting features (may take a bit)...")
    # preprocess_input rescales to [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(images.copy())
    features = base.predict(x, batch_size=32, verbose=1)
    print(f"Feature shape: {features.shape}")  # should be (1907, 1280)

    X_train = features[train_idx]
    X_test  = features[test_idx]
    y_train = labels[train_idx]
    y_test  = labels[test_idx]
    test_names = [names[i] for i in test_idx]

    # scale before SVM
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print("Training LinearSVC...")
    svm = LinearSVC(C=1.0, max_iter=5000, random_state=42)
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc * 100:.2f}%")

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(14, 14))
    ConfusionMatrixDisplay(cm).plot(ax=ax, cmap="viridis", colorbar=False)
    ax.set_title("MobileNetV2 + LinearSVM Confusion Matrix")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "mobilenet_svm_cm.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("Saved confusion matrix")

    # per-class accuracy
    accs, counts = [], []
    for c in range(len(SPECIES_NAMES)):
        mask = y_test == c
        if mask.sum() == 0:
            accs.append(None)
            counts.append(0)
        else:
            accs.append(accuracy_score(y_test[mask], y_pred[mask]))
            counts.append(int(mask.sum()))

    with open(TABLES_DIR / "mobilenet_svm_per_class.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "species", "accuracy", "n_test"])
        for i, (a, n) in enumerate(zip(accs, counts)):
            w.writerow([i, SPECIES_NAMES[i], f"{a:.4f}" if a is not None else "N/A", n])

    # bar chart per-class
    valid_idx = [i for i, a in enumerate(accs) if a is not None]
    valid_acc = [accs[i] for i in valid_idx]
    short_names = [SPECIES_NAMES[i].split()[-1] for i in valid_idx]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(valid_acc)), valid_acc, color="steelblue")
    ax.set_xticks(range(len(valid_acc)))
    ax.set_xticklabels(short_names, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class accuracy - MobileNetV2 + LinearSVM")
    ax.axhline(np.mean(valid_acc), color="tomato", linestyle="--", label=f"mean = {np.mean(valid_acc):.2f}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "mobilenet_svm_per_class.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("Saved per-class accuracy")

    # easy/hard examples
    # confidence = decision score for the predicted class
    decision   = svm.decision_function(X_test)
    confidence = decision[np.arange(len(y_pred)), y_pred]

    correct = y_test == y_pred
    easy_idx = np.where(correct)[0]
    easy_idx = easy_idx[np.argsort(confidence[easy_idx])[::-1]][:4]

    wrong_idx = np.where(~correct)[0]
    if len(wrong_idx) == 0:
        wrong_idx = np.argsort(confidence)[:4]  # shouldn't happen but just in case
    else:
        wrong_idx = wrong_idx[np.argsort(confidence[wrong_idx])][:4]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for j, i in enumerate(easy_idx):
        img = cv2.imread(str(RAW_DIR / test_names[i]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[0, j].imshow(img)
        axes[0, j].set_title(
            f"True: {SPECIES_NAMES[y_test[i]].split()[-1]}\n"
            f"Pred: {SPECIES_NAMES[y_pred[i]].split()[-1]}\nconf={confidence[i]:.2f}",
            fontsize=7, color="green"
        )
        axes[0, j].axis("off")

    for j, i in enumerate(wrong_idx):
        img = cv2.imread(str(RAW_DIR / test_names[i]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[1, j].imshow(img)
        axes[1, j].set_title(
            f"True: {SPECIES_NAMES[y_test[i]].split()[-1]}\n"
            f"Pred: {SPECIES_NAMES[y_pred[i]].split()[-1]}\nconf={confidence[i]:.2f}",
            fontsize=7, color="red"
        )
        axes[1, j].axis("off")

    axes[0, 0].set_ylabel("Easy", fontsize=10)
    axes[1, 0].set_ylabel("Hard", fontsize=10)
    plt.suptitle("Easy (top) vs Hard (bottom) examples")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "mobilenet_svm_easy_hard.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("Saved easy/hard examples")

    print(f"\nDone. Accuracy: {acc*100:.2f}%")


if __name__ == "__main__":
    main()
