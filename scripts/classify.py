"""Unified comparison of the classical and modern pipelines on one shared split.

Trains and evaluates:
  1. Naive Bayes    on classical features (Hu moments + Fourier descriptors) - shape only
  2. k-NN           on classical features                                    - shape only
  3. Linear SVM     on MobileNetV2 features from silhouettes                  - shape only, learned
  4. Linear SVM     on MobileNetV2 features from raw color images             - shape + appearance, learned

All four are trained/evaluated on the exact same train/test filenames
(data/processed/split.json), so results are directly comparable.
"""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from species import SPECIES_NAMES, get_label_from_filename, NUM_CLASSES

PROCESSED_DIR = Path('../data/processed')
RAW_DIR = Path('../data/raw')
SPLIT_FILE = PROCESSED_DIR / 'split.json'
FIGURES_DIR = Path('../results/figures')
TABLES_DIR = Path('../results/tables')


def load_feature_csv(path):
    """Loads a features CSV into a dict: filename -> np.array of floats."""
    data = {}
    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            data[row[0]] = np.array([float(v) for v in row[1:]])
    return data


def build_xy(feature_dict, filenames):
    X = np.stack([feature_dict[f] for f in filenames])
    y = np.array([get_label_from_filename(f) for f in filenames])
    return X, y


def evaluate_model(model, X_test, y_test, model_name):
    """Computes accuracy/per-class accuracy, saves a confusion matrix figure."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    cm = confusion_matrix(y_test, predictions, labels=range(NUM_CLASSES))
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_acc = np.diag(cm) / cm.sum(axis=1)
    per_class_acc = np.nan_to_num(per_class_acc)
    macro_acc = per_class_acc.mean()

    print(f"{model_name}: overall accuracy = {accuracy * 100:.2f}%, "
          f"macro per-class accuracy = {macro_acc * 100:.2f}%")

    # Per-class accuracy table
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    slug = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    with open(TABLES_DIR / f'{slug}_per_class_accuracy.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['species', 'accuracy', 'support'])
        for i, name in enumerate(SPECIES_NAMES):
            writer.writerow([name, f"{per_class_acc[i]:.4f}", int(cm[i].sum())])

    # Confusion matrix figure
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=SPECIES_NAMES)
    fig, ax = plt.subplots(figsize=(16, 16))
    disp.plot(ax=ax, cmap='viridis', colorbar=False, xticks_rotation=90)
    ax.set_title(f'{model_name} Confusion Matrix (accuracy={accuracy * 100:.1f}%)')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'{slug}_cm.png', dpi=150)
    plt.close(fig)

    return {
        'model': model_name,
        'accuracy': accuracy,
        'macro_per_class_accuracy': macro_acc,
        'predictions': predictions,
    }


def train_linear_svm(X_train, y_train):
    pipeline = make_pipeline(StandardScaler(), LinearSVC(max_iter=20000, dual='auto'))
    param_grid = {'linearsvc__C': [0.001, 0.01, 0.1, 1, 10]}
    search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"  best C = {search.best_params_['linearsvc__C']}")
    return search.best_estimator_


def save_example_grid(filenames, true_labels, results, correct_counts, mode, out_path, n=12):
    """Saves a grid of thumbnails: 'easy' = most agreed-correct, 'hard' = most agreed-wrong."""
    order = np.argsort(-correct_counts if mode == 'easy' else correct_counts)
    chosen = order[:n]

    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
    axes = axes.flatten()

    method_labels = ['NB', 'kNN', 'SVM-sil', 'SVM-raw']
    for ax_idx, idx in enumerate(chosen):
        ax = axes[ax_idx]
        fname = filenames[idx]
        img = plt.imread(RAW_DIR / fname)
        ax.imshow(img)
        ax.axis('off')
        true_name = SPECIES_NAMES[true_labels[idx]]
        lines = [f"True: {true_name}"]
        for m_label, res in zip(method_labels, results):
            pred_name = SPECIES_NAMES[res['predictions'][idx]]
            mark = '✓' if res['predictions'][idx] == true_labels[idx] else '✗'
            lines.append(f"{m_label}: {pred_name} {mark}")
        ax.set_title('\n'.join(lines), fontsize=7)

    for ax_idx in range(len(chosen), len(axes)):
        axes[ax_idx].axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    with open(SPLIT_FILE) as f:
        split = json.load(f)
    train_files, test_files = split['train'], split['test']

    classical = load_feature_csv(PROCESSED_DIR / 'classical_features.csv')
    deep_silhouette = load_feature_csv(PROCESSED_DIR / 'deep_features_silhouette.csv')
    deep_raw = load_feature_csv(PROCESSED_DIR / 'deep_features_raw.csv')

    y_test_ref = np.array([get_label_from_filename(f) for f in test_files])

    results = []

    # 1. Naive Bayes on classical features
    X_train, y_train = build_xy(classical, train_files)
    X_test, y_test = build_xy(classical, test_files)
    nb = GaussianNB().fit(X_train, y_train)
    results.append(evaluate_model(nb, X_test, y_test, "Naive Bayes (classical)"))

    # 2. k-NN on classical features
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
    results.append(evaluate_model(knn, X_test, y_test, "k-NN (classical)"))

    # 3. Linear SVM on deep silhouette features
    X_train, y_train = build_xy(deep_silhouette, train_files)
    X_test, y_test = build_xy(deep_silhouette, test_files)
    svm_sil = train_linear_svm(X_train, y_train)
    results.append(evaluate_model(svm_sil, X_test, y_test, "Linear SVM (deep silhouette)"))

    # 4. Linear SVM on deep raw-image features
    X_train, y_train = build_xy(deep_raw, train_files)
    X_test, y_test = build_xy(deep_raw, test_files)
    svm_raw = train_linear_svm(X_train, y_train)
    results.append(evaluate_model(svm_raw, X_test, y_test, "Linear SVM (deep raw)"))

    # Comparison table
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    with open(TABLES_DIR / 'model_comparison.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'overall_accuracy', 'macro_per_class_accuracy'])
        for r in results:
            writer.writerow([r['model'], f"{r['accuracy']:.4f}", f"{r['macro_per_class_accuracy']:.4f}"])
    print("Saved model comparison table.")

    # Easy / hard examples (agreement across all 4 methods)
    correct_counts = np.zeros(len(test_files), dtype=int)
    for r in results:
        correct_counts += (r['predictions'] == y_test_ref).astype(int)

    save_example_grid(test_files, y_test_ref, results, correct_counts, 'easy',
                       FIGURES_DIR / 'easy_examples.png')
    save_example_grid(test_files, y_test_ref, results, correct_counts, 'hard',
                       FIGURES_DIR / 'hard_examples.png')
    print("Saved easy/hard example grids.")

    # Per-image predictions for every method, so downstream analysis scripts
    # don't need to retrain anything.
    with open(TABLES_DIR / 'test_predictions.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'true_species', 'naive_bayes_pred',
                          'knn_pred', 'svm_silhouette_pred', 'svm_raw_pred'])
        for i, fname in enumerate(test_files):
            row = [fname, SPECIES_NAMES[y_test_ref[i]]]
            row += [SPECIES_NAMES[r['predictions'][i]] for r in results]
            writer.writerow(row)
    print("Saved per-image predictions.")


if __name__ == "__main__":
    main()
