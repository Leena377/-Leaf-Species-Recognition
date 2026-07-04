"""Two illustrative figures built directly from actual leaf images:

  1. top_confusion_pairs_shape.png       - for Naive Bayes' most-confused
     species pairs, the actual misclassified leaf's silhouette next to a
     reference silhouette of the species it got mistaken for, so the shape
     similarity that caused the error is visible directly.

  2. shape_fails_appearance_succeeds.png - test leaves where BOTH shape-only
     methods (classical Naive Bayes and the silhouette-fed deep SVM) got it
     wrong, but the raw-image deep SVM got it right - i.e. cases only color
     and texture could resolve. Silhouette (why shape failed) next to raw
     color image (what appearance actually saw), side by side.

Reads results/tables/test_predictions.csv (written by classify.py) - no
retraining involved.
"""
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from species import get_label_from_filename
from classify import load_feature_csv

RAW_DIR = Path('../data/raw')
PROCESSED_DIR = Path('../data/processed')
FIGURES_DIR = Path('../results/figures')
PREDICTIONS_FILE = Path('../results/tables/test_predictions.csv')

INK = '#0b0b0b'
INK_SECONDARY = '#52514e'
SURFACE = '#fcfcfb'


def load_predictions():
    with open(PREDICTIONS_FILE) as f:
        return list(csv.DictReader(f))


def nearest_reference_silhouette(misclassified_file, pred_species, classical_features):
    """Among all images labeled pred_species, the one closest in classical
    (Hu + Fourier) feature space to the misclassified leaf - i.e. the actual
    leaf that made Naive Bayes think "this looks like pred_species", rather
    than an arbitrary example of that species."""
    from species import SPECIES_NAMES
    target = SPECIES_NAMES.index(pred_species)
    query = classical_features[misclassified_file]

    best_file, best_dist = None, np.inf
    for fname, feats in classical_features.items():
        if get_label_from_filename(fname) != target:
            continue
        dist = np.linalg.norm(feats - query)
        if dist < best_dist:
            best_file, best_dist = fname, dist
    if best_file is None:
        raise ValueError(f"No reference image found for {pred_species}")
    return best_file


def plot_top_confusion_pairs_shape(n_pairs=6):
    rows = load_predictions()
    pair_counts = Counter()
    pair_example = {}
    for row in rows:
        true_sp, pred_sp = row['true_species'], row['naive_bayes_pred']
        if true_sp == pred_sp:
            continue
        pair_counts[(true_sp, pred_sp)] += 1
        pair_example.setdefault((true_sp, pred_sp), row['filename'])

    top_pairs = pair_counts.most_common(n_pairs)
    classical_features = load_feature_csv(PROCESSED_DIR / 'classical_features.csv')

    fig, axes = plt.subplots(len(top_pairs), 2, figsize=(6, 3.3 * len(top_pairs)),
                              gridspec_kw={'hspace': 0.7})
    fig.patch.set_facecolor(SURFACE)

    for row_idx, ((true_sp, pred_sp), count) in enumerate(top_pairs):
        misclassified_file = pair_example[(true_sp, pred_sp)]
        reference_file = nearest_reference_silhouette(misclassified_file, pred_sp, classical_features)

        ax_true = axes[row_idx, 0]
        ax_pred = axes[row_idx, 1]

        ax_true.imshow(plt.imread(PROCESSED_DIR / misclassified_file), cmap='gray')
        ax_true.set_title(f"True: {true_sp}\n(this leaf, misclassified)", fontsize=9, color=INK)
        ax_true.axis('off')

        ax_pred.imshow(plt.imread(PROCESSED_DIR / reference_file), cmap='gray')
        ax_pred.set_title(f"Predicted as: {pred_sp}\n(closest-matching example)", fontsize=9, color=INK)
        ax_pred.axis('off')

        ax_true.text(1.0, -0.06, f"confused {count}x by Naive Bayes", ha='center',
                     va='top', fontsize=9, color=INK_SECONDARY, style='italic',
                     transform=ax_true.transAxes)

    fig.suptitle("Why Naive Bayes confuses these pairs: near-identical silhouettes",
                 fontsize=13, color=INK, y=1.0)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    plt.savefig(FIGURES_DIR / 'top_confusion_pairs_shape.png', dpi=150,
                facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved top_confusion_pairs_shape.png ({len(top_pairs)} pairs)")


def plot_shape_fails_appearance_succeeds(max_examples=8):
    rows = load_predictions()
    matches = [
        row for row in rows
        if row['naive_bayes_pred'] != row['true_species']
        and row['svm_silhouette_pred'] != row['true_species']
        and row['svm_raw_pred'] == row['true_species']
    ]
    print(f"Found {len(matches)} test leaves where both shape-only methods failed "
          f"but the raw-image SVM succeeded.")
    chosen = matches[:max_examples]

    fig, axes = plt.subplots(len(chosen), 2, figsize=(6, 3 * len(chosen)))
    fig.patch.set_facecolor(SURFACE)
    if len(chosen) == 1:
        axes = axes.reshape(1, 2)

    for row_idx, row in enumerate(chosen):
        fname = row['filename']
        ax_sil = axes[row_idx, 0]
        ax_raw = axes[row_idx, 1]

        ax_sil.imshow(plt.imread(PROCESSED_DIR / fname), cmap='gray')
        ax_sil.set_title(
            f"Shape only\nNB: {row['naive_bayes_pred']}\nSVM-sil: {row['svm_silhouette_pred']}",
            fontsize=8, color=INK)
        ax_sil.axis('off')

        ax_raw.imshow(plt.imread(RAW_DIR / fname))
        ax_raw.set_title(
            f"True: {row['true_species']}\nSVM-raw: {row['svm_raw_pred']} ✓",
            fontsize=8, color=INK)
        ax_raw.axis('off')

    fig.suptitle("Shape says one thing, color/texture says another\n"
                 "(both shape-only methods wrong, raw-image SVM correct)",
                 fontsize=13, color=INK, y=1.0)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(FIGURES_DIR / 'shape_fails_appearance_succeeds.png', dpi=150,
                facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved shape_fails_appearance_succeeds.png ({len(chosen)} examples)")


def main():
    plot_top_confusion_pairs_shape()
    plot_shape_fails_appearance_succeeds()


if __name__ == '__main__':
    main()
