"""Extra analysis figures layered on top of the four-method comparison:

  1. per_class_accuracy_heatmap.png - all 4 methods x 32 species, one view
  2. feature_space_pca.png          - 2D PCA of classical vs. deep feature
                                      spaces, a handful of species highlighted
  3. top_confusions_naive_bayes.png - which species pairs the weakest
                                      (classical) model actually confuses

Reuses the split/features already produced by make_split.py,
extract_features.py and extract_deep_features.py - no retraining except a
cheap refit of GaussianNB to recover its confusion matrix.
"""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

from species import SPECIES_NAMES, get_label_from_filename, NUM_CLASSES
from classify import load_feature_csv, build_xy

PROCESSED_DIR = Path('../data/processed')
FIGURES_DIR = Path('../results/figures')
SPLIT_FILE = PROCESSED_DIR / 'split.json'

# Validated palette (dataviz skill reference palette.md)
INK = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
GRID = '#e1e0d9'
SURFACE = '#fcfcfb'
CATEGORICAL = ['#2a78d6', '#1baf7a', '#eda100', '#008300',
               '#4a3aa7', '#e34948', '#e87ba4']  # blue,aqua,yellow,green,violet,red,magenta
OTHER_GRAY = '#c3c2b7'

SEQ_BLUE_STEPS = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec',
                  '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab',
                  '#184f95', '#104281', '#0d366b']
SEQ_BLUE = LinearSegmentedColormap.from_list('seq_blue', SEQ_BLUE_STEPS)


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_MUTED)


def load_per_class(slug):
    rows = {}
    with open(FIGURES_DIR.parent / 'tables' / f'{slug}_per_class_accuracy.csv') as f:
        for row in csv.DictReader(f):
            rows[row['species']] = float(row['accuracy'])
    return rows


def plot_per_class_heatmap():
    methods = [
        ('Naive Bayes\n(classical)', 'naive_bayes_classical'),
        ('k-NN\n(classical)', 'k-nn_classical'),
        ('Linear SVM\n(deep silhouette)', 'linear_svm_deep_silhouette'),
        ('Linear SVM\n(deep raw)', 'linear_svm_deep_raw'),
    ]
    per_class = {label: load_per_class(slug) for label, slug in methods}

    order = sorted(SPECIES_NAMES, key=lambda s: per_class['Naive Bayes\n(classical)'][s])
    matrix = np.array([[per_class[label][s] for label, _ in methods] for s in order])

    fig, ax = plt.subplots(figsize=(6.5, 11))
    fig.patch.set_facecolor(SURFACE)
    im = ax.imshow(matrix, cmap=SEQ_BLUE, vmin=0, vmax=1, aspect='auto')

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([label for label, _ in methods], color=INK, fontsize=9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, color=INK, fontsize=8)
    ax.set_title('Per-class accuracy, all four methods\n(rows sorted by Naive Bayes accuracy)',
                  color=INK, fontsize=12, pad=12)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            text_color = INK if val < 0.6 else '#ffffff'
            ax.text(j, i, f'{val * 100:.0f}', ha='center', va='center',
                     fontsize=7, color=text_color)

    ax.set_xticks(np.arange(-0.5, len(methods), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
    ax.grid(which='minor', color=SURFACE, linewidth=1.5)
    ax.tick_params(which='minor', length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label('Accuracy', color=INK_SECONDARY)
    cbar.ax.yaxis.set_tick_params(color=INK_MUTED, labelcolor=INK_MUTED)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'per_class_accuracy_heatmap.png', dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print('Saved per_class_accuracy_heatmap.png')


def plot_feature_space_pca():
    highlight_species = [
        'Cedrus deodara',        # shape-distinctive, easy for every method
        'Ginkgo biloba',         # shape-distinctive, easy for every method
        'Nerium oleander',       # 0% classical -> 100% raw-image SVM
        'Cinnamomum camphora',   # 0% classical -> 100% raw-image SVM
        'Prunus persica',        # 0% classical -> 91% raw-image SVM
        'Viburnum awabuki',      # 0% classical -> 100% raw-image SVM
        'Ilex macrocarpa',       # the exception: raw-image SVM < silhouette SVM
    ]
    color_map = dict(zip(highlight_species, CATEGORICAL))

    with open(SPLIT_FILE) as f:
        split = json.load(f)
    all_files = sorted(split['train'] + split['test'])

    sources = [
        ('Classical\n(Hu + Fourier, 23-d)', 'classical_features.csv'),
        ('Deep, silhouette\n(MobileNetV2, 1280-d)', 'deep_features_silhouette.csv'),
        ('Deep, raw image\n(MobileNetV2, 1280-d)', 'deep_features_raw.csv'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.patch.set_facecolor(SURFACE)

    for ax, (title, filename) in zip(axes, sources):
        feats = load_feature_csv(PROCESSED_DIR / filename)
        X, y = build_xy(feats, all_files)
        names = np.array([SPECIES_NAMES[label] for label in y])

        X_scaled = StandardScaler().fit_transform(X)
        coords = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

        style_axes(ax)
        other = ~np.isin(names, highlight_species)
        ax.scatter(coords[other, 0], coords[other, 1], s=10, color=OTHER_GRAY,
                   alpha=0.5, linewidths=0, label='Other (25 species)')
        for species in highlight_species:
            mask = names == species
            ax.scatter(coords[mask, 0], coords[mask, 1], s=24,
                       color=color_map[species], alpha=0.95,
                       edgecolors=INK, linewidths=0.3)

        ax.set_title(title, color=INK, fontsize=11)
        ax.set_xlabel('PC1', color=INK_SECONDARY, fontsize=9)
        ax.set_ylabel('PC2', color=INK_SECONDARY, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    legend_handles = [Line2D([0], [0], marker='o', linestyle='', color=OTHER_GRAY,
                              alpha=0.6, markersize=7, label='Other (25 species)')]
    legend_handles += [Line2D([0], [0], marker='o', linestyle='', color=color_map[s],
                               markersize=7, label=s) for s in highlight_species]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4, frameon=False,
               labelcolor=INK, fontsize=9, bbox_to_anchor=(0.5, -0.06))

    fig.suptitle('Same 7 species in three feature spaces (2D PCA projection)',
                 color=INK, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'feature_space_pca.png', dpi=150,
                facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    print('Saved feature_space_pca.png')


def plot_top_confusions():
    with open(SPLIT_FILE) as f:
        split = json.load(f)
    classical = load_feature_csv(PROCESSED_DIR / 'classical_features.csv')
    X_train, y_train = build_xy(classical, split['train'])
    X_test, y_test = build_xy(classical, split['test'])

    nb = GaussianNB().fit(X_train, y_train)
    predictions = nb.predict(X_test)
    cm = confusion_matrix(y_test, predictions, labels=range(NUM_CLASSES))
    np.fill_diagonal(cm, 0)

    flat = [(cm[i, j], i, j) for i in range(NUM_CLASSES) for j in range(NUM_CLASSES) if cm[i, j] > 0]
    flat.sort(reverse=True)
    top = flat[:15]
    labels = [f'{SPECIES_NAMES[i]} -> {SPECIES_NAMES[j]}' for _, i, j in top]
    counts = [c for c, _, _ in top]

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    y_pos = np.arange(len(labels))[::-1]
    ax.barh(y_pos, counts, color=CATEGORICAL[0])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color=INK, fontsize=9)
    ax.set_xlabel('Test-set images misclassified this way', color=INK_SECONDARY)
    ax.set_title('Naive Bayes (classical): most common misclassifications\n(true species -> predicted species)',
                 color=INK, fontsize=12)
    for y, c in zip(y_pos, counts):
        ax.text(c + 0.1, y, str(c), va='center', color=INK, fontsize=8)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'top_confusions_naive_bayes.png', dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print('Saved top_confusions_naive_bayes.png')


def main():
    plot_per_class_heatmap()
    plot_feature_space_pca()
    plot_top_confusions()


if __name__ == '__main__':
    main()
