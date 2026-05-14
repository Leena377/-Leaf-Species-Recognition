# Leaf-Species-Recognition

Automatic Image Analysis project

## Topic A - Shape and Contour: Leaf Species Recognition

### Dataset

- [Flavia Leaf Dataset](https://flavia.sourceforge.net/): 1,907 leaf images, 32 plant species.
- Alternative: [Leafsnap](https://leafsnap.com/dataset).

### Core task

Given a segmented leaf silhouette, classify the plant species.

### Main question

How far can shape alone go, and when do learned image features add
useful information?

## Required work

- **Classical pipeline**: Fourier descriptors of the normalized boundary and Hu moments.
- **Classifiers**: naive Bayes and k-NN on a stratifed train/test split.
- **Modern pipeline**: frozen MobileNetV2 features with a linear SVM.
- Compare all methods on the same split.

### Expected outputs

- confusion matrices
- per-class accuracy
- example leaves that are easy or hard
- short discussion of when boundary information is sufficient

### Theory focus

- Fourier descriptor invariance
- Hu moments
- the independence assumption in naive Bayes
