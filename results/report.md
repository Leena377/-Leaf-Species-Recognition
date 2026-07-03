# Leaf Species Recognition — Results and Discussion

Dataset: [Flavia](https://flavia.sourceforge.net/), 1907 leaf images, 32 species.
All four methods below are trained and evaluated on the **same** stratified
80/20 split (`data/processed/split.json`, 1525 train / 382 test images,
`random_state=42`), so accuracies are directly comparable. Chance-level
accuracy for 32 balanced classes is ≈3.1%.

## Comparison table

| Method | Input | Overall accuracy | Macro per-class accuracy |
|---|---|---|---|
| Naive Bayes | classical (7 Hu moments + 16 Fourier descriptor magnitudes) | 50.00% | 49.86% |
| k-NN (k=5) | classical (same features) | 58.38% | 56.89% |
| Linear SVM | MobileNetV2 features, **silhouette** (shape only, replicated to 3 channels) | 96.34% | 96.20% |
| Linear SVM | MobileNetV2 features, **raw color image** (shape + texture + venation + color) | 98.95% | 98.83% |

(Full per-class tables: `results/tables/*_per_class_accuracy.csv`. Confusion
matrices: `results/figures/*_cm.png`.)

## How far does shape alone go?

Two different pipelines answer this, and they disagree sharply:

- **Hand-crafted shape descriptors** (Hu moments + Fourier descriptors) cap
  out at 50–58% with naive Bayes / k-NN.
- **Learned shape features** — MobileNetV2 applied to nothing but the binary
  silhouette mask, i.e. the *same* information budget as the classical
  pipeline — reach 96.3%.

So "shape alone" is far more informative than the classical descriptors
suggest. The bottleneck in the classical pipeline isn't the absence of
appearance cues, it's that 7 Hu moments and 16 truncated Fourier magnitudes
are a very lossy, low-order summary of a contour: Hu moments only use
normalized moments up to 3rd order, and the Fourier descriptors are
low-pass truncated to 16 harmonics with all phase information discarded
(see below). MobileNetV2's convolutional filters, even fed a binary mask,
extract a much richer, multi-scale description of the same boundary
(curvature at many scales, local convexity/concavity patterns, etc.).

Species that are already perfectly (or near-perfectly) classified from
shape alone by even the weakest classical model (Naive Bayes) have visually
distinctive outlines: `Cedrus deodara` (needle), `Acer palmatum` (palmate
lobes), `Mahonia bealei` (spiny compound leaflet), `Chimonanthus praecox`,
`Phyllostachys edulis` (bamboo, narrow lanceolate). These are exactly the
"easy" examples in `results/figures/easy_examples.png` — the needle-shaped
`Cedrus deodara` and the fan-lobed `Ginkgo biloba` are correctly identified
by all four methods, because their silhouette is already close to
class-unique.

## When does learned appearance add value beyond shape?

Going from silhouette-only deep features (96.3%) to raw-image deep features
(98.9%) is a smaller jump than going from classical to deep, but it is not
noise: several species jump from **0% to 100%** per-class accuracy for
Naive Bayes once color/texture/venation are available via the raw-image
SVM — `Viburnum awabuki`, `Prunus serrulata`, `Nerium oleander`,
`Cinnamomum japonicum`, `Cinnamomum camphora`, `Berberis anhweiensis`,
`Aesculus chinensis`, `Prunus persica` (91%). All of these are simple
ovate/lanceolate/elliptic leaves whose *outlines* are close to
indistinguishable from several other species in the dataset — shape is
genuinely insufficient here, and only appearance (leaf color, surface
texture, gloss, venation pattern) breaks the tie. This is visible directly
in `results/figures/hard_examples.png`: most of the leaves that stump every
shape-based method (classical *and* silhouette-CNN) are plain, elongated,
smooth-margined blades, and the raw-image SVM recovers the correct label
for most of them by using color/texture instead of contour.

One notable exception: `Ilex macrocarpa` is the *only* species where the
raw-image SVM (80%) underperforms the silhouette-only SVM (100%), despite
Naive Bayes doing poorly on it from shape alone (20%). Its silhouette
(a compact, slightly undulating oval) is apparently distinctive enough for
a CNN once you go beyond 3rd-order moments, while its raw appearance
(variable lighting/gloss on the photographed leaves) seems to introduce
noise the appearance-based classifier has to fight through.

**Bottom line:** boundary information is sufficient whenever a species has
a geometrically distinctive silhouette (needles, lobed/palmate margins,
compound leaflets, strongly elongated blades) — a *learned* shape encoder
gets these right almost independently of appearance. It is insufficient
for the many simple, smooth, ovate/elliptic leaves in this dataset, where
species only really separate once color, gloss, and venation enter the
picture. Hand-crafted shape descriptors (Hu + Fourier) are strictly weaker
than either — they lose too much shape resolution even in the regime where
shape alone would otherwise be enough.

## Additional analysis

Three more figures dig into *why* the numbers above look the way they do.

### Per-class accuracy, one view (`results/figures/per_class_accuracy_heatmap.png`)

All 32 species, all four methods, sorted by Naive Bayes accuracy. Two things
stand out that the summary table hides:

- **The classical column is bimodal, not just "low."** Naive Bayes gets 8
  species at 100% and 8 species at exactly 0% — there's almost no middle
  ground. That matches the theory: species with an outlier-shaped silhouette
  (needles, palmate lobes, compound leaflets) sit far from every other
  class's Gaussian in Hu/Fourier space and are trivial; everything with an
  ordinary ovate/elliptic outline collides with several other classes and is
  essentially unrecoverable from shape statistics alone, not just "harder."
- **k-NN is not uniformly better than Naive Bayes per class**, even though it
  wins overall. `Magnolia grandiflora` goes 36% (NB) → 9% (k-NN) — worse —
  while `Indigofera tinctoria` goes 20% → 87%. Aggregate accuracy hides this;
  the two classical classifiers make *different* mistakes, they don't just
  differ in how many mistakes they make.
- Once you're in deep-feature space (either column on the right), the
  heatmap is almost solid dark blue — the ceiling effect from the classical
  columns disappears almost entirely.

### Most common classical confusions (`results/figures/top_confusions_naive_bayes.png`)

Ranking Naive Bayes' actual misclassifications on the test set shows it isn't
making 32-way-random errors — it collapses many unrelated species onto a
handful of "generic shape" bins: `Phyllostachys edulis` (narrow bamboo blade)
absorbs `Nerium oleander` and `Podocarpus macrophyllus`; `Magnolia
grandiflora` (broad ovate) absorbs `Aesculus chinensis`, `Viburnum awabuki`,
`Indigofera tinctoria`; `Chimonanthus praecox` / `Phoebe nanmu` absorb five
more. With only 3rd-order Hu moments and 16 truncated Fourier harmonics,
many genuinely different species reduce to the same handful of coarse
"archetype" vectors, and Naive Bayes' Gaussian-per-feature model just picks
whichever archetype's mean is nearest — this is the independence assumption
and the low-order truncation compounding each other.

### Confused pairs, side by side (`results/figures/top_confusion_pairs_shape.png`)

For the six most frequent Naive Bayes confusions, this shows the actual
misclassified leaf's silhouette next to the *closest-matching* silhouette of
the species it got predicted as (nearest neighbor in raw Hu+Fourier feature
space, not just an arbitrary example — so this is a genuine best-case
justification for the error, not a cherry-picked lookalike). Five of the six
pairs are visually convincing on sight: `Nerium oleander` vs. `Phyllostachys
edulis`, `Prunus persica` vs. `Phoebe nanmu`, and `Podocarpus macrophyllus`
vs. `Phyllostachys edulis` are all long, thin, gently curved blades that are
essentially the same silhouette at different curvatures; `Aesculus chinensis`
vs. `Magnolia grandiflora` and `Viburnum awabuki` vs. `Magnolia grandiflora`
are both broad symmetric ovals.

`Indigofera tinctoria` vs. `Magnolia grandiflora` is the one pair in this set
that *doesn't* look alike — the misclassified leaf is a small, nearly round
oval, and even its nearest neighbor in feature space is a much more elongated
blade. That's actually informative rather than a bug: it's a direct,
concrete illustration of the naive-independence problem discussed below.
`GaussianNB` doesn't require overall shape similarity — it only requires each
of the 23 Hu/Fourier dimensions to independently score better under
`Magnolia grandiflora`'s per-feature Gaussians than under `Indigofera
tinctoria`'s. A leaf can rack up a high joint likelihood for the wrong class
one coordinate at a time without ever resembling a real member of that class
overall, which a true joint (non-independent) model would penalize.

### Where only appearance saves you (`results/figures/shape_fails_appearance_succeeds.png`)

Filtering the test set for leaves where Naive Bayes **and** the
silhouette-only deep SVM both got it wrong, but the raw-image deep SVM got it
right, turns up 9 leaves (8 shown). Every single one is a smooth-margined,
smallish, single-color-looking-in-silhouette oval or lanceolate blade — in
other words, exactly the shape archetype the confusion analysis above
identified as an attractor bin. Once the actual raw photo is shown next to
its silhouette, the leaves stop looking alike: distinct greens (`Cinnamomum
camphora`'s glossy mid-green vs. `Viburnum awabuki`'s near-black-green),
visible venation patterns, and surface gloss are all available to the
raw-image SVM and to none of the shape-only methods. This is the most direct
demonstration in the whole analysis of the report's central claim: for this
subset of species, the outline genuinely carries too little information,
and color/texture is doing the actual discriminating work.

### Feature space, visualized (`results/figures/feature_space_pca.png`)

Same 7 species (2 shape-distinctive controls, 4 that jump from 0%→100%
between classical and raw-image SVM, plus the `Ilex macrocarpa` exception),
projected to 2D with PCA (features standardized first) in all three feature
spaces side by side. This is the clearest single picture in the whole
analysis:

- In **classical space**, only `Cedrus deodara` (the needle) separates from
  the pack — as an extreme aspect-ratio outlier along PC1. Every other
  highlighted species, including the visually distinctive `Ginkgo biloba`,
  collapses into the same dense column as the 25 unhighlighted species. Two
  principal components can't resolve them, which is consistent with the
  heatmap's 0% rows — though note this doesn't prove they're inseparable in
  the full 23-d space, only that the dominant two directions of variance
  don't carry species identity for these classes.
- In **both deep spaces**, the same species spread out along a smooth arc
  and form largely distinct islands — `Cedrus deodara`, `Ginkgo biloba`, and
  `Nerium oleander` are cleanly isolated in both, matching their 100%
  accuracy under either deep pipeline.
- `Ilex macrocarpa` (pink) is the interesting one: it looks similarly mixed
  with `Cinnamomum camphora`/`Viburnum awabuki` in *both* the silhouette and
  raw-image panels — yet the measured accuracy is 100% for silhouette-SVM
  and only 80% for raw-SVM. That mismatch is a genuine caveat about reading
  too much into a 2D projection: a linear SVM operates in the full 1280-d
  space, where classes can be perfectly separated by a direction that
  contributes almost nothing to the first two principal components. The
  chart illustrates the overall shift in separability well, but it is not a
  faithful stand-in for what the classifier actually sees.

## Theory notes

### Fourier descriptor invariance

The boundary is traced as a complex sequence `z(t) = x(t) + i·y(t)` and
transformed with a DFT, `Z(k) = Σ_t z(t) e^{-2πikt/N}`. Three invariances
are built in deliberately (`scripts/extract_features.py`):

- **Translation**: shifting every boundary point by a constant vector `c`
  only changes the DC term, `Z(0) → Z(0) + c·N`; all other coefficients are
  untouched. Dropping `Z(0)` removes translation dependence entirely.
- **Scale**: scaling the shape by `s` multiplies every `Z(k)` by `s`.
  Dividing the retained coefficients by `|Z(1)|` normalizes this out.
- **Rotation and start-point**: rotating the contour by `θ` multiplies
  `z(t)` (and hence every `Z(k)`) by the constant phasor `e^{iθ}`; picking a
  different starting point on the contour multiplies `Z(k)` by
  `e^{-2πikt0/N}`. Both operations only change *phase*, never *magnitude*.
  Taking `|Z(k)|` as the descriptor is therefore invariant to both — which
  matters in practice because `cv2.findContours` gives no guarantee about
  where a contour "starts."

The cost of this invariance is that all phase information is discarded, so
the descriptor cannot distinguish a shape from constructions that share the
same power spectrum but differ in the spatial arrangement of that energy,
and cannot detect mirror-image (chirality) differences. Truncating to 16
harmonics is a low-pass filter: it keeps the coarse envelope of the outline
and throws away fine serration/venation-scale detail — a second, compounding
source of the classical pipeline's ceiling.

### Hu moments

The seven Hu invariants are specific polynomial combinations of the
normalized central moments (moments computed about the centroid, then
scaled by powers of the zeroth moment/area) chosen so that translation,
scale, and rotation all cancel algebraically (Hu, 1962); the 7th moment
additionally flips sign under reflection, giving weak chirality sensitivity.
Because raw moment values span many orders of magnitude across shapes and
moment orders, `extract_features.py` applies a signed log transform,
`-sign(h)·log10(|h|)`, so no single moment dominates a Euclidean/Gaussian
comparison purely due to scale. Like the truncated Fourier descriptors,
Hu moments only reach 3rd order — they summarize coarse elongation/symmetry
but have no resolution for the boundary irregularities that separate
otherwise-similar leaves.

### The independence assumption in Naive Bayes

`GaussianNB` models `P(x | y) = Π_i P(x_i | y)`, i.e. it scores each Hu
moment and each Fourier magnitude as if it were conditionally independent
of every other feature given the species. That assumption is clearly false
here: the Fourier magnitudes are ordered harmonics of one spectrum (their
relative energies are correlated by construction — an elongated blade has a
characteristic joint pattern across several harmonics, not one independent
signal per harmonic), and the seven Hu moments are nested polynomials of
the same underlying central moments. Naive Bayes can't represent the
*joint* evidence "this specific combination of aspect ratio and boundary
undulation" — it can only multiply per-coordinate scores, which
double-counts correlated evidence and misses interaction effects. k-NN uses
the full joint feature space directly (no independence assumption at all)
and, on the *identical* classical features, outperforms Naive Bayes by
+8.4 points overall (58.4% vs. 50.0%) — a direct, controlled measurement of
what the independence assumption costs on this feature set.

## Outputs

- `results/tables/model_comparison.csv` — the four-method summary above.
- `results/tables/*_per_class_accuracy.csv` — per-species accuracy + support for each method.
- `results/figures/*_cm.png` — confusion matrices (species-labeled) for each method.
- `results/figures/easy_examples.png` / `hard_examples.png` — test leaves ranked by cross-method agreement, with true label and all four predictions annotated.
- `results/figures/per_class_accuracy_heatmap.png` — all 4 methods x 32 species in one view.
- `results/figures/top_confusions_naive_bayes.png` — most frequent true→predicted species pairs for the classical Naive Bayes model.
- `results/figures/top_confusion_pairs_shape.png` — misclassified leaf vs. closest-matching silhouette of the predicted species, for the 6 most common confusions.
- `results/figures/shape_fails_appearance_succeeds.png` — leaves both shape-only methods got wrong that the raw-image SVM got right, silhouette next to raw photo.
- `results/figures/feature_space_pca.png` — 2D PCA of 7 species across classical / deep-silhouette / deep-raw feature spaces.
- `results/tables/test_predictions.csv` — per-test-image predictions from all 4 methods (used to build the two figures above).
