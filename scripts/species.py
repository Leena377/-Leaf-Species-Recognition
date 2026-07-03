"""Shared Flavia leaf label mapping: image ID ranges -> species index/name.

Ranges validated against the 1907 files in data/raw/ (exact coverage, no
gaps or overlaps), so this mapping is safe to reuse across all pipelines.
"""

FLAVIA_RANGES = [
    (1001, 1059, 0),   # Phyllostachys edulis
    (1060, 1122, 1),   # Aesculus chinensis
    (1552, 1616, 2),   # Berberis anhweiensis
    (1123, 1194, 3),   # Cercis chinensis
    (1195, 1267, 4),   # Indigofera tinctoria
    (1268, 1323, 5),   # Acer palmatum
    (1324, 1385, 6),   # Phoebe nanmu
    (1386, 1437, 7),   # Kalopanax septemlobus
    (1497, 1551, 8),   # Cinnamomum japonicum
    (1438, 1496, 9),   # Koelreuteria paniculata
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

# Index -> scientific name, in label order (0-31).
SPECIES_NAMES = [
    "Phyllostachys edulis",
    "Aesculus chinensis",
    "Berberis anhweiensis",
    "Cercis chinensis",
    "Indigofera tinctoria",
    "Acer palmatum",
    "Phoebe nanmu",
    "Kalopanax septemlobus",
    "Cinnamomum japonicum",
    "Koelreuteria paniculata",
    "Ilex macrocarpa",
    "Pittosporum tobira",
    "Chimonanthus praecox",
    "Cinnamomum camphora",
    "Viburnum awabuki",
    "Osmanthus fragrans",
    "Cedrus deodara",
    "Ginkgo biloba",
    "Lagerstroemia indica",
    "Nerium oleander",
    "Podocarpus macrophyllus",
    "Prunus serrulata",
    "Ligustrum lucidum",
    "Toona sinensis",
    "Prunus persica",
    "Manglietia fordiana",
    "Acer buergerianum",
    "Mahonia bealei",
    "Magnolia grandiflora",
    "Populus x canadensis",
    "Liriodendron chinense",
    "Citrus reticulata",
]

NUM_CLASSES = len(SPECIES_NAMES)


def get_label_from_filename(filename):
    """Assigns the correct label (0-31) based on the image ID range."""
    image_id = int(''.join(filter(str.isdigit, filename)))

    for start, end, label in FLAVIA_RANGES:
        if start <= image_id <= end:
            return label

    raise ValueError(f"File {filename} does not fall into any known Flavia ID range.")


def label_name(label):
    return SPECIES_NAMES[label]
