import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# File paths
FEATURES_FILE = Path('../data/processed/classical_features.csv')

# Flavia mapping based on provided metadata
FLAVIA_RANGES = [
    (1001, 1059, 0),  # Phyllostachys edulis (Label 1 -> Index 0)
    (1060, 1122, 1),  # Aesculus chinensis
    (1552, 1616, 2),  # Berberis anhweiensis
    (1123, 1194, 3),  # Cercis chinensis
    (1195, 1267, 4),  # Indigofera tinctoria
    (1268, 1323, 5),  # Acer Palmatum
    (1324, 1385, 6),  # Phoebe nanmu
    (1386, 1437, 7),  # Kalopanax septemlobus
    (1497, 1551, 8),  # Cinnamomum japonicum
    (1438, 1496, 9),  # Koelreuteria paniculata
    (2001, 2050, 10), # Ilex macrocarpa
    (2051, 2113, 11), # Pittosporum tobira
    (2114, 2165, 12), # Chimonanthus praecox
    (2166, 2230, 13), # Cinnamomum camphora
    (2231, 2290, 14), # Viburnum awabuki
    (2291, 2346, 15), # Osmanthus fragrans
    (2347, 2423, 16), # Cedrus deodara
    (2424, 2485, 17), # Ginkgo biloba
    (2486, 2546, 18), # Lagerstroemia indica
    (2547, 2612, 19), # Nerium oleander
    (2616, 2675, 20), # Podocarpus macrophyllus
    (3001, 3055, 21), # Prunus serrulata
    (3056, 3110, 22), # Ligustrum lucidum
    (3111, 3175, 23), # Tonna sinensis
    (3176, 3229, 24), # Prunus persica
    (3230, 3281, 25), # Manglietia fordiana
    (3282, 3334, 26), # Acer buergerianum
    (3335, 3389, 27), # Mahonia bealei
    (3390, 3446, 28), # Magnolia grandiflora
    (3447, 3510, 29), # Populus ×canadensis
    (3511, 3563, 30), # Liriodendron chinense
    (3566, 3621, 31), # Citrus reticulata
]

def get_label_from_filename(filename):
    """Assigns the correct label (0-31) based on the image ID range."""
    # Extract the number from the filename (e.g., '1005.jpg' -> 1005)
    image_id = int(''.join(filter(str.isdigit, filename)))
    
    for start, end, label in FLAVIA_RANGES:
        if start <= image_id <= end:
            return label
            
    print(f"Warning: File {filename} does not fall into any known range.")
    return -1

def load_data():
    """Loads features and labels from the CSV."""
    X = []
    y = []
    
    print(f"Loading features from {FEATURES_FILE}...")
    with open(FEATURES_FILE, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader) 
        
        for row in reader:
            filename = row[0]
            features = [float(val) for val in row[1:]] 
            label = get_label_from_filename(filename)
            
            if label != -1:
                X.append(features)
                y.append(label)
            
    return np.array(X), np.array(y)

def evaluate_model(model, X_test, y_test, model_name):
    """Generates predictions, accuracy, and confusion matrix plots."""
    print(f"\nEvaluating {model_name}...")
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"{model_name} Accuracy: {accuracy * 100:.2f}%")
    
    # Generate Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    
    # Plot formatting
    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(ax=ax, cmap='viridis', colorbar=False)
    plt.title(f'{model_name} Confusion Matrix')
    
    # Ensure results directory exists
    Path('../results/figures').mkdir(parents=True, exist_ok=True)
    plt.savefig(f'../results/figures/{model_name.lower().replace(" ", "_")}_cm.png')
    print(f"Saved confusion matrix plot for {model_name}.")

def main():
    X, y = load_data()
    print(f"Successfully loaded {len(X)} samples with {X.shape[1]} features each.")
    
    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Split data: {len(X_train)} train, {len(X_test)} test.")
    
    # Evaluate k-NN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    evaluate_model(knn, X_test, y_test, "k-Nearest Neighbors")
    
    # Evaluate Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    evaluate_model(nb, X_test, y_test, "Naive Bayes")

if __name__ == "__main__":
    main()