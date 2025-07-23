import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Enable debugging output
DEBUG = True

def parse_annotations(file_path):
    """Parse annotation files and convert to YOLO-like format."""
    # This function assumes a specific format you might be using.
    # It looks for lines like '<DOCUMENT> 0 0.1 0.2 0.3 0.4 </DOCUMENT>'
    # and converts them. This may need adjustment for your exact annotation style.
    annotations = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # A simple check for a potential format
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        class_label = int(parts[0])
                        x_center, y_center, w, h = map(float, parts[1:])
                        annotations.append((class_label, x_center, y_center, w, h))
                    except ValueError:
                        if DEBUG:
                            print(f"Skipping invalid line in {file_path}: {line}")
                        continue
    except Exception as e:
        if DEBUG:
            print(f"Error reading {file_path}: {str(e)}")
    if DEBUG and annotations:
        print(f"Parsed {len(annotations)} annotations from {file_path}")
    return annotations

def process_image(img):
    """Preprocess an image section for feature extraction."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    _, thresh = cv2.threshold(saturation, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    adaptive = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 12)
    kernel = np.ones((5,5), np.uint8)
    processed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
    return processed

def extract_features(processed_img):
    """Extract texture features from the processed image."""
    processed_img = processed_img.astype(np.uint8)
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(processed_img, distances=[1], angles=angles, symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast').mean()
    energy = graycoprops(glcm, 'energy').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    
    white_percentage = np.sum(processed_img == 255) / processed_img.size
    
    features = [contrast, energy, homogeneity, correlation, dissimilarity, white_percentage]
    return np.array(features)

def load_and_process_dataset(dataset_path):
    """Load dataset and process images with annotations."""
    all_features = []
    all_labels = []
    
    # We assume a simple structure: dataset_path/{agglutinated, non_agglutinated}/image.jpg
    # 0 = non_agglutinated, 1 = agglutinated
    for label, class_name in enumerate(['non_agglutinated', 'agglutinated']):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            if DEBUG:
                print(f"Directory not found, skipping: {class_path}")
            continue
        
        if DEBUG:
            print(f"Processing images in: {class_path}")
        
        for file_name in os.listdir(class_path):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(class_path, file_name)
                img = cv2.imread(image_path)
                if img is None:
                    if DEBUG:
                        print(f"Failed to load image: {image_path}")
                    continue
                
                processed = process_image(img)
                features = extract_features(processed)
                all_features.append(features)
                all_labels.append(label) # 0 for non_agglutinated, 1 for agglutinated
    
    if DEBUG:
        print(f"Collected a total of {len(all_features)} samples.")
    return np.array(all_features), np.array(all_labels)


def train_model(dataset_path):
    """Train a RandomForestClassifier on the dataset."""
    X, y = load_and_process_dataset(dataset_path)
    
    if len(X) < 10: # Need a minimum number of samples to train
        print("Not enough total samples to train the model. Need at least 10.")
        return None
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    if DEBUG:
        print(f"Training with {len(X_train)} samples, testing with {len(X_test)} samples.")
    
    # Train a RandomForest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    if DEBUG:
        print(f"\nModel accuracy on test set: {accuracy:.2%}")
    
    return model

if __name__ == "__main__":
    # IMPORTANT: You need to create a dataset folder with this structure:
    # blood_dataset/
    # |-- agglutinated/
    # |   |-- image1.jpg
    # |   |-- image2.png
    # |   `-- ...
    # `-- non_agglutinated/
    #     |-- image3.jpg
    #     |-- image4.jpeg
    #     `-- ...
    dataset_path = "blood_dataset" 
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset folder '{dataset_path}' not found!")
        print("Please create it and add your labeled image data.")
        exit()
    
    print(f"Starting model training using data from '{dataset_path}'...")
    trained_classifier = train_model(dataset_path)
    
    if trained_classifier is not None:
        # Save the trained model to a file
        model_filename = "blood_type_classifier.pkl"
        joblib.dump(trained_classifier, model_filename)
        print(f"\nModel successfully trained and saved as '{model_filename}'")
    else:
        print("\nFailed to train model. Please check your dataset folder and its contents.")

