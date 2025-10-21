import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import os
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.morphology import erosion, dilation, disk
from skimage.measure import label, regionprops
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input


num_classes = 8

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS_INITIAL = 5
EPOCHS_FINE_TUNE = 5
LEARNING_RATE_HEAD = 0.001
LEARNING_RATE_FINE_TUNE = 0.0001

models_dir = "training_artifacts"
os.makedirs(models_dir, exist_ok=True)
keras_model_path_initial_head = os.path.join(models_dir, "keras_model_initial_head.weights.h5")
keras_model_path_fine_tuned = os.path.join(models_dir, "keras_model_fine_tuned.weights.h5")
stacking_classifier_path = os.path.join(models_dir, "stacking_classifier.pkl")
scaler_path = os.path.join(models_dir, "scaler.pkl")
label_encoder_path = os.path.join(models_dir, "label_encoder.pkl")

blood_group_labels = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

print("--- Loading or Defining Keras Model for DL Feature Extraction ---")
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
loaded_keras_model = Model(inputs=base_model.input, outputs=x)
for layer in base_model.layers:
    layer.trainable = False
print("\nKeras model for DL feature extraction loaded/defined.")


dl_feature_extractor = loaded_keras_model

def analyze_agglutination(image_path):
    AGGLUTINATION_FEATURE_SIZE = 3
    img = cv2.imread(image_path)
    if img is None:
         return np.zeros(AGGLUTINATION_FEATURE_SIZE)
    try:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img_blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
        block_size = 11
        C_value = 2
        if block_size % 2 == 0: block_size += 1
        if block_size <= 1: block_size = 3
        try: binary_mask = cv2.adaptiveThreshold(gray_img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, C_value)
        except Exception: return np.zeros(AGGLUTINATION_FEATURE_SIZE)
        selem = disk(3)
        binary_mask = erosion(binary_mask, selem)
        binary_mask = dilation(binary_mask, selem)
        binary_mask = erosion(binary_mask, selem)
        labeled_image = label(binary_mask, connectivity=2)
        regions = regionprops(labeled_image)
        min_area = 50
        valid_regions = [r for r in regions if r.area > min_area]
        num_valid_clusters = len(valid_regions)
        if num_valid_clusters == 0: return np.zeros(AGGLUTINATION_FEATURE_SIZE)
        total_area = sum(r.area for r in valid_regions)
        average_cluster_area = total_area / num_valid_clusters
        density = total_area / (img.shape[0] * img.shape[1] + 1e-6)
        features = np.array([num_valid_clusters, average_cluster_area, density])
        if len(features) != AGGLUTINATION_FEATURE_SIZE:
             padded_features = np.zeros(AGGLUTINATION_FEATURE_SIZE)
             padded_features[:min(len(features), AGGLUTINATION_FEATURE_SIZE)] = features[:min(len(features), AGGLUTINATION_FEATURE_SIZE)]
             return padded_features.flatten()
        return features.flatten()
    except Exception: return np.zeros(AGGLUTINATION_FEATURE_SIZE)


def extract_advanced_handcrafted_features(image_path):
    EXPECTED_GLCM_LEN = 61
    EXPECTED_COLOR_LEN = 256 * 3
    EXPECTED_LBP_LEN = 24
    EXPECTED_AGGLUTINATION_LEN = 3
    TOTAL_HANDCRAFTED_FEATURE_SIZE = EXPECTED_GLCM_LEN + EXPECTED_COLOR_LEN + EXPECTED_LBP_LEN + EXPECTED_AGGLUTINATION_LEN

    img = cv2.imread(image_path)
    if img is None:
        return np.zeros(TOTAL_HANDCRAFTED_FEATURE_SIZE)
    try:
        denoised_img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        lab = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        processed_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        gray_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        glcm_features = np.zeros(EXPECTED_GLCM_LEN)
        lbp_features = np.zeros(EXPECTED_LBP_LEN)

        if gray_img.shape[0] >= 5 and gray_img.shape[1] >= 5 and len(np.unique(gray_img)) >= 2:
             distances = [1, 3, 5] ; angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
             properties = ['contrast', 'energy', 'homogeneity', 'correlation', 'dissimilarity']
             all_glcm_features = []
             try:
                 gray_img_uint8 = gray_img.astype(np.uint8)
                 if np.all(gray_img_uint8 == gray_img_uint8[0,0]):
                     glcm = np.zeros((len(distances), len(angles), 256, 256))
                 else:
                     glcm = graycomatrix(gray_img_uint8, distances=distances, angles=angles, symmetric=True, normed=True)
                 for prop in properties:
                     prop_values = graycoprops(glcm, prop).flatten()
                     prop_values[np.isnan(prop_values)] = 0
                     all_glcm_features.extend(list(prop_values))
                 try:
                     _, thresholded_gray = cv2.threshold(gray_img_uint8, 128, 255, cv2.THRESH_BINARY)
                     white_percentage = np.sum(thresholded_gray == 255) / (thresholded_gray.size + 1e-6)
                     all_glcm_features.append(white_percentage)
                 except Exception: all_glcm_features.append(0)
                 glcm_features = np.array(all_glcm_features)
             except Exception as e:
                 glcm_features = np.zeros(EXPECTED_GLCM_LEN)

             radius = 3 ; n_points = 8 * radius
             try:
                 lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
                 n_bins = int(lbp.max() + 1)
                 if n_bins > 0 and lbp.size > 0:
                     hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
                     hist = hist.astype("float") / (hist.sum() + 1e-6)
                 else: hist = np.zeros(EXPECTED_LBP_LEN)
                 lbp_features = hist
                 if len(lbp_features) != EXPECTED_LBP_LEN:
                      padded_lbp_features = np.zeros(EXPECTED_LBP_LEN)
                      padded_lbp_features[:min(len(lbp_features), EXPECTED_LBP_LEN)] = lbp_features[:min(len(lbp_features), EXPECTED_LBP_LEN)]
                      lbp_features = padded_lbp_features
             except Exception as e:
                 lbp_features = np.zeros(EXPECTED_LBP_LEN)

        color_features = np.zeros(EXPECTED_COLOR_LEN)
        if processed_img is not None and processed_img.shape[0] > 0 and processed_img.shape[1] > 0:
            try:
                hist_b = cv2.calcHist([processed_img], [0], None, [256], [0, 256])
                hist_g = cv2.calcHist([processed_img], [1], None, [256], [0, 256])
                hist_r = cv2.calcHist([processed_img], [2], None, [256], [0, 256])
                hist_b = cv2.normalize(hist_b, hist_b, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).flatten()
                hist_g = cv2.normalize(hist_g, hist_g, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).flatten()
                hist_r = cv2.normalize(hist_r, hist_r, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).flatten()
                color_features = np.concatenate((hist_b, hist_g, hist_r))
            except Exception as e:
                color_features = np.zeros(EXPECTED_COLOR_LEN)

        agglutination_features = analyze_agglutination(image_path)

        if len(glcm_features) != EXPECTED_GLCM_LEN: glcm_features = np.zeros(EXPECTED_GLCM_LEN)
        if len(color_features) != EXPECTED_COLOR_LEN: color_features = np.zeros(EXPECTED_COLOR_LEN)
        if len(lbp_features) != EXPECTED_LBP_LEN: lbp_features = np.zeros(EXPECTED_LBP_LEN)
        if len(agglutination_features) != EXPECTED_AGGLUTINATION_LEN: agglutination_features = np.zeros(EXPECTED_AGGLUTINATION_LEN)

        combined_handcrafted_features = np.concatenate((glcm_features, color_features, lbp_features, agglutination_features))
        if len(combined_handcrafted_features) != TOTAL_HANDCRAFTED_FEATURE_SIZE:
             return np.zeros(TOTAL_HANDCRAFTED_FEATURE_SIZE).flatten()
        return combined_handcrafted_features.flatten()
    except Exception as e:
        return np.zeros(TOTAL_HANDCRAFTED_FEATURE_SIZE).flatten()


def extract_dl_features(image_path, feature_extractor):
    expected_dl_size = 512
    if loaded_keras_model is not None:
        try:
            feature_extraction_layer_name = 'global_average_pooling2d'
            layer = loaded_keras_model.get_layer(feature_extraction_layer_name)
            layer_output_shape = layer.output_shape
            if len(layer_output_shape) == 2: expected_dl_size = layer_output_shape[1]
            else: expected_dl_size = np.prod(layer_output_shape[1:])
        except Exception: pass

    if feature_extractor is None: return np.zeros(expected_dl_size if expected_dl_size is not None else 512).flatten()
    img = cv2.imread(image_path)
    if img is None: return np.zeros(expected_dl_size if expected_dl_size is not None else 512).flatten()
    try:
        processed_img = img
        img_resized = cv2.resize(processed_img, (IMG_WIDTH, IMG_HEIGHT))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_preprocessed = vgg16_preprocess_input(img_rgb)
        img_batch = np.expand_dims(img_preprocessed, axis=0)
        dl_features = feature_extractor.predict(img_batch, verbose=0)
        if len(dl_features.shape) > 2: dl_features = dl_features.reshape(dl_features.shape[0], -1)
        dl_features = dl_features[0]
        actual_dl_size = dl_features.shape[0]
        if expected_dl_size is not None and actual_dl_size != expected_dl_size:
             padded_dl_features = np.zeros(expected_dl_size)
             padded_dl_features[:min(actual_dl_size, expected_dl_size)] = dl_features[:min(actual_dl_size, expected_dl_size)]
             return padded_dl_features.flatten()
        return dl_features.flatten()
    except Exception as e:
        return np.zeros(expected_dl_size if expected_dl_size is not None else 512).flatten()

def filter_inconsistent_features(handcrafted_features, dl_features, labels, image_paths, expected_handcrafted_size, expected_dl_size):
    filtered_handcrafted = []
    filtered_dl = []
    filtered_labels = []
    filtered_paths = []
    for i in range(len(labels)):
        hc_features = handcrafted_features[i]
        dl_features_i = dl_features[i]
        hc_array = np.array(hc_features).flatten()
        dl_array_i = np.array(dl_features_i).flatten()
        if (hc_array is not None and hc_array.shape == (expected_handcrafted_size,)) and \
           (dl_array_i is not None and dl_array_i.shape == (expected_dl_size,)):
            filtered_handcrafted.append(hc_array)
            filtered_dl.append(dl_array_i)
            filtered_labels.append(labels[i])
            filtered_paths.append(image_paths[i])
        else:
            pass
    return np.array(filtered_handcrafted), np.array(filtered_dl), np.array(filtered_labels), filtered_paths

EXPECTED_GLCM_LEN = 61
EXPECTED_COLOR_LEN = 256 * 3
EXPECTED_LBP_LEN = 24
EXPECTED_AGGLUTINATION_LEN = 3
TOTAL_HANDCRAFTED_FEATURE_SIZE = EXPECTED_GLCM_LEN + EXPECTED_COLOR_LEN + EXPECTED_LBP_LEN + EXPECTED_AGGLUTINATION_LEN

dl_feature_size = 512
if 'loaded_keras_model' in locals() and loaded_keras_model is not None:
    try:
        feature_extraction_layer_name = 'global_average_pooling2d'
        layer = loaded_keras_model.get_layer(feature_extraction_layer_name)
        layer_output_shape = layer.output_shape
        if len(layer_output_shape) == 2: dl_feature_size = layer_output_shape[1]
        elif len(layer_output_shape) > 2:
             dl_feature_size = np.prod(layer_output_shape[1:])
    except Exception: pass


print("\n--- Starting Data Loading and Preparation ---")
all_image_paths = []
all_labels = []
label_mapping = {}

if 'dataset_path_location' in globals() and dataset_path_location is not None:
    dataset_root = dataset_path_location
    print(f"Loading image paths from: {dataset_root}")
    try:
        for split in ['train', 'valid', 'test']:
            split_dir = os.path.join(dataset_root, split)
            if os.path.exists(split_dir):
                for class_name in os.listdir(split_dir):
                    class_dir = os.path.join(split_dir, class_name)
                    if os.path.isdir(class_dir):
                        if class_name not in label_mapping:
                            label_mapping[class_name] = len(label_mapping)
                        for img_file in os.listdir(class_dir):
                            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                all_image_paths.append(os.path.join(class_dir, img_file))
                                all_labels.append(label_mapping[class_name])

        print(f"Found {len(all_image_paths)} images across all splits.")
        print(f"Found {len(label_mapping)} classes: {label_mapping}")

        if len(all_image_paths) > 0:
            try:
                X_train_paths, X_temp_paths, y_train, y_temp = train_test_split(
                    all_image_paths, all_labels, test_size=0.3, random_state=42, stratify=all_labels
                )
                X_val_paths, X_test_paths, y_val, y_test = train_test_split(
                    X_temp_paths, y_temp, test_size=0.5, random_state=42, stratify=y_temp
                )

                print(f"Train samples: {len(X_train_paths)}")
                print(f"Validation samples: {len(X_val_paths)}")
                print(f"Test samples: {len(X_test_paths)}")

                print("\nExtracting features for training set...")
                train_handcrafted_features = [extract_advanced_handcrafted_features(p) for p in X_train_paths]
                train_dl_features = [extract_dl_features(p, dl_feature_extractor) for p in X_train_paths]

                print("Extracting features for validation set...")
                val_handcrafted_features = [extract_advanced_handcrafted_features(p) for p in X_val_paths]
                val_dl_features = [extract_dl_features(p, dl_feature_extractor) for p in X_val_paths]

                print("Extracting features for test set...")
                test_handcrafted_features = [extract_advanced_handcrafted_features(p) for p in X_test_paths]
                test_dl_features = [extract_dl_features(p, dl_feature_extractor) for p in X_test_paths]

                print("\nFiltering out inconsistent features...")
                train_handcrafted_features, train_dl_features, y_train_filtered, X_train_paths_filtered = filter_inconsistent_features(
                     train_handcrafted_features, train_dl_features, y_train, X_train_paths, TOTAL_HANDCRAFTED_FEATURE_SIZE, dl_feature_size
                )
                val_handcrafted_features, val_dl_features, y_val_filtered, X_val_paths_filtered = filter_inconsistent_features(
                     val_handcrafted_features, val_dl_features, y_val, X_val_paths, TOTAL_HANDCRAFTED_FEATURE_SIZE, dl_feature_size
                )
                test_handcrafted_features, test_dl_features, y_test_filtered, X_test_paths_filtered = filter_inconsistent_features(
                     test_handcrafted_features, test_dl_features, y_test, X_test_paths, TOTAL_HANDCRAFTED_FEATURE_SIZE, dl_feature_size
                )

                y_train_final = y_train_filtered
                y_val_final = y_val_filtered
                y_test_final = y_test_filtered

                print(f"\nTrain samples after feature extraction and filtering: {len(y_train_final)}")
                print(f"Validation samples after feature extraction and filtering: {len(y_val_final)}")
                print(f"Test samples after feature extraction and filtering: {len(y_test_final)}")

                if len(y_train_final) > 0 and len(y_test_final) > 0:
                     print("\nCombining features...")
                     X_train_combined = np.hstack((train_handcrafted_features, train_dl_features))
                     X_val_combined = np.hstack((val_handcrafted_features, val_dl_features))
                     X_test_combined = np.hstack((test_handcrafted_features, test_dl_features))

                     X_train_combined_flat = X_train_combined.reshape(X_train_combined.shape[0], -1)
                     X_val_combined_flat = X_val_combined.reshape(X_val_combined.shape[0], -1)
                     X_test_combined_flat = X_test_combined.reshape(X_test_combined.shape[0], -1)

                     print(f"Combined train features shape: {X_train_combined_flat.shape}")
                     print(f"Combined validation features shape: {X_val_combined_flat.shape}")
                     print(f"Combined test features shape: {X_test_combined_flat.shape}")

                     print("\nScaling combined features...")
                     scaler = StandardScaler()
                     X_train_scaled_combined = scaler.fit_transform(X_train_combined_flat)
                     X_val_scaled_combined = scaler.transform(X_val_combined_flat)
                     X_test_scaled_combined = scaler.transform(X_test_combined_flat)
                     loaded_scaler = scaler
                     print("Features scaled successfully.")
                     joblib.dump(scaler, scaler_path)
                     print(f"Scaler saved to {scaler_path}")

                     if 'label_encoder' not in locals() or label_encoder is None:
                          print("\nCreating and fitting LabelEncoder...")
                          label_encoder = LabelEncoder()
                          all_labels_combined = np.concatenate((y_train_final, y_val_final, y_test_final))
                          label_encoder.fit(np.unique(all_labels_combined))
                          loaded_label_encoder = label_encoder
                          print(f"LabelEncoder fitted on labels: {list(label_encoder.classes_)}")
                     if 'loaded_label_encoder' in locals() and loaded_label_encoder is not None:
                          joblib.dump(loaded_label_encoder, label_encoder_path)
                          print(f"LabelEncoder saved to {label_encoder_path}")
                     print("\n--- Data Loading and Preparation Finished ---")
                else:
                     print("\nInsufficient samples after feature extraction and filtering. Skipping model training.")
                     X_train_scaled_combined = X_val_scaled_combined = X_test_combined = y_train_final = y_val_final = y_test_final = None
            except Exception as e:
                print(f"\nAn error occurred during data loading, splitting, or feature extraction: {e}")
                print("Data preparation failed. Skipping model training.")
                X_train_scaled_combined = X_val_scaled_combined = X_test_combined = y_train_final = y_val_final = y_test_final = None
    except Exception as e:
        print(f"\nAn error occurred during initial data loading or splitting: {e}")
        print("Data loading failed. Skipping model training.")
        X_train_scaled_combined = X_val_scaled_combined = X_test_combined = y_train_final = y_val_final = y_test_final = None
else:
    print("\nSkipping data loading and preparation: Dataset not available or empty.")
    X_train_scaled_combined = X_val_combined = X_test_combined = y_train_final = y_val_final = y_test_final = None


# --- Manual Stacking Implementation ---
print("\n--- Starting Manual Stacking Implementation ---")
if 'X_train_scaled_combined' in locals() and X_train_scaled_combined is not None and \
   'y_train_final' in locals() and y_train_final is not None and \
   'X_test_scaled_combined' in locals() and X_test_scaled_combined is not None and \
   'y_test_final' in locals() and y_test_final is not None:
    try:
        print("\nTraining Base Estimators...")
        rf_base = RandomForestClassifier(n_estimators=100, random_state=42)
        svc_base = SVC(probability=True, random_state=42)
        rf_base.fit(X_train_scaled_combined, y_train_final)
        loaded_rf_base = rf_base
        print("Random Forest Base Estimator trained.")
        svc_base.fit(X_train_scaled_combined, y_train_final)
        loaded_svc_base = svc_base
        print("SVC Base Estimator trained.")

        print("\nGenerating base estimator predictions on training data...")
        rf_train_preds = loaded_rf_base.predict_proba(X_train_scaled_combined)
        svc_train_preds = loaded_svc_base.predict_proba(X_train_scaled_combined)
        X_meta_train = np.hstack((rf_train_preds, svc_train_preds))
        print(f"Meta-classifier training data shape: {X_meta_train.shape}")

        print("\nTraining Meta-Classifier...")
        meta_classifier_manual = LogisticRegression(random_state=42)
        meta_classifier_manual.fit(X_meta_train, y_train_final)
        loaded_meta_classifier = meta_classifier_manual
        print("Meta-Classifier trained.")

        print("\nEvaluating Manual Stacked Model on Test Data...")
        rf_test_preds = loaded_rf_base.predict_proba(X_test_scaled_combined)
        svc_test_preds = loaded_svc_base.predict_proba(X_test_scaled_combined)
        X_meta_test = np.hstack((rf_test_preds, svc_test_preds))
        print(f"Meta-classifier test data shape: {X_meta_test.shape}")
        y_pred_test_stacked = loaded_meta_classifier.predict(X_meta_test)
        stacked_test_accuracy = accuracy_score(y_test_final, y_pred_test_stacked)
        print(f"Manual Stacked Model Test Accuracy: {stacked_test_accuracy:.4f}")

        joblib.dump(loaded_rf_base, rf_base_path)
        joblib.dump(loaded_svc_base, svc_base_path)
        joblib.dump(loaded_meta_classifier, meta_classifier_manual_path)
        print("\nManual Stacked Model components saved.")
        print("\n--- Manual Stacking Implementation Finished ---")

    except Exception as e:
        print(f"\nAn error occurred during manual stacking implementation: {e}")
        print("Manual stacking implementation failed.")
else:
    print("\nSkipping manual stacking: Required data not available after data preparation.")
    print("Please review the data preparation steps for errors.")


# --- Optional: CNN Fine-tuning ---
# This section is for training the Keras model directly on images.
# It assumes you have train_ds, val_ds (TensorFlow Datasets).
# If you are only using the hybrid approach, you can ignore or remove this section.

print("\n--- Starting CNN Fine-tuning (Optional) ---")
# Assuming train_ds and val_ds are created from your dataset (e.g., using image_dataset_from_directory)
# Placeholder for creating train_ds and val_ds if they don't exist
# You would replace this with your actual data loading code to create TensorFlow Datasets.
train_ds = None
val_ds = None

# Example placeholder for creating TF Datasets (replace with your actual data loading)
if 'dataset_path_location' in globals() and dataset_path_location is not None:
    dataset_root = dataset_path_location
    try:
        # Assuming your dataset is organized into train/valid/test folders with class subfolders
        train_dir = os.path.join(dataset_root, 'train')
        val_dir = os.path.join(dataset_root, 'valid')
        # test_dir = os.path.join(dataset_root, 'test') # Not needed for CNN fine-tuning training

        if os.path.exists(train_dir) and os.path.exists(val_dir):
            print(f"\nCreating TensorFlow Datasets from: {dataset_root}")
            train_ds = tf.keras.utils.image_dataset_from_directory(
                train_dir,
                labels='inferred',
                label_mode='int', # Use 'int' for sparse_categorical_crossentropy
                image_size=(IMG_HEIGHT, IMG_WIDTH),
                interpolation='nearest',
                batch_size=BATCH_SIZE,
                shuffle=True
            )
            val_ds = tf.keras.utils.image_dataset_from_directory(
                val_dir,
                labels='inferred',
                label_mode='int',
                image_size=(IMG_HEIGHT, IMG_WIDTH),
                interpolation='nearest',
                batch_size=BATCH_SIZE,
                shuffle=False # Usually don't shuffle validation data
            )
            print("TensorFlow Datasets created.")
        else:
            print("\nSkipping TensorFlow Dataset creation: Train or validation directories not found.")
            print(f"Expected train directory: {train_dir}")
            print(f"Expected validation directory: {val_dir}")

    except Exception as e:
        print(f"\nAn error occurred during TensorFlow Dataset creation: {e}")
        print("TensorFlow Dataset creation failed.")


if train_ds is not None and val_ds is not None and loaded_keras_model is not None:
    try:
        model_to_fine_tune = loaded_keras_model

        # Add a new classification head on top of the VGG16 base model
        # This is done here to allow fine-tuning of the combined model if needed.
        # If you only want to fine-tune the base, this head should be added earlier.
        # Let's add a simple head for fine-tuning purposes if the base model is loaded.
        # Check if a head is already added (e.g., if loaded_keras_model is the full model)
        # Assuming loaded_keras_model is just the feature extractor without the final dense layers.
        # If loaded_keras_model is the full model, skip adding the head here.

        # Simple check: if the output shape is the feature vector size, assume no head.
        # If output shape is (None, num_classes), assume head is present.
        if loaded_keras_model.output_shape[-1] != num_classes:
             print("\nAdding classification head for fine-tuning...")
             x = loaded_keras_model.output
             # Add Dropout and Dense layers for the classification head
             x = Dropout(0.5)(x) # Example Dropout rate
             predictions = Dense(num_classes, activation='softmax')(x) # Use softmax for multi-class classification

             # Create the full fine-tuning model
             model_to_fine_tune = Model(inputs=loaded_keras_model.input, outputs=predictions)
             print("Classification head added.")

        # Unfreeze the base model for fine-tuning
        # Identify the layer to unfreeze from (e.g., block4_conv1 or earlier layers)
        # Inspect model_to_fine_tune.summary() to pick the layer name.
        # Example: Unfreeze from 'block4_conv1' onwards
        unfreeze_from_layer = 'block4_conv1' # Adjust this layer name based on VGG16 structure

        print(f"\nUnfreezing layers from '{unfreeze_from_layer}' onwards for fine-tuning...")
        set_trainable = False
        for layer in model_to_fine_tune.layers:
            if layer.name == unfreeze_from_layer:
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False

        # Re-compile the model after making layers trainable
        print("Re-compiling model for fine-tuning...")
        model_to_fine_tune.compile(optimizer=Adam(learning_rate=LEARNING_RATE_FINE_TUNE),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # Define callbacks for fine-tuning
        checkpoint_callback_fine_tune = ModelCheckpoint(filepath=keras_model_path_fine_tuned,
                                                        save_weights_only=True,
                                                        save_best_only=True,
                                                        monitor='val_accuracy',
                                                        verbose=1)
        early_stopping_callback_fine_tune = EarlyStopping(monitor='val_accuracy',
                                                          patience=5,
                                                          verbose=1,
                                                          restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=3, verbose=1)

        print(f"Starting fine-tuning for {EPOCHS_FINE_TUNE} epochs...")
        history_fine_tune = model_to_fine_tune.fit(
          train_ds,
          validation_data=val_ds,
          epochs=EPOCHS_FINE_TUNE,
          callbacks=[checkpoint_callback_fine_tune, early_stopping_callback_fine_tune, reduce_lr]
        )

        print("\nCNN Fine-tuning Finished.")

    except Exception as e:
        print(f"\nAn error occurred during CNN fine-tuning: {e}")
        print("CNN fine-tuning failed.")

else:
    print("\nSkipping CNN Fine-tuning: TensorFlow Datasets (train_ds, val_ds) or Keras model not available.")