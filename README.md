# Assignment11-CLY

Assignment 11 — CIFAR‑10 Image Classification with Random Forest
This project implements an end‑to‑end machine learning pipeline for classifying images from the CIFAR‑10 dataset using a Random Forest classifier. The workflow includes preprocessing, hyperparameter tuning, model evaluation, feature importance visualization, and prediction on external test images.

Dataset
CIFAR‑10 contains 60,000 color images (32×32 pixels) across 10 balanced classes:

airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Training set: 50,000 images

Test set: 10,000 images

Image size: 32×32×3

Format: RGB

Preprocessing
The preprocessing pipeline includes:

Normalizing pixel values to the 0–1 range

Flattening each image into a 3,072‑element vector

Standardizing features using StandardScaler

Preparing train/test matrices for modeling

Final shapes:

Code
X_train: (50000, 3072)
X_test:  (10000, 3072)
y:       10 classes

Model: Random Forest Classifier
A Random Forest model was tuned using GridSearchCV (3‑fold CV).
To keep runtime manageable, a reduced parameter grid was used:

python
{
    'n_estimators': [50],
    'max_depth': [None, 20],
    'min_samples_split': [2],
    'min_samples_leaf': [1]
}

Best Parameters
Code
{'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}

Model Performance
Accuracy: ~44.7%
This is expected for Random Forests on CIFAR‑10 due to the lack of spatial awareness in tree‑based models.

Classification Report
Balanced performance across classes, with stronger results on visually distinct categories (e.g., ships, airplanes) and more confusion among animals (e.g., cats, dogs, deer).

Feature Importance
Feature importances were extracted from the trained model.
Because each feature corresponds to a single pixel in the flattened image, importances appear noisy — a known limitation of Random Forests on image data.

Predictions on External Images
A custom prediction function was implemented to classify new images from GitHub.
The function:

Downloads the image

Resizes to 32×32

Normalizes and flattens

Applies the same scaler

Predicts using the trained model

Test Images
Five external images were tested.

Example:

Cat → predicted as deer (a common RF confusion and a fun one)
