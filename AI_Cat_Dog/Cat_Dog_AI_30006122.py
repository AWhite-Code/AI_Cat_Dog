#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Ensure the 'plots' directory exists for saving figures
os.makedirs('plots', exist_ok=True)

def create_tf_dataset(dataset_dir, categories, img_size=(128, 128), batch_size=32, shuffle=True, subset=None, augment=False):
    """
    Creates a TensorFlow dataset for image classification with optional validation split and augmentation.
    
    Parameters:
    - dataset_dir: Path to the dataset directory.
    - categories: List of category names.
    - img_size: Tuple specifying the size to which images will be resized.
    - batch_size: Number of samples per batch.
    - shuffle: Whether to shuffle the dataset.
    - subset: 'training' or 'validation' for splitting.
    - augment: Whether to apply data augmentation.
    
    Returns:
    - dataset: A DirectoryIterator object.
    """
    if augment:
        datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    else:
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    dataset = datagen.flow_from_directory(
        directory=dataset_dir,
        target_size=img_size,
        color_mode='rgb',
        classes=categories,
        class_mode='sparse',
        batch_size=batch_size,
        shuffle=shuffle,
        subset=subset
    )
    print(f"Class Indices for {dataset_dir}: {dataset.class_indices}")
    return dataset

def plot_confusion_matrix(conf_matrix, categories, model_name):
    """
    Plot and save the confusion matrix as a heatmap.
    
    Parameters:
    - conf_matrix: Confusion matrix array.
    - categories: List of category names.
    - model_name: Name of the model (for title and filename).
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=True,
                xticklabels=categories, yticklabels=categories)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plot_filename = f'plots/{model_name.lower().replace(" ", "_")}_confusion_matrix.png'
    plt.savefig(plot_filename)
    plt.close()
    print(f"Confusion matrix saved as {plot_filename}")

def plot_training_history(history):
    """
    Plot and save the training history of the CNN.
    
    Parameters:
    - history: History object returned by model.fit().
    """
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training History')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plot_filename = 'plots/cnn_training_history.png'
    plt.savefig(plot_filename)
    plt.close()
    print(f"Training history plot saved as {plot_filename}")

def apply_pca(X_train, X_test, n_components=100):
    """
    Applies Principal Component Analysis (PCA) to reduce dimensionality.
    
    Parameters:
    - X_train: Training data features.
    - X_test: Testing data features.
    - n_components: Number of principal components.
    
    Returns:
    - X_train_pca: Transformed training data.
    - X_test_pca: Transformed testing data.
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca

def compute_class_weights(y_train):
    """
    Computes class weights to handle class imbalance.
    
    Parameters:
    - y_train: Training data labels.
    
    Returns:
    - class_weights: Dictionary mapping class indices to weights.
    """
    class_weights_array = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = {i: weight for i, weight in enumerate(class_weights_array)}
    print(f"Computed class weights: {class_weights}")
    return class_weights

def train_decision_tree(X_train_pca, y_train, X_test_pca, y_test, categories):
    """
    Train and evaluate a Decision Tree classifier.
    
    Parameters:
    - X_train_pca: PCA-transformed training data features.
    - y_train: Training data labels.
    - X_test_pca: PCA-transformed testing data features.
    - y_test: Testing data labels.
    - categories: List of category names.
    """
    clf = tree.DecisionTreeClassifier(max_depth=10, min_samples_split=10, random_state=42)
    print("Training Decision Tree...")
    clf.fit(X_train_pca, y_train)
    y_pred = clf.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Decision Tree Accuracy: {accuracy:.4f}")
    conf_matrix = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(conf_matrix, categories, 'Decision Tree')

def extract_data(directory_iterator):
    """
    Extracts all data from a DirectoryIterator.

    Parameters:
    - directory_iterator: A DirectoryIterator object.

    Returns:
    - X: Numpy array of flattened image data.
    - y: Numpy array of labels.
    """
    X = []
    y = []
    steps = len(directory_iterator)
    for _ in range(steps):
        X_batch, y_batch = next(directory_iterator)
        X.extend(X_batch.reshape(X_batch.shape[0], -1))
        y.extend(y_batch)
    return np.array(X), np.array(y)

def train_cnn_model(train_ds, val_ds, test_ds, categories, class_weights):
    """
    Builds, trains, and evaluates a Convolutional Neural Network (CNN).
    
    Parameters:
    - train_ds: Training dataset.
    - val_ds: Validation dataset.
    - test_ds: Testing dataset.
    - categories: List of category names.
    - class_weights: Dictionary mapping class indices to weights.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(categories), activation='softmax')
    ])
    
    # Compile the CNN
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Early Stopping Callback
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    
    # Train the CNN
    print("Training CNN...")
    history = model.fit(
        train_ds,
        epochs=30,
        validation_data=val_ds,
        callbacks=[early_stopping],
        class_weight=class_weights  # Apply class weights
    )
    
    # Evaluate the CNN
    print("Evaluating CNN...")
    test_loss, test_accuracy = model.evaluate(test_ds)
    print(f"CNN Test Accuracy: {test_accuracy:.4f}")
    
    # Predict and plot confusion matrix
    y_pred_cnn = model.predict(test_ds)
    y_pred_cnn = np.argmax(y_pred_cnn, axis=-1)
    conf_matrix = confusion_matrix(test_ds.classes, y_pred_cnn)
    plot_confusion_matrix(conf_matrix, categories, 'CNN')
    
    # Plot training history
    plot_training_history(history)

def visualize_samples(generator, categories, num_samples=5):
    """
    Visualizes sample images from the generator.
    
    Parameters:
    - generator: DirectoryIterator object.
    - categories: List of category names.
    - num_samples: Number of samples to visualize.
    """
    X, y = next(generator)
    for i in range(num_samples):
        plt.imshow(X[i])
        plt.title(categories[int(y[i])])
        plt.axis('off')
        plt.show()

def check_class_distribution(y, categories):
    """
    Prints the distribution of classes.
    
    Parameters:
    - y: Numpy array of labels.
    - categories: List of category names.
    """
    unique, counts = np.unique(y, return_counts=True)
    distribution = dict(zip(unique, counts))

def main():
    # Options Menu
    print("Select the model to train and evaluate:")
    print("1. Decision Tree")
    print("2. CNN")
    print("3. Both Decision Tree and CNN")
    
    while True:
        try:
            choice = int(input("Enter your choice (1, 2, or 3): "))
            if choice in [1, 2, 3]:
                break
            else:
                print("Invalid input. Please enter 1, 2, or 3.")
        except ValueError:
            print("Invalid input. Please enter a numerical value (1, 2, or 3).")
    
    # Prepare data for training and testing
    categories = ['cats', 'dogs']
    train_dir = 'dataset/training_set'
    test_dir = 'dataset/test_set'
    
    # Create separate generators for Decision Tree
    print("\nLoading data for Decision Tree...")
    train_dataset_dt = create_tf_dataset(train_dir, categories, subset='training', shuffle=False, augment=False)
    test_dataset_dt = create_tf_dataset(test_dir, categories, shuffle=False, augment=False)
    
    # Create separate generators for CNN
    print("\nLoading data for CNN...")
    train_dataset_cnn = create_tf_dataset(train_dir, categories, subset='training', shuffle=True, augment=True)
    validation_dataset_cnn = create_tf_dataset(train_dir, categories, subset='validation', shuffle=False, augment=False)
    test_dataset_cnn = create_tf_dataset(test_dir, categories, shuffle=False, augment=False)
    
    # Extract all training data for Decision Tree
    print("\nExtracting data for Decision Tree...")
    X_train_flat, y_train_flat = extract_data(train_dataset_dt)
    print("Training Data Distribution:")
    check_class_distribution(y_train_flat, categories)
    
    # Extract all testing data for Decision Tree
    print("\nExtracting data for Decision Tree...")
    X_test_flat, y_test_flat = extract_data(test_dataset_dt)
    print("Testing Data Distribution:")
    check_class_distribution(y_test_flat, categories)
    
    # Apply PCA for Decision Tree
    print("\nApplying PCA for Decision Tree...")
    X_train_pca, X_test_pca = apply_pca(X_train_flat, X_test_flat, n_components=100)
    
    # Compute class weights for CNN
    print("\nComputing class weights for CNN...")
    class_weights = compute_class_weights(y_train_flat)
    
    # Execute based on user choice
    if choice == 1:
        train_decision_tree(X_train_pca, y_train_flat, X_test_pca, y_test_flat, categories)
    elif choice == 2:
        train_cnn_model(train_dataset_cnn, validation_dataset_cnn, test_dataset_cnn, categories, class_weights)
    elif choice == 3:
        train_decision_tree(X_train_pca, y_train_flat, X_test_pca, y_test_flat, categories)
        train_cnn_model(train_dataset_cnn, validation_dataset_cnn, test_dataset_cnn, categories, class_weights)

if __name__ == "__main__":
    main()
