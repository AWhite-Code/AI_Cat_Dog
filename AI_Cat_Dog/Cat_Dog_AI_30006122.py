#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Ensure the 'plots' directory exists for saving figures
os.makedirs('plots', exist_ok=True)

def create_tf_dataset(dataset_dir, categories, img_size=(128, 128), batch_size=32, shuffle=True, subset=None):
    """
    Creates a TensorFlow dataset for image classification with optional validation split.
    
    Parameters:
    - dataset_dir: Path to the dataset directory.
    - categories: List of category names.
    - img_size: Tuple specifying the size to which images will be resized.
    - batch_size: Number of samples per batch.
    - shuffle: Whether to shuffle the dataset.
    - subset: 'training' or 'validation' for splitting.
    
    Returns:
    - dataset: A TensorFlow dataset object.
    """
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
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=True)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(np.arange(len(categories)) + 0.5, categories, rotation=45)
    plt.yticks(np.arange(len(categories)) + 0.5, categories, rotation=0)
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

def train_cnn_model(train_ds, val_ds, test_ds, categories):
    """
    Builds, trains, and evaluates a Convolutional Neural Network (CNN).
    
    Parameters:
    - train_ds: Training dataset.
    - val_ds: Validation dataset.
    - test_ds: Testing dataset.
    - categories: List of category names.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(categories), activation='softmax')
    ])
    
    # Compile the CNN
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Early Stopping Callback
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
    
    # Train the CNN
    print("Training CNN...")
    history = model.fit(
        train_ds,
        epochs=10,
        validation_data=val_ds,
        callbacks=[early_stopping]
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
    
    print("Loading data...")
    train_dataset = create_tf_dataset(train_dir, categories, subset='training')
    validation_dataset = create_tf_dataset(train_dir, categories, subset='validation')
    test_dataset = create_tf_dataset(test_dir, categories, shuffle=False)
    
    # Extract all training data for Decision Tree
    print("Extracting data for Decision Tree...")
    X_train_flat = []
    y_train_flat = []
    for _ in range(len(train_dataset)):
        X_batch, y_batch = train_dataset.next()
        X_train_flat.extend(X_batch.reshape(X_batch.shape[0], -1))
        y_train_flat.extend(y_batch)
    X_train_flat = np.array(X_train_flat)
    y_train_flat = np.array(y_train_flat)
    
    # Extract all testing data for Decision Tree
    print("Extracting data for Decision Tree...")
    X_test_flat = []
    y_test_flat = []
    for _ in range(len(test_dataset)):
        X_batch, y_batch = test_dataset.next()
        X_test_flat.extend(X_batch.reshape(X_batch.shape[0], -1))
        y_test_flat.extend(y_batch)
    X_test_flat = np.array(X_test_flat)
    y_test_flat = np.array(y_test_flat)
    
    # Apply PCA for Decision Tree
    print("Applying PCA for Decision Tree...")
    X_train_pca, X_test_pca = apply_pca(X_train_flat, X_test_flat, n_components=100)
    
    # Execute based on user choice
    if choice == 1:
        train_decision_tree(X_train_pca, y_train_flat, X_test_pca, y_test_flat, categories)
    elif choice == 2:
        train_cnn_model(train_dataset, validation_dataset, test_dataset, categories)
    elif choice == 3:
        train_decision_tree(X_train_pca, y_train_flat, X_test_pca, y_test_flat, categories)
        train_cnn_model(train_dataset, validation_dataset, test_dataset, categories)

if __name__ == "__main__":
    main()
