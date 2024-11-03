#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imread
from skimage.transform import resize
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras import layers, models

# Ensure the 'plots' directory exists for saving figures
os.makedirs('plots', exist_ok=True)

def load_images_and_labels(categories, dataset_dir, img_size=(128, 128)):
    """
    Load images from the dataset directory, preprocess them, and assign labels.

    Parameters:
    - categories: List of category names.
    - dataset_dir: Path to the dataset directory.
    - img_size: Tuple specifying the size to which images will be resized.

    Returns:
    - data: Numpy array of image data.
    - labels: Numpy array of corresponding labels.
    """
    data = []
    labels = []
    for category in categories:
        path = os.path.join(dataset_dir, category)
        class_num = categories.index(category)
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = imread(img_path)
                # If image has an alpha channel, remove it
                if img.shape[-1] == 4:
                    img = img[..., :3]
                img = resize(img, img_size, anti_aliasing=True)
                data.append(img)
                labels.append(class_num)
            except Exception as e:
                print(f"An error occurred with {img_name}: {e}")
    return np.array(data), np.array(labels)

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
    # Save the figure
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
    # Save the figure
    plot_filename = 'plots/cnn_training_history.png'
    plt.savefig(plot_filename)
    plt.close()
    print(f"Training history plot saved as {plot_filename}")

def train_decision_tree(X_train, y_train, X_test, y_test, categories):
    """
    Train and evaluate a Decision Tree classifier.

    Parameters:
    - X_train: Training data features.
    - y_train: Training data labels.
    - X_test: Testing data features.
    - y_test: Testing data labels.
    - categories: List of category names.
    """
    clf = tree.DecisionTreeClassifier()
    print("Training Decision Tree...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Decision Tree Accuracy: {accuracy:.4f}")
    conf_matrix = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(conf_matrix, categories, 'Decision Tree')

def train_cnn(X_train, y_train, X_test, y_test, categories):
    """
    Build, train, and evaluate a Convolutional Neural Network (CNN).

    Parameters:
    - X_train: Training data features.
    - y_train: Training data labels.
    - X_test: Testing data features.
    - y_test: Testing data labels.
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
    
    print("Training CNN...")
    history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)
    
    # Evaluate the CNN
    print("Evaluating CNN...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"CNN Test Accuracy: {test_accuracy:.4f}")
    y_pred_cnn = np.argmax(model.predict(X_test), axis=-1)
    conf_matrix = confusion_matrix(y_test, y_pred_cnn)
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
    X_train, y_train = load_images_and_labels(categories, train_dir)
    X_test, y_test = load_images_and_labels(categories, test_dir)
    
    # Reshape data for CNN if needed
    # Ensure that X_train and X_test have shape (-1, 128, 128, 3)
    if len(X_train.shape) == 2:
        X_train = X_train.reshape(-1, 128, 128, 3)
    if len(X_test.shape) == 2:
        X_test = X_test.reshape(-1, 128, 128, 3)
    
    # Execute based on user choice
    if choice == 1:
        train_decision_tree(X_train, y_train, X_test, y_test, categories)
    elif choice == 2:
        train_cnn(X_train, y_train, X_test, y_test, categories)
    elif choice == 3:
        train_decision_tree(X_train, y_train, X_test, y_test, categories)
        train_cnn(X_train, y_train, X_test, y_test, categories)

if __name__ == "__main__":
    main()
