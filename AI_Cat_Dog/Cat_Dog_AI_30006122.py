#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imread
from skimage.transform import resize
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras import layers, models

# Function to load images from a dataset directory and convert them into a usable format
def load_images_and_labels(categories, dataset_dir, img_size=(128, 128)):
    data = []
    labels = []
    for category in categories:
        # Full path to the category folder
        path = os.path.join(dataset_dir, category)
        class_num = categories.index(category)
        for img_name in os.listdir(path):
            try:
                # Read and preprocess the image
                img_path = os.path.join(path, img_name)
                img = imread(img_path)
                img = resize(img, img_size, anti_aliasing=True)
                img = img.flatten()  # Convert the matrix to a flat array
                data.append(img)
                labels.append(class_num)
            except Exception as e:
                print(f"An error occurred with {img_name}: {e}")
    return np.array(data), np.array(labels)


""" 
Create the confusion matrix
Top left: True Positive for Cats (Model guessed cat correctly)
Top right: False Negative for Cats (Model guessed dog when it was actually a cat)
Bottom Left: False Positive for Cats (Model guessed cat when it was actually a dog)
Bottom Right: True Negative for Cats (Model guessed dog correctly)
"""
def plot_confusion_matrix(conf_matrix, categories, model_name):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=True)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(np.arange(len(categories)) + 0.5, categories, rotation=45) # X axis is angled at 45 degrees to make it slightly easier to read
    plt.yticks(np.arange(len(categories)) + 0.5, categories, rotation=0)
    plt.show()

# Prepare data for training and testing
categories = ['cats', 'dogs']
train_dir = 'dataset/training_set'
test_dir = 'dataset/test_set'
print("Test")

# Load the datasets
print("Loading data...")
X_train, y_train = load_images_and_labels(categories, train_dir)
X_test, y_test = load_images_and_labels(categories, test_dir)

# Train a decision tree classifier
clf = tree.DecisionTreeClassifier()
print("Training decision tree...")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Accuracy: {accuracy}")
plot_confusion_matrix(confusion_matrix(y_test, y_pred), categories, 'Decision Tree')

# Build and train a CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(categories), activation='softmax')
])

# Compile and train the CNN
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Training CNN...")
history = model.fit(X_train.reshape((-1, 128, 128, 3)), y_train, epochs=10, validation_split=0.2)

# Evaluate the CNN
print("Evaluating CNN...")
test_loss, test_accuracy = model.evaluate(X_test.reshape((-1, 128, 128, 3)), y_test)
print(f"CNN Test Accuracy: {test_accuracy}")
y_pred_cnn = np.argmax(model.predict(X_test.reshape((-1, 128, 128, 3))), axis=-1)
plot_confusion_matrix(confusion_matrix(y_test, y_pred_cnn), categories, 'CNN')

# Plot the training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training History')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:




