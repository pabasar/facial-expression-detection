# Install the required libraries
!pip install -q dlib

# Import necessary modules
import pickle
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import itertools
import torch
from google.colab import drive

# Check if a GPU is available and print its name, if found
if not torch.cuda.is_available():
    raise SystemError('GPU device not found')
print('Found GPU:', torch.cuda.get_device_name())

# Mount Google Drive for accessing files
drive.mount('/content/drive')

# Function to load facial landmark data from a given path
def load_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Convert landmarks dictionary to arrays of data and corresponding labels
def convert_dict_to_arrays(data_dict):
    data = []
    labels = []
    for img_path, landmarks in data_dict.items():
        data.append(np.array(landmarks).flatten())
        labels.append(img_path.split("/")[-2])  # Extract class name (label) from the path structure
    return np.asarray(data), np.asarray(labels)

# Split dataset into training and testing sets using stratified sampling
def split_data(data, labels, test_size=0.2):
    return train_test_split(data, labels, test_size=test_size, shuffle=True, stratify=labels)

# Train the model using the best hyperparameters identified
def train_model_with_best_params(x_train, y_train):
    model = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        colsample_bytree=1.0,
        learning_rate=0.05,
        max_depth=5,
        n_estimators=100,
        subsample=0.8,
        tree_method='gpu_hist'  # Use GPU for training
    )
    model.fit(x_train, y_train)
    return model

# Evaluate the trained model and return metrics like accuracy, classification report, and confusion matrix
def evaluate_model(model, x_test, y_test):
    y_predict = model.predict(x_test)
    accuracy = accuracy_score(y_predict, y_test)
    report = classification_report(y_test, y_predict)
    matrix = confusion_matrix(y_test, y_predict)
    return accuracy, report, matrix

# Plot the confusion matrix for visualization of model's performance
def plot_confusion_matrix(matrix, labels, title='Confusion Matrix'):
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, format(matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# Display metrics like confusion matrix, accuracy, and the classification report
def plot_metrics(matrix, labels, accuracy, report):
    plt.figure(figsize=(10,5))
    plot_confusion_matrix(matrix, labels)
    plt.show()
    print(f"Classification Accuracy: {accuracy*100:.2f}%")
    print("\nClassification Report:\n", report)

# Save the trained model to a specified path
def save_model(model, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump({'model': model}, f)

# Main code execution
try:
    # Load facial landmark data from the .pkl file
    data_dict = load_data('/content/drive/MyDrive/facial_expressions/code/models/dlib/dlib_landmark_data.pkl')
    data, labels = convert_dict_to_arrays(data_dict)

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    label_classes = label_encoder.classes_
    x_train, x_test, y_train, y_test = split_data(data, labels_encoded)

    model = train_model_with_best_params(x_train, y_train)
    accuracy, report, matrix = evaluate_model(model, x_test, y_test)
    plot_metrics(matrix, label_classes, accuracy, report)

    save_model(model, '/content/drive/MyDrive/facial_expressions/code/models/dlib/dlib_model.p')

except Exception as e:
    print("An error occurred:", str(e))

