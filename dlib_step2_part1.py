# Install the required libraries
!pip install -q dlib

# Import necessary modules
from google.colab import drive
import pickle
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import itertools
import torch

# Check for GPU availability and print its name
if not torch.cuda.is_available():
    raise SystemError('GPU device not found')
print('Found GPU:', torch.cuda.get_device_name())

# Mount the Google Drive to access the model file and images
drive.mount('/content/drive')

# Load the landmark data extracted using dlib
def load_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Convert the landmarks dictionary into arrays for data and their corresponding labels
def convert_dict_to_arrays(data_dict):
    data = []
    labels = []
    for img_path, landmarks in data_dict.items():
        data.append(np.array(landmarks).flatten())
        labels.append(img_path.split("/")[-2])
    return np.asarray(data), np.asarray(labels)

# Split the dataset into training and test sets
def split_data(data, labels, test_size=0.2):
    return train_test_split(data, labels, test_size=test_size, shuffle=True, stratify=labels)

# Tune hyperparameters for the XGBoost model
def tune_hyperparameters(x_train, y_train):
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'tree_method': ['gpu_hist']  # GPU training
    }

    xgb = XGBClassifier(objective='multi:softprob', eval_metric='mlogloss')
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
    grid_search.fit(x_train, y_train)

    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

    return grid_search.best_estimator_

# Evaluate the model's performance
def evaluate_model(model, x_test, y_test):
    y_predict = model.predict(x_test)
    accuracy = accuracy_score(y_predict, y_test)
    report = classification_report(y_test, y_predict)
    matrix = confusion_matrix(y_test, y_predict)
    return accuracy, report, matrix

# Plot the model's confusion matrix
def plot_confusion_matrix(matrix, labels):
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, format(matrix[i, j], 'd'), horizontalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# Display the results including accuracy, confusion matrix, and classification report
def plot_metrics(matrix, labels, accuracy, report):
    plt.figure(figsize=(10,5))
    plot_confusion_matrix(matrix, labels)
    plt.show()
    print(f"Classification Accuracy: {accuracy*100:.2f}%")
    print("\nClassification Report:\n", report)

# Save the trained model
def save_model(model, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump({'model': model}, f)

# Main code
try:
    # Load data, convert it, and split into training and test sets
    data_dict = load_data('/content/drive/MyDrive/facial_expressions/code/models/dlib/dlib_landmark_data.pkl')
    data, labels = convert_dict_to_arrays(data_dict)
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    label_classes = label_encoder.classes_
    x_train, x_test, y_train, y_test = split_data(data, labels_encoded)

    # Hyperparameter tuning, evaluation, and save the model
    best_model = tune_hyperparameters(x_train, y_train)
    accuracy, report, matrix = evaluate_model(best_model, x_test, y_test)
    plot_metrics(matrix, label_classes, accuracy, report)
    save_model(best_model, '/content/drive/MyDrive/facial_expressions/code/models/dlib/dlib_model.p')

except Exception as e:
    print("An error occurred:", str(e))

