# Install the required libraries: MediaPipe for processing facial landmarks and torch for PyTorch
!pip install -q mediapipe

# Import the necessary libraries and modules
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

# Check if a GPU is available for computations. If not, raise an error
if not torch.cuda.is_available():
    raise SystemError('GPU device not found')
print('Found GPU:', torch.cuda.get_device_name())

# Mount the Google Drive to access files stored there
drive.mount('/content/drive')

# Function to load landmark data from a specified path
def load_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Convert the landmarks dictionary to data arrays and their corresponding labels
def convert_dict_to_arrays(data_dict):
    data = []
    labels = []
    for img_path, landmarks in data_dict.items():
        data.append(np.array(landmarks).flatten())
        # Extract the disease label from the file path
        labels.append(img_path.split("/")[-2])
    return np.asarray(data), np.asarray(labels)

# Split the dataset into training and testing sets
def split_data(data, labels, test_size=0.2):
    return train_test_split(data, labels, test_size=test_size, shuffle=True, stratify=labels)

# Function to tune hyperparameters for the XGBoost model using GridSearch
def tune_hyperparameters(x_train, y_train):
    # Define possible values for hyperparameters
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'tree_method': ['gpu_hist']  # Use GPU for training
    }

    xgb = XGBClassifier(objective='multi:softprob', eval_metric='mlogloss')

    # Use GridSearchCV for finding the best hyperparameters
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
    grid_search.fit(x_train, y_train)
    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
    return grid_search.best_estimator_

# Function to evaluate the model's performance on the test set
def evaluate_model(model, x_test, y_test):
    y_predict = model.predict(x_test)
    accuracy = accuracy_score(y_predict, y_test)
    report = classification_report(y_test, y_predict)
    matrix = confusion_matrix(y_test, y_predict)
    return accuracy, report, matrix

# Plot a confusion matrix to visualize the classification results
def plot_confusion_matrix(matrix, labels, title='Confusion Matrix'):
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    # Display the actual counts inside the matrix cells
    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, format(matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# Display accuracy, confusion matrix, and classification report
def plot_metrics(matrix, labels, accuracy, report):
    plt.figure(figsize=(10,5))
    plot_confusion_matrix(matrix, labels)
    plt.show()
    print(f"Classification Accuracy: {accuracy*100:.2f}%")
    print("\nClassification Report:\n", report)

# Save the trained model to a specified path for future use
def save_model(model, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump({'model': model}, f)

# Main execution begins here
try:
    # Load landmark data, convert to suitable format, and split into training and test sets
    data_dict = load_data('/content/drive/MyDrive/facial_expressions/code/models/mediapipe/mp_landmark_data.pkl')
    data, labels = convert_dict_to_arrays(data_dict)
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    label_classes = label_encoder.classes_
    x_train, x_test, y_train, y_test = split_data(data, labels_encoded)

    # Tune model, evaluate it, and save the trained model
    best_model = tune_hyperparameters(x_train, y_train)
    accuracy, report, matrix = evaluate_model(best_model, x_test, y_test)
    plot_metrics(matrix, label_classes, accuracy, report)
    save_model(best_model, '/content/drive/MyDrive/facial_expressions/code/models/mediapipe/mp_model.p')

# Handle any errors or exceptions
except Exception as e:
    print("An error occurred:", str(e))

