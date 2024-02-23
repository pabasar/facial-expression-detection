# Install necessary libraries
!pip install dlib 

# Importing required modules
from google.colab import drive
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2
import dlib

# Mount Google Drive to access the model file and images
drive.mount('/content/drive')

# Load the pre-trained expression prediction model from the drive
def initialize_model():
    with open('/content/drive/MyDrive/facial_expressions/code/models/dlib/dlib_model.p', 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model']

# Set up dlib's face detector and landmark predictor
def initialize_dlib_models():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('/content/drive/MyDrive/facial_expressions/code/models/dlib/shape_predictor_68_face_landmarks.dat')
    return detector, predictor

# Predict the expression using an image and overlay landmarks on the image
def predict_expression(img_path, clf, detector, predictor):
    label_to_expression = {
        "0": "Anger",
        "1": "Contempt",
        "2": "Disgust",
        "3": "Fear",
        "4": "Happy",
        "5": "Neutral",
        "6": "Sad",
        "7": "Surprise"
    }

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if not faces:
        print(f"No face detected in {img_path}.")
        return

    img_with_landmarks = img.copy()
    landmarks_list = []
    for face in faces:
        shape = predictor(img, face)
        landmark_data = []
        for point in shape.parts():
            landmark_data.extend([point.x, point.y])
            cv2.circle(img_with_landmarks, (point.x, point.y), 1, (0, 255, 0), -1)
        landmarks_list.append(landmark_data)

    for landmark_data in landmarks_list:
        prediction_label = str(clf.predict([landmark_data])[0])
        confidence = np.max(clf.predict_proba([landmark_data]))
        prediction_expression = label_to_expression[prediction_label]
        display_result(img_with_landmarks, prediction_expression, confidence)

def display_result(img_with_landmarks, prediction_expression, confidence):
    dpi = 550 / 6
    plt.figure(dpi=dpi)
    plt.imshow(cv2.cvtColor(img_with_landmarks, cv2.COLOR_BGR2RGB))
    plt.title(f"{prediction_expression} ({confidence*100:.2f}%)")
    plt.axis('off')
    plt.show()

# Initialize the expression prediction model and the dlib models
clf = initialize_model()
detector, predictor = initialize_dlib_models()

# Test the model using an image from the drive
predict_expression('/content/drive/MyDrive/facial_expressions/code/testing/happy/ffhq_0.png', clf, detector, predictor)


