# Install Dlib library
!pip install -q dlib

# Import necessary libraries
from google.colab import drive
import os
import cv2
import pickle
import dlib

# Mount Google Drive to access and store files directly from/to Drive
drive.mount('/content/drive')

# Initialize Dlib's face detector and the 68 landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/content/drive/MyDrive/facial_expressions/code/models/dlib/shape_predictor_68_face_landmarks.dat')

# Define a function to gather facial landmarks using Dlib's predictor model
def gather_landmark_data(directory):
    # Dictionary to store the path of each image as the key and its landmarks as the value
    landmark_data = {}

    # Iterate over each class directory
    for class_name in os.listdir(directory):
        # Iterate over each image in the class directory
        for filename in os.listdir(os.path.join(directory, class_name)):
            img_path = os.path.join(directory, class_name, filename)
            # Read the image using OpenCV
            img = cv2.imread(img_path)
            # Convert the image from BGR to RGB format (Dlib uses RGB)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Detect faces in the image
            faces = detector(img_rgb)

            # Check if a face is detected in the image
            if faces:
                shape = predictor(img_rgb, faces[0])
                coords = [(point.x, point.y) for point in shape.parts()]
                # Store landmarks in the dictionary
                landmark_data[img_path] = coords
            else:
                # Print a message if no face is detected in the image
                print(f"No face detected in {img_path}. Skipping...")

    return landmark_data

# Define a function to save the landmark data to a file using pickle
def save_landmark_data(landmark_data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(landmark_data, f)

# Specify the directory containing the facial images
directory = '/content/drive/MyDrive/facial_expressions/code/dataset'
# Gather landmarks for each image in the specified directory
landmark_data = gather_landmark_data(directory)

# Define the output path for the pickle file and save the landmark data
output_filepath = '/content/drive/MyDrive/facial_expressions/code/models/dlib/dlib_landmark_data.pkl'
save_landmark_data(landmark_data, output_filepath)

