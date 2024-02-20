# Install MediaPipe library, which offers tools for face landmark detection
!pip install -q mediapipe

# Import necessary libraries
from google.colab import drive
import os
import cv2
import pickle
import mediapipe as mp
import torch

# Check if a GPU is available for computations. If not, raise an error
if not torch.cuda.is_available():
    raise SystemError('GPU device not found')
print('Found GPU:', torch.cuda.get_device_name())

# Mount Google Drive to access and store files directly from/to Drive
drive.mount('/content/drive')

# Define a function to gather facial landmarks using MediaPipe's face mesh model
def gather_landmark_data(directory):
    # Initialize MediaPipe's face mesh module
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    # Dictionary to store the path of each image as the key and its landmarks as the value
    landmark_data = {}

    # Iterate over each class directory (e.g., different facial expressions)
    for class_name in os.listdir(directory):
        # Iterate over each image in the class directory
        for filename in os.listdir(os.path.join(directory, class_name)):
            img_path = os.path.join(directory, class_name, filename)
            # Read the image using OpenCV
            img = cv2.imread(img_path)
            # Convert the image from BGR to RGB format
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Process the RGB image to get face landmarks
            results = face_mesh.process(img_rgb)

            # Check if landmarks are detected for the face in the image
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                coords = []
                # Extract x, y, z coordinates for each landmark
                for landmark in landmarks.landmark:
                    coords.append((landmark.x, landmark.y, landmark.z))
                # Store landmarks in the dictionary
                landmark_data[img_path] = coords
            else:
                # Print a message if no face is detected in the image
                print(f"No face detected in {img_path}. Skipping...")

    # Close the face mesh module
    face_mesh.close()
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
output_filepath = '/content/drive/MyDrive/facial_expressions/code/models/mediapipe/mp_landmark_data.pkl'
save_landmark_data(landmark_data, output_filepath)

