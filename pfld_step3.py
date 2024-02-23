# Install necessary libraries
!pip install -q torch torchvision
!git clone https://github.com/polarisZhao/PFLD-pytorch.git

# Change to the directory of the cloned repository
import os
os.chdir('/content/PFLD-pytorch/')

# Importing required modules
from google.colab import drive
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

# Load the PFLD model
from models.pfld import PFLDInference

# Mount Google Drive to access the model file and images
drive.mount('/content/drive')

# Load the pre-trained expression prediction model from the drive
def initialize_model():
    # Open and read the model file
    with open('/content/drive/MyDrive/facial_expressions/code/models/pfld/pfld_model.p', 'rb') as f:
        model_data = pickle.load(f)
    # Return the loaded model
    return model_data['model']

# Initialize the PFLD model
def initialize_pfld_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PFLDInference().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['pfld_backbone'])
    model.eval()
    return model

# Predict the expression using an image and overlay landmarks on the image
def predict_expression(img_path, clf, pfld_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    transform = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor()])
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    _, landmarks = pfld_model(img_tensor)
    landmarks = landmarks.squeeze().cpu().detach().numpy().reshape(-1, 2)

    landmark_data = landmarks.flatten()

    prediction_label = str(clf.predict([landmark_data])[0])
    confidence = np.max(clf.predict_proba([landmark_data]))
    prediction_expression = label_to_expression[prediction_label]

    # Convert PIL Image to NumPy array for visualization and landmark plotting
    img_np = np.array(img)
    for (x, y) in landmarks:
        cv2.circle(img_np, (int(x * img_np.shape[1]), int(y * img_np.shape[0])), 1, (0, 255, 0), -1)

    display_result(img_np, prediction_expression, confidence)


def display_result(img_with_landmarks, prediction_expression, confidence):
    dpi = 550 / 6
    plt.figure(dpi=dpi)
    plt.imshow(img_with_landmarks)
    plt.title(f"{prediction_expression} ({confidence*100:.2f}%)")
    plt.axis('off')
    plt.show()

# Initialize the expression prediction model
clf = initialize_model()
# Initialize the PFLD model and provide path to the checkpoint
pfld_model_path = '/content/PFLD-pytorch/checkpoint/snapshot/checkpoint.pth.tar'
pfld_model = initialize_pfld_model(pfld_model_path)

# Test the model using an image from the drive
predict_expression('/content/drive/MyDrive/facial_expressions/code/testing/happy/ffhq_0.png', clf, pfld_model)


