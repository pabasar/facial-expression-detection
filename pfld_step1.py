# Install necessary libraries
!pip install -q torch torchvision
!git clone https://github.com/polarisZhao/PFLD-pytorch.git

# Change to the directory of the cloned repository
import os
os.chdir('/content/PFLD-pytorch/')

# Import required modules
from google.colab import drive
import cv2
import pickle
import torch
import torchvision.transforms as transforms
from PIL import Image

# Ensure GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mount Google Drive to access and store files directly from/to Drive
drive.mount('/content/drive')

# Import PFLD model
from models.pfld import PFLDInference

# Load the PFLD model
def load_pfld_model(model_path):
    model = PFLDInference().to(device)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'pfld_backbone' in checkpoint:
            model.load_state_dict(checkpoint['pfld_backbone'])
            model.eval()
        else:
            print("The expected key 'pfld_backbone' is not found in the checkpoint.")
            return None
    else:
        print("Model not found at the specified path.")
        return None

    return model

# Define function to get facial landmarks using PFLD
def gather_landmark_data(directory, model):
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])
    landmark_data = {}

    for class_name in os.listdir(directory):
        for filename in os.listdir(os.path.join(directory, class_name)):
            img_path = os.path.join(directory, class_name, filename)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)

            _, landmarks = model(img_tensor)
            landmarks = landmarks.squeeze().cpu().detach().numpy().reshape(-1, 2)

            # Store landmarks in dictionary
            landmark_data[img_path] = landmarks.tolist()

    return landmark_data

# Define function to save landmark data
def save_landmark_data(landmark_data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(landmark_data, f)

# Specify path to the PFLD pre-trained model and load it
pfld_model_path = '/content/PFLD-pytorch/checkpoint/snapshot/checkpoint.pth.tar'
pfld_model = load_pfld_model(pfld_model_path)

# If model is loaded, proceed further
if pfld_model:
    # Specify the directory containing the facial images
    directory = '/content/drive/MyDrive/facial_expressions/code/dataset'
    # Gather landmarks for each image in the directory
    landmark_data = gather_landmark_data(directory, pfld_model)

    # Define the output path for the pickle file and save the landmark data
    output_filepath = '/content/drive/MyDrive/facial_expressions/code/models/pfld/pfld_landmark_data.pkl'
    save_landmark_data(landmark_data, output_filepath)

