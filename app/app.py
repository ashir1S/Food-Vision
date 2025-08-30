### 1. Imports and class names setup ###
import gradio as gr
import os
import torch
# import pickle # We'll save class names as a text file
import torchvision.transforms as transforms # Using transforms from torchvision
from PIL import Image
from typing import Tuple, Dict
from timeit import default_timer as timer

# Import the model creation function
from model import create_foodvision_model

# Setup class names - Load from a text file
# Assuming class_names.txt is in the same directory as app.py (will save it later)
class_names_save_path = 'class_names.txt'
class_names = []
try:
    with open(class_names_save_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"Classes list loaded successfully from {class_names_save_path}")
except FileNotFoundError:
    print(f"Error: Classes file not found at {class_names_save_path}. Please ensure the file exists.")
    # Provide a fallback or raise an error if classes are essential
    # Fallback to placeholder names if loading fails
    class_names = [f"Class {i}" for i in range(101)] # Using a default of 101 classes
    print("Using placeholder class names.")
except Exception as e:
    print(f"An error occurred while loading the classes list: {e}")
    # Fallback to placeholder names
    class_names = [f"Class {i}" for i in range(101)] # Using a default of 101 classes
    print("Using placeholder class names.")


### 2. Model and transforms preparation ###

# Create Food Vision model (using the function from model.py)
num_classes = len(class_names) if class_names and class_names[0] != "Class 0" else 101 # Use loaded classes or default
loaded_model = create_foodvision_model(num_classes=num_classes)

# Load saved weights
# Assuming the model file is in the same directory as app.py (copied earlier)
# Find the latest checkpoint file in the current directory (where app.py will be)
def find_latest_ckpt_in_dir(model_dir="."):
    files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    if not files:
        return None
    # Choose newest by file mtime
    files = sorted(files, key=lambda fn: os.path.getmtime(os.path.join(model_dir, fn)), reverse=True)
    return os.path.join(model_dir, files[0])

model_save_path = find_latest_ckpt_in_dir(".") # Look for the latest .pth file in the current directory

# Check if the model file exists before loading
if model_save_path and os.path.exists(model_save_path):
    try:
        # Load the state dictionary
        checkpoint = torch.load(model_save_path, map_location=torch.device("cpu"))

        # If the checkpoint is a dict with 'model_state_dict' (our format), load that
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
             loaded_model.load_state_dict(checkpoint["model_state_dict"])
             print(f"Model state dictionary loaded successfully from checkpoint dict in {model_save_path}")
        # Otherwise, assume it's just a raw state_dict
        elif isinstance(checkpoint, dict):
             loaded_model.load_state_dict(checkpoint)
             print(f"Model state dictionary loaded successfully from raw state_dict in {model_save_path}")
        else:
             print(f"Warning: Checkpoint format not recognized in {model_save_path}. Attempting to load as raw state_dict.")
             loaded_model.load_state_dict(checkpoint) # Try loading directly


        print(f"Model weights loaded from {model_save_path}")

    except Exception as e:
        print(f"An error occurred while loading the model state dictionary from {model_save_path}: {e}")
        print("Model will use default initialized weights.")
else:
    print(f"Error: Model file not found in the current directory ({os.getcwd()}). Please ensure a .pth model file is present.")
    print("Model will use default initialized weights.")


# Define transforms (using test transforms)
# These should match the test_transform used during training/evaluation
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(), # Convert PIL Image to PyTorch Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
])


### 3. Predict function ###

# Create predict function
def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer
    start_time = timer()

    # Ensure img is a PIL Image before applying transforms
    if not isinstance(img, Image.Image):
        try:
            img = Image.fromarray(img) # Attempt to convert if it's a numpy array or similar
        except:
             print("Warning: Input is not a PIL Image. Attempting prediction anyway.")


    # Transform the target image and add a batch dimension
    # Use the defined test_transform
    img = test_transform(img).unsqueeze(0)

    # Put model into evaluation mode and turn on inference mode
    loaded_model.eval() # Use the loaded_model
    # Use CPU for inference as map_location was set to cpu
    device = torch.device("cpu")
    loaded_model.to(device)

    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        img = img.to(device) # Move image to the device
        pred_probs = torch.softmax(loaded_model(img), dim=1).squeeze(0) # Use the loaded_model and remove batch dimension

    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    if class_names: # Ensure class_names is not empty
        pred_labels_and_probs = {class_names[i]: float(pred_probs[i]) for i in range(len(class_names))}
    else:
        # Fallback if class names weren't loaded
        pred_labels_and_probs = {f"Class {i}": float(pred_probs[i]) for i in range(len(pred_probs))}
        print("Using placeholder class names for predictions.")


    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)

    # Return the prediction dictionary and prediction time
    return pred_labels_and_probs, pred_time

### 4. Gradio app ###

# Create title, description and article strings
title = "Food Vision Model 🍔"
description = "A Vision Transformer (ViT) model trained to classify images of food."
article = "Built using PyTorch and Gradio." # You can update this with a link to your notebook or project if desired.

# Create examples list from "examples/" directory
# This assumes the examples directory is a subdirectory of where app.py is located
example_list = [["examples/" + example] for example in os.listdir("examples") if example.endswith(('.jpg', '.jpeg', '.png'))]


# Create the Gradio demo
demo = gr.Interface(fn=predict, # mapping function from input to output
                    inputs=gr.Image(type="pil"), # what are the inputs?
                    outputs=[gr.Label(num_top_classes=5, label="Predictions"), # what are the outputs?
                             gr.Number(label="Prediction time (s)")], # our fn has two outputs, therefore we have two outputs
                    # Create examples list from "examples/" directory
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)

# Launch the demo!
if __name__ == "__main__":
    demo.launch()
