# from flask import Flask, request, jsonify
# from flask_cors import CORS  # Import CORS
# import torch
# import torch.nn as nn  # Import nn module
# from torchvision import transforms, models  # Import models from torchvision
# from PIL import Image
# import io

# # Define the model
# class IncidentClassifier(nn.Module):
#     def __init__(self, num_classes):
#         super(IncidentClassifier, self).__init__()
#         # Use ResNet50 as the base model
#         self.model = models.resnet50(pretrained=True)
        
#         # Replace the final fully connected layer
#         num_features = self.model.fc.in_features
#         self.model.fc = nn.Sequential(
#             nn.Linear(num_features, 512),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(512, num_classes)
#         )
        
#     def forward(self, x):
#         return self.model(x)

# # Load model function
# def load_model(model_path, num_classes):
#     model = IncidentClassifier(num_classes)
#     model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
#     model.eval()
#     return model

# # Load the model (ensure the path is correct)
# model = load_model('best_model.pth', num_classes=6)

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# # Home route
# @app.route('/')
# def home():
#     return "Welcome to the Incident Classifier API!"

# # Favicon route
# @app.route('/favicon.ico')
# def favicon():
#     return '', 204  # No content for favicon

# # Prediction route
# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         file = request.files['image']
#         img_bytes = file.read()
#         image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

#         transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

#         image = transform(image).unsqueeze(0)  # Add batch dimension

#         with torch.no_grad():
#             outputs = model(image)
#         _, predicted = torch.max(outputs, 1)
#         return jsonify({'class': predicted.item()})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

#app.py
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import torch
import torch.nn as nn  # Import nn module
from torchvision import transforms, models  # Import models from torchvision
from PIL import Image
import io
import os

# Define the model
class IncidentClassifier(nn.Module):
    def __init__(self, num_classes):
        super(IncidentClassifier, self).__init__()
        # Use ResNet50 as the base model
        self.model = models.resnet50(weights='IMAGENET1K_V1')
        
        # Replace the final fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

# Load model function
def load_model(model_path, num_classes):
    model = IncidentClassifier(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Load the model (ensure the path is correct)
model = load_model('best_model.pth', num_classes=6)

# Define class names
class_names = [
    "Building Fires",
    "Forest Fires",
    "Industrial Fires",
    "Vehicle Fires",
    "Potholes",
    "Public Hygiene Issues"
]

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Home route
@app.route('/')
def home():
    return "Welcome to the Incident Classifier API!"

# Favicon route
@app.route('/favicon.ico')
def favicon():
    return '', 204  # No content for favicon

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = transform(image).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        
        # Return the class name instead of index
        class_index = predicted.item()
        class_name = class_names[class_index]
        
        return jsonify({'class_index': class_index, 'class_name': class_name})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
