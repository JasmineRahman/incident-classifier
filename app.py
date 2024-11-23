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
# from flask import Flask, request, jsonify
# from flask_cors import CORS  # Import CORS
# import torch
# import torch.nn as nn  # Import nn module
# from torchvision import transforms, models  # Import models from torchvision
# from PIL import Image
# import io
# import os

# # Define the model
# class IncidentClassifier(nn.Module):
#     def __init__(self, num_classes):
#         super(IncidentClassifier, self).__init__()
#         # Use ResNet50 as the base model
#         self.model = models.resnet50(weights='IMAGENET1K_V1')
        
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

# # Define class names
# class_names = [
#     "Building Fires",
#     "Forest Fires",
#     "Industrial Fires",
#     "Vehicle Fires",
#     "Potholes",
#     "Public Hygiene Issues"
# ]

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
        
#         # Return the class name instead of index
#         class_index = predicted.item()
#         class_name = class_names[class_index]
        
#         return jsonify({'class_index': class_index, 'class_name': class_name})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))


# from flask import Flask, request, jsonify
# from flask_cors import CORS, cross_origin
# import torch
# import torch.nn as nn
# from torchvision import transforms, models
# from PIL import Image
# import io
# import logging
# import os
# import sys

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s [%(levelname)s] %(message)s',
#     handlers=[logging.StreamHandler(sys.stdout)]
# )
# logger = logging.getLogger(__name__)

# # Define the model
# class IncidentClassifier(nn.Module):
#     def __init__(self, num_classes):
#         super(IncidentClassifier, self).__init__()
#         self.model = models.resnet50(pretrained=True)
#         num_features = self.model.fc.in_features
#         self.model.fc = nn.Sequential(
#             nn.Linear(num_features, 512),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(512, num_classes)
#         )
        
#     def forward(self, x):
#         return self.model(x)

# def load_model(model_path, num_classes):
#     try:
#         model = IncidentClassifier(num_classes)
#         if not os.path.exists(model_path):
#             logger.error(f"Model file not found at {model_path}")
#             raise FileNotFoundError(f"Model file not found at {model_path}")
        
#         model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
#         model.eval()
#         logger.info("Model loaded successfully")
#         return model
#     except Exception as e:
#         logger.error(f"Error loading model: {str(e)}")
#         raise

# class_names = [
#     "Building Fires",
#     "Forest Fires",
#     "Industrial Fires",
#     "Vehicle Fires",
#     "Potholes",
#     "Public Hygiene Issues"
# ]

# # Initialize Flask app
# app = Flask(__name__)
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# # Configure CORS
# CORS(app, resources={
#     r"/*": {
#         "origins": "*",
#         "methods": ["GET", "POST", "OPTIONS"],
#         "allow_headers": ["Content-Type", "Authorization"]
#     }
# })

# # Initialize model globally
# try:
#     model = load_model('best_model.pth', num_classes=6)
# except Exception as e:
#     logger.error(f"Failed to load model at startup: {str(e)}")
#     model = None

# @app.route('/', methods=['GET'])
# @cross_origin()
# def home():
#     return "Welcome to the Incident Classifier API!"

# @app.route('/favicon.ico', methods=['GET'])
# @cross_origin()
# def favicon():
#     return '', 204

# @app.route('/health', methods=['GET'])
# @cross_origin()
# def health_check():
#     return jsonify({
#         'status': 'healthy',
#         'model_loaded': model is not None
#     })

# @app.route('/predict', methods=['POST', 'OPTIONS'])
# @cross_origin()
# def predict():
#     # Handle preflight requests
#     if request.method == 'OPTIONS':
#         return '', 204
    
#     try:
#         logger.info("Received prediction request")
        
#         # Log request headers for debugging
#         logger.info(f"Request Headers: {dict(request.headers)}")
#         logger.info(f"Request Method: {request.method}")
        
#         # Check if model is loaded
#         if model is None:
#             logger.error("Model not initialized")
#             return jsonify({'error': 'Model not initialized'}), 503

#         # Check if file exists in request
#         if 'image' not in request.files:
#             logger.error("No image file in request")
#             return jsonify({'error': 'No image file provided'}), 400

#         file = request.files['image']
        
#         # Validate file
#         if file.filename == '':
#             logger.error("Empty filename")
#             return jsonify({'error': 'No selected file'}), 400

#         # Read and validate image
#         img_bytes = file.read()
#         if not img_bytes:
#             logger.error("Empty file content")
#             return jsonify({'error': 'Empty file'}), 400

#         # Process image
#         try:
#             image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
#         except Exception as e:
#             logger.error(f"Error opening image: {str(e)}")
#             return jsonify({'error': 'Invalid image file'}), 400

#         # Transform image
#         transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

#         image = transform(image).unsqueeze(0)

#         # Make prediction
#         try:
#             with torch.no_grad():
#                 outputs = model(image)
#             _, predicted = torch.max(outputs, 1)
#             class_index = predicted.item()
#             class_name = class_names[class_index]
            
#             logger.info(f"Successful prediction: {class_name}")
#             return jsonify({
#                 'class_index': class_index,
#                 'class_name': class_name,
#                 'status': 'success'
#             })

#         except Exception as e:
#             logger.error(f"Error during prediction: {str(e)}")
#             return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

#     except Exception as e:
#         logger.error(f"Unexpected error: {str(e)}")
#         return jsonify({'error': f'Internal server error: {str(e)}'}), 500

# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 5000))
#     app.run(host='0.0.0.0', port=port)


from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import logging
import os
import sys
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Define the model
class IncidentClassifier(nn.Module):
    def __init__(self, num_classes):
        super(IncidentClassifier, self).__init__()
        self.model = models.resnet50(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

def download_model(model_url, destination_path):
    try:
        logger.info("Downloading model...")
        response = requests.get(model_url, stream=True)
        response.raise_for_status()  # Will raise an exception for HTTP errors
        with open(destination_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=128):
                f.write(chunk)
        logger.info(f"Model downloaded successfully to {destination_path}")
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise

def load_model(model_url, destination_path, num_classes):
    try:
        # Download model if not present
        if not os.path.exists(destination_path):
            logger.info("Model file not found locally, downloading...")
            download_model(model_url, destination_path)

        model = IncidentClassifier(num_classes)
        model.load_state_dict(torch.load(destination_path, map_location=torch.device('cpu')))
        model.eval()
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# Define class names for prediction output
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
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# Configure CORS
CORS(app, resources={ r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type", "Authorization"]}})

# Model URL (Google Drive direct download link)
model_url = "https://drive.google.com/file/d/1UFsAKp4vbPP_LnX1Sokn_aI9wLG7FGAa/view?usp=drive_link"
model_path = "best_model.pth"

# Initialize model globally
try:
    model = load_model(model_url, model_path, num_classes=6)
except Exception as e:
    logger.error(f"Failed to load model at startup: {str(e)}")
    model = None

@app.route('/', methods=['GET'])
@cross_origin()
def home():
    return "Welcome to the Incident Classifier API!"

@app.route('/favicon.ico', methods=['GET'])
@cross_origin()
def favicon():
    return '', 204

@app.route('/health', methods=['GET'])
@cross_origin()
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST', 'OPTIONS'])
@cross_origin()
def predict():
    # Handle preflight requests
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        logger.info("Received prediction request")
        
        # Log request headers for debugging
        logger.info(f"Request Headers: {dict(request.headers)}")
        logger.info(f"Request Method: {request.method}")
        
        # Check if model is loaded
        if model is None:
            logger.error("Model not initialized")
            return jsonify({'error': 'Model not initialized'}), 503

        # Check if file exists in request
        if 'image' not in request.files:
            logger.error("No image file in request")
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        
        # Validate file
        if file.filename == '':
            logger.error("Empty filename")
            return jsonify({'error': 'No selected file'}), 400

        # Read and validate image
        img_bytes = file.read()
        if not img_bytes:
            logger.error("Empty file content")
            return jsonify({'error': 'Empty file'}), 400

        # Process image
        try:
            image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        except Exception as e:
            logger.error(f"Error opening image: {str(e)}")
            return jsonify({'error': 'Invalid image file'}), 400

        # Transform image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = transform(image).unsqueeze(0)

        # Make prediction
        try:
            with torch.no_grad():
                outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            class_index = predicted.item()
            class_name = class_names[class_index]
            
            logger.info(f"Successful prediction: {class_name}")
            return jsonify({
                'class_index': class_index,
                'class_name': class_name,
                'status': 'success'
            })

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
