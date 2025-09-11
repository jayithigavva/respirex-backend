"""
FastAPI backend for Respiratory Disease Classification
Updated to use trained models
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import librosa
import numpy as np
import pandas as pd
import io
import json
import logging
import os
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Respiratory Disease Classification API",
    description="API for classifying respiratory diseases using trained models",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
TARGET_SR = 22050
DURATION = 10.0
N_MELS = 128
HOP_LENGTH = 512

# Disease classes
DISEASE_CLASSES = [
    'Asthma', 'Bronchiectasis', 'Bronchiolitis', 'COPD', 
    'Healthy', 'LRTI', 'Pneumonia', 'URTI'
]

# Global variables for models
model1 = None
model2 = None
device = None

# Model 1: Disease Classifier
class RespiratoryDiseaseClassifier(nn.Module):
    """
    CNN model for respiratory disease classification.
    """
    
    def __init__(self, 
                 input_height: int = 128,
                 input_width: int = 431,
                 num_disease_classes: int = 8,
                 dropout_rate: float = 0.3):
        super(RespiratoryDiseaseClassifier, self).__init__()
        
        self.input_height = input_height
        self.input_width = input_width
        self.num_disease_classes = num_disease_classes
        
        # CNN encoder
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 2))
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 2))
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d((2, 2))
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d((2, 2))
        
        # Calculate CNN output size
        self.cnn_output_size = self._get_cnn_output_size()
        
        # Disease classification head
        self.disease_head = nn.Sequential(
            nn.Linear(self.cnn_output_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_disease_classes)
        )
        
    def _get_cnn_output_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.input_height, self.input_width)
            x = self.pool1(torch.relu(self.bn1(self.conv1(dummy_input))))
            x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
            x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
            x = self.pool4(torch.relu(self.bn4(self.conv4(x))))
            return x.size(1) * x.size(2) * x.size(3)
    
    def forward(self, x):
        # CNN feature extraction
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool4(torch.relu(self.bn4(self.conv4(x))))
        
        # Flatten for classification
        x = x.view(x.size(0), -1)
        
        # Disease classification
        disease_logits = self.disease_head(x)
        
        return disease_logits

# Model 2: Doctor-Assisted Annotation Model
class DoctorAnnotationModel(nn.Module):
    """
    Pure annotation-based model for doctor-assisted disease prediction.
    """
    
    def __init__(self, 
                 annotation_dim: int = 100,
                 num_disease_classes: int = 8,
                 dropout_rate: float = 0.3):
        super(DoctorAnnotationModel, self).__init__()
        
        self.annotation_dim = annotation_dim
        self.num_disease_classes = num_disease_classes
        
        # Annotation feature encoder
        self.annotation_encoder = nn.Sequential(
            nn.Linear(annotation_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Disease prediction head
        self.disease_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_disease_classes)
        )
        
    def forward(self, annotation_features):
        # Encode annotation features
        annotation_encoded = self.annotation_encoder(annotation_features)
        
        # Predict disease
        disease_logits = self.disease_head(annotation_encoded)
        
        return disease_logits

def load_models():
    """
    Load the trained models.
    """
    global model1, model2, device
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load Model 1: Disease Classifier
        model1_path = 'models/model1_disease_event_classifier.pth'
        if os.path.exists(model1_path):
            model1 = RespiratoryDiseaseClassifier(
                input_height=128,
                input_width=431,
                num_disease_classes=8,
                dropout_rate=0.3
            ).to(device)
            
            model1.load_state_dict(torch.load(model1_path, map_location=device))
            model1.eval()
            logger.info("✅ Model 1 (Disease Classifier) loaded successfully")
        else:
            logger.warning(f"❌ Model 1 not found at {model1_path}")
        
        # Load Model 2: Doctor-Assisted Annotation Model
        model2_path = 'models/model2_doctor_assisted.pth'
        if os.path.exists(model2_path):
            model2 = DoctorAnnotationModel(
                annotation_dim=100,
                num_disease_classes=8,
                dropout_rate=0.3
            ).to(device)
            
            model2.load_state_dict(torch.load(model2_path, map_location=device))
            model2.eval()
            logger.info("✅ Model 2 (Doctor-Assisted Annotation) loaded successfully")
        else:
            logger.warning(f"❌ Model 2 not found at {model2_path}")
        
        return model1 is not None, model2 is not None
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False, False

def preprocess_audio(audio_data: bytes) -> Dict[str, Any]:
    """
    Preprocess audio data for model inference.
    """
    try:
        # Load audio from bytes
        audio, sr = librosa.load(io.BytesIO(audio_data), sr=TARGET_SR)
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
        
        # Pad or truncate to target duration
        target_length = int(DURATION * TARGET_SR)
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=TARGET_SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return {
            'audio': audio,
            'mel_spectrogram': mel_spec_db,
            'duration': DURATION,
            'sample_rate': TARGET_SR
        }
        
    except Exception as e:
        logger.error(f"Error preprocessing audio: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Audio preprocessing failed: {str(e)}")

def create_annotation_features_from_csv(annotation_data: str) -> torch.Tensor:
    """
    Create annotation features from CSV data (for Model 2).
    """
    try:
        # Parse CSV data
        lines = annotation_data.strip().split('\n')
        annotations = []
        
        for line in lines[1:]:  # Skip header
            parts = line.split(',')
            if len(parts) >= 8:
                start = float(parts[0])
                end = float(parts[1])
                crackles = int(parts[2])
                wheezes = int(parts[3])
                annotations.append({
                    'start': start,
                    'end': end,
                    'crackles': crackles,
                    'wheezes': wheezes
                })
        
        # Create time bins (100 bins for 10 seconds)
        num_bins = 100
        bin_duration = 10.0 / num_bins
        features = np.zeros(num_bins)
        
        # Process each annotation
        for ann in annotations:
            start_time = ann['start']
            end_time = ann['end']
            crackles = ann['crackles']
            wheezes = ann['wheezes']
            
            # Convert to bin indices
            start_bin = int(start_time / bin_duration)
            end_bin = int(end_time / bin_duration)
            
            # Ensure bins are within range
            start_bin = max(0, min(start_bin, num_bins - 1))
            end_bin = max(0, min(end_bin, num_bins - 1))
            
            # Set features based on annotations
            if crackles > 0:
                features[start_bin:end_bin+1] += 1  # Crackle indicator
            if wheezes > 0:
                features[start_bin:end_bin+1] += 2  # Wheeze indicator
        
        return torch.FloatTensor(features)
        
    except Exception as e:
        logger.error(f"Error creating annotation features: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Annotation processing failed: {str(e)}")

# Load models on startup
models_loaded = load_models()

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Respiratory Disease Classification API",
        "version": "2.0.0",
        "status": "running",
        "models_loaded": {
            "model1_disease_classifier": models_loaded[0],
            "model2_doctor_assisted": models_loaded[1]
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": {
            "model1_disease_classifier": models_loaded[0],
            "model2_doctor_assisted": models_loaded[1]
        },
        "device": str(device)
    }

@app.post("/predict_disease")
async def predict_disease_endpoint(file: UploadFile = File(...)):
    """
    Predict respiratory disease from uploaded audio file using Model 1.
    
    Args:
        file: Audio file (.wav, .mp3, etc.)
        
    Returns:
        JSON response with disease prediction
    """
    try:
        if not models_loaded[0]:
            raise HTTPException(status_code=503, detail="Model 1 not loaded")
        
        # Validate file type
        if not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Read file content
        audio_data = await file.read()
        
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        logger.info(f"Processing disease prediction for: {file.filename}, size: {len(audio_data)} bytes")
        
        # Preprocess audio
        features = preprocess_audio(audio_data)
        
        # Prepare input tensor
        mel_spec_tensor = torch.FloatTensor(features['mel_spectrogram']).unsqueeze(0).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            logits = model1(mel_spec_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Create class probabilities dictionary
        class_probabilities = {
            DISEASE_CLASSES[i]: float(probabilities[0][i].item())
            for i in range(len(DISEASE_CLASSES))
        }
        
        # Prepare response
        response = {
            "success": True,
            "filename": file.filename,
            "prediction": DISEASE_CLASSES[predicted_class],
            "confidence": confidence,
            "class_probabilities": class_probabilities,
            "audio_info": {
                "duration": features['duration'],
                "sample_rate": features['sample_rate']
            }
        }
        
        logger.info(f"Disease prediction completed: {DISEASE_CLASSES[predicted_class]} (confidence: {confidence:.3f})")
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during disease prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/predict_annotation")
async def predict_annotation_endpoint(file: UploadFile = File(...)):
    """
    Predict disease from doctor's annotation data using Model 2.
    
    Args:
        file: CSV file with annotation data
        
    Returns:
        JSON response with disease prediction
    """
    try:
        if not models_loaded[1]:
            raise HTTPException(status_code=503, detail="Model 2 not loaded")
        
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV file")
        
        # Read file content
        annotation_data = await file.read()
        
        if len(annotation_data) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        logger.info(f"Processing annotation prediction for: {file.filename}, size: {len(annotation_data)} bytes")
        
        # Create annotation features
        annotation_features = create_annotation_features_from_csv(annotation_data.decode('utf-8'))
        annotation_tensor = annotation_features.unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            logits = model2(annotation_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Create class probabilities dictionary
        class_probabilities = {
            DISEASE_CLASSES[i]: float(probabilities[0][i].item())
            for i in range(len(DISEASE_CLASSES))
        }
        
        # Prepare response
        response = {
            "success": True,
            "filename": file.filename,
            "disease": DISEASE_CLASSES[predicted_class],
            "confidence": confidence,
            "class_probabilities": class_probabilities,
            "annotation_info": {
                "num_annotations": len(annotation_data.decode('utf-8').split('\n')) - 1,
                "model_type": "Doctor-Assisted Annotation Model"
            }
        }
        
        logger.info(f"Annotation prediction completed: {DISEASE_CLASSES[predicted_class]} (confidence: {confidence:.3f})")
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during annotation prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)