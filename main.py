"""
FastAPI backend for Respiratory Disease Classification
Supports both Disease Classification and Annotation Models
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import soundfile as sf
import io
import json
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Respiratory Disease Classification API",
    description="API for classifying respiratory diseases and detecting anomalies from audio files",
    version="1.0.0"
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
N_MFCC = 13
N_MELS = 128
HOP_LENGTH = 512

# Disease classes
DISEASE_CLASSES = [
    'Asthma', 'Bronchiectasis', 'Bronchiolitis', 'COPD', 
    'Healthy', 'LRTI', 'Pneumonia', 'URTI'
]

# Event classes for annotation model
EVENT_CLASSES = ['Normal', 'Crackle', 'Wheeze', 'Both']

class RespiratoryCNN(nn.Module):
    """
    CNN architecture for respiratory disease classification.
    """
    
    def __init__(self, 
                 input_height: int = 128,
                 input_width: int = 431,
                 num_classes: int = 8,
                 dropout_rate: float = 0.3):
        super(RespiratoryCNN, self).__init__()
        
        self.input_height = input_height
        self.input_width = input_width
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Calculate flattened size
        self.flattened_size = self._get_flattened_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, num_classes)
        
    def _get_flattened_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.input_height, self.input_width)
            x = self.pool1(torch.relu(self.bn1(self.conv1(dummy_input))))
            x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
            x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
            x = self.pool4(torch.relu(self.bn4(self.conv4(x))))
            return x.numel()
    
    def forward(self, x):
        # Convolutional layers
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool4(torch.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

class MultiTaskRespiratoryModel(nn.Module):
    """
    Multi-task model for respiratory event annotation and disease classification.
    """
    
    def __init__(self,
                 input_height: int = 128,
                 input_width: int = 431,
                 num_event_classes: int = 4,
                 num_disease_classes: int = 8,
                 dropout_rate: float = 0.3):
        super(MultiTaskRespiratoryModel, self).__init__()
        
        self.input_height = input_height
        self.input_width = input_width
        self.num_event_classes = num_event_classes
        self.num_disease_classes = num_disease_classes
        
        # Shared CNN encoder
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 1))  # Pool only in frequency dimension
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 1))
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d((2, 1))
        
        # Calculate CNN output size
        self.cnn_output_size = self._get_cnn_output_size()
        
        # Temporal convolution for sequence modeling
        self.temporal_conv = nn.Conv1d(self.cnn_output_size, 256, kernel_size=3, padding=1)
        self.temporal_bn = nn.BatchNorm1d(256)
        
        # Task-specific heads
        # Event detection head (sequence labeling)
        self.event_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_event_classes)
        )
        
        # Disease classification head
        self.disease_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_disease_classes)
        )
        
    def _get_cnn_output_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.input_height, self.input_width)
            x = self.pool1(torch.relu(self.bn1(self.conv1(dummy_input))))
            x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
            x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
            return x.size(1) * x.size(2)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN feature extraction
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        
        # Reshape for temporal convolution
        x = x.view(batch_size, self.cnn_output_size, -1)
        
        # Temporal convolution
        x = torch.relu(self.temporal_bn(self.temporal_conv(x)))
        
        # Transpose for task heads
        x = x.transpose(1, 2)  # (batch_size, seq_len, features)
        
        # Event detection (sequence labeling)
        event_logits = self.event_head(x)
        
        # Disease classification (global pooling)
        global_features = torch.mean(x, dim=1)  # Average pooling over time
        disease_logits = self.disease_head(global_features)
        
        return event_logits, disease_logits

# Global model variables
disease_model = None
annotation_model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_models():
    """Load both trained models."""
    global disease_model, annotation_model
    
    try:
        # Load Disease Classification Model
        disease_model = RespiratoryCNN(
            input_height=128,
            input_width=431,
            num_classes=len(DISEASE_CLASSES),
            dropout_rate=0.3
        )
        disease_model.load_state_dict(torch.load('models/disease_classifier.pth', map_location=device))
        disease_model.to(device)
        disease_model.eval()
        logger.info("Disease classification model loaded successfully")
        
        # Load Annotation Model
        annotation_model = MultiTaskRespiratoryModel(
            input_height=128,
            input_width=431,
            num_event_classes=len(EVENT_CLASSES),
            num_disease_classes=len(DISEASE_CLASSES),
            dropout_rate=0.3
        )
        annotation_model.load_state_dict(torch.load('models/annotation_model.pth', map_location=device))
        annotation_model.to(device)
        annotation_model.eval()
        logger.info("Annotation model loaded successfully")
        
        return True
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False

def preprocess_audio(audio_data: bytes) -> Dict[str, Any]:
    """
    Preprocess audio data for model inference.
    
    Args:
        audio_data: Raw audio bytes
        
    Returns:
        Dictionary containing processed features
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
        
        # Extract features
        mfccs = librosa.feature.mfcc(y=audio, sr=TARGET_SR, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=TARGET_SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return {
            'audio': audio,
            'mfccs': mfccs,
            'mel_spectrogram': mel_spec_db,
            'duration': DURATION,
            'sample_rate': TARGET_SR
        }
        
    except Exception as e:
        logger.error(f"Error preprocessing audio: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Audio preprocessing failed: {str(e)}")

def predict_disease(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict disease from audio features using the disease classification model.
    
    Args:
        features: Preprocessed audio features
        
    Returns:
        Dictionary containing prediction results
    """
    global disease_model
    
    if disease_model is None:
        raise HTTPException(status_code=500, detail="Disease model not loaded")
    
    try:
        # Prepare input tensor
        mel_spec = features['mel_spectrogram']
        mel_spec_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        mel_spec_tensor = mel_spec_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = disease_model(mel_spec_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
        # Get results
        predicted_disease = DISEASE_CLASSES[predicted_class.item()]
        confidence_score = confidence.item()
        
        # Get all class probabilities
        all_probabilities = probabilities.squeeze().cpu().numpy()
        class_probabilities = {
            DISEASE_CLASSES[i]: float(all_probabilities[i]) 
            for i in range(len(DISEASE_CLASSES))
        }
        
        return {
            'predicted_disease': predicted_disease,
            'confidence': confidence_score,
            'class_probabilities': class_probabilities
        }
        
    except Exception as e:
        logger.error(f"Error during disease prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Disease prediction failed: {str(e)}")

def predict_annotation(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict annotations and disease from audio features using the annotation model.
    
    Args:
        features: Preprocessed audio features
        
    Returns:
        Dictionary containing annotation results
    """
    global annotation_model
    
    if annotation_model is None:
        raise HTTPException(status_code=500, detail="Annotation model not loaded")
    
    try:
        # Prepare input tensor
        mel_spec = features['mel_spectrogram']
        mel_spec_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        mel_spec_tensor = mel_spec_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            event_logits, disease_logits = annotation_model(mel_spec_tensor)
            
            # Get disease prediction
            disease_probabilities = torch.softmax(disease_logits, dim=1)
            disease_confidence, predicted_disease_class = torch.max(disease_probabilities, 1)
            
            # Get event predictions
            event_probabilities = torch.softmax(event_logits, dim=-1)
            event_predictions = event_probabilities.argmax(dim=-1)
            
        # Process disease results
        predicted_disease = DISEASE_CLASSES[predicted_disease_class.item()]
        disease_confidence_score = disease_confidence.item()
        
        # Process event results
        sequence_length = event_predictions.size(1)
        frame_rate = TARGET_SR / HOP_LENGTH
        
        # Convert frame predictions to time segments
        events = []
        current_event = None
        
        for frame_idx in range(sequence_length):
            event_class = event_predictions[0, frame_idx].item()
            event_name = EVENT_CLASSES[event_class]
            
            if event_name != 'Normal':
                current_time = frame_idx / frame_rate
                
                if current_event is None or current_event['label'] != event_name:
                    # Start new event
                    if current_event is not None:
                        events.append(current_event)
                    current_event = {
                        'start': current_time,
                        'end': current_time,
                        'label': event_name.lower(),
                        'confidence': float(event_probabilities[0, frame_idx, event_class])
                    }
                else:
                    # Extend current event
                    current_event['end'] = current_time
                    current_event['confidence'] = max(current_event['confidence'], 
                                                   float(event_probabilities[0, frame_idx, event_class]))
            else:
                # End current event
                if current_event is not None:
                    events.append(current_event)
                    current_event = None
        
        # Add final event if exists
        if current_event is not None:
            events.append(current_event)
        
        # Filter events by minimum duration and confidence
        filtered_events = []
        for event in events:
            duration = event['end'] - event['start']
            if duration >= 0.1 and event['confidence'] >= 0.3:  # Minimum 100ms duration and 30% confidence
                filtered_events.append({
                    'start': round(event['start'], 2),
                    'end': round(event['end'], 2),
                    'label': event['label'],
                    'confidence': round(event['confidence'], 3)
                })
        
        return {
            'predicted_disease': predicted_disease,
            'confidence': disease_confidence_score,
            'events': filtered_events
        }
        
    except Exception as e:
        logger.error(f"Error during annotation prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Annotation prediction failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    logger.info("Starting up Respiratory Disease Classification API...")
    if not load_models():
        logger.error("Failed to load models on startup")
    else:
        logger.info("API ready for predictions")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Respiratory Disease Classification API",
        "version": "1.0.0",
        "status": "running",
        "models_loaded": {
            "disease_classifier": disease_model is not None,
            "annotation_model": annotation_model is not None
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": {
            "disease_classifier": disease_model is not None,
            "annotation_model": annotation_model is not None
        },
        "device": str(device)
    }

@app.post("/predict_disease")
async def predict_disease_endpoint(file: UploadFile = File(...)):
    """
    Predict respiratory disease from uploaded audio file.
    
    Args:
        file: Audio file (.wav, .mp3, etc.)
        
    Returns:
        JSON response with disease prediction
    """
    try:
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
        
        # Predict disease
        prediction = predict_disease(features)
        
        # Prepare response
        response = {
            "success": True,
            "filename": file.filename,
            "prediction": prediction['predicted_disease'],
            "confidence": prediction['confidence'],
            "class_probabilities": prediction['class_probabilities'],
            "audio_info": {
                "duration": features['duration'],
                "sample_rate": features['sample_rate']
            }
        }
        
        logger.info(f"Disease prediction completed: {prediction['predicted_disease']} (confidence: {prediction['confidence']:.3f})")
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during disease prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/predict_annotation")
async def predict_annotation_endpoint(file: UploadFile = File(...)):
    """
    Predict respiratory events and disease from uploaded audio file.
    
    Args:
        file: Audio file (.wav, .mp3, etc.)
        
    Returns:
        JSON response with annotation results
    """
    try:
        # Validate file type
        if not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Read file content
        audio_data = await file.read()
        
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        logger.info(f"Processing annotation prediction for: {file.filename}, size: {len(audio_data)} bytes")
        
        # Preprocess audio
        features = preprocess_audio(audio_data)
        
        # Predict annotations
        prediction = predict_annotation(features)
        
        # Prepare response
        response = {
            "success": True,
            "filename": file.filename,
            "disease": prediction['predicted_disease'],
            "confidence": prediction['confidence'],
            "events": prediction['events'],
            "audio_info": {
                "duration": features['duration'],
                "sample_rate": features['sample_rate']
            }
        }
        
        logger.info(f"Annotation prediction completed: {prediction['predicted_disease']} with {len(prediction['events'])} events")
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during annotation prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)