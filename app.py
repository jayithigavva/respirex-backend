"""
FastAPI backend for Respiratory Disease Classification
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
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
    description="API for classifying respiratory diseases from audio files",
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

# Disease classes (update based on your trained model)
DISEASE_CLASSES = [
    'Asthma', 'Bronchiectasis', 'Bronchiolitis', 'COPD', 
    'Healthy', 'LRTI', 'Pneumonia', 'URTI'
]

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

# Global model variable
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    """Load the trained model."""
    global model, device
    try:
        # Initialize device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Check if model file exists
        import os
        if not os.path.exists('model.pth'):
            logger.error("Model file 'model.pth' not found")
            return False
            
        model = RespiratoryCNN(
            input_height=128,
            input_width=431,
            num_classes=len(DISEASE_CLASSES),
            dropout_rate=0.3
        )
        
        # Load the trained weights
        model.load_state_dict(torch.load('model.pth', map_location=device))
        model.to(device)
        model.eval()
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
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
        
        # Spectral features for anomaly detection
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=TARGET_SR, hop_length=HOP_LENGTH)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=TARGET_SR, hop_length=HOP_LENGTH)[0]
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LENGTH)[0]
        
        return {
            'audio': audio,
            'mfccs': mfccs,
            'mel_spectrogram': mel_spec_db,
            'spectral_centroid': spectral_centroid,
            'spectral_rolloff': spectral_rolloff,
            'zcr': zcr,
            'duration': DURATION,
            'sample_rate': TARGET_SR
        }
        
    except Exception as e:
        logger.error(f"Error preprocessing audio: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Audio preprocessing failed: {str(e)}")

def detect_anomalies(features: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Detect anomalous segments in the audio (wheezes, crackles).
    This is a simplified version - you can enhance this with your trained anomaly detection model.
    
    Args:
        features: Preprocessed audio features
        
    Returns:
        List of anomaly segments with timestamps
    """
    anomalies = []
    
    # Simple threshold-based detection for demonstration
    # In practice, you would use your trained anomaly detection model
    
    # Detect potential wheezes (high spectral centroid)
    spectral_centroid = features['spectral_centroid']
    wheeze_threshold = np.mean(spectral_centroid) + 2 * np.std(spectral_centroid)
    
    # Detect potential crackles (high zero-crossing rate)
    zcr = features['zcr']
    crackle_threshold = np.mean(zcr) + 2 * np.std(zcr)
    
    frame_rate = TARGET_SR / HOP_LENGTH
    
    # Find wheeze segments
    wheeze_frames = np.where(spectral_centroid > wheeze_threshold)[0]
    if len(wheeze_frames) > 0:
        # Group consecutive frames
        groups = []
        current_group = [wheeze_frames[0]]
        
        for i in range(1, len(wheeze_frames)):
            if wheeze_frames[i] - wheeze_frames[i-1] <= 5:  # Within 5 frames
                current_group.append(wheeze_frames[i])
            else:
                groups.append(current_group)
                current_group = [wheeze_frames[i]]
        groups.append(current_group)
        
        for group in groups:
            if len(group) >= 10:  # Minimum duration threshold
                start_time = group[0] / frame_rate
                end_time = group[-1] / frame_rate
                anomalies.append({
                    'type': 'wheeze',
                    'start_time': start_time,
                    'end_time': end_time,
                    'confidence': min(0.9, len(group) / 50)  # Simple confidence based on duration
                })
    
    # Find crackle segments
    crackle_frames = np.where(zcr > crackle_threshold)[0]
    if len(crackle_frames) > 0:
        # Group consecutive frames
        groups = []
        current_group = [crackle_frames[0]]
        
        for i in range(1, len(crackle_frames)):
            if crackle_frames[i] - crackle_frames[i-1] <= 3:  # Within 3 frames
                current_group.append(crackle_frames[i])
            else:
                groups.append(current_group)
                current_group = [crackle_frames[i]]
        groups.append(current_group)
        
        for group in groups:
            if len(group) >= 5:  # Minimum duration threshold
                start_time = group[0] / frame_rate
                end_time = group[-1] / frame_rate
                anomalies.append({
                    'type': 'crackle',
                    'start_time': start_time,
                    'end_time': end_time,
                    'confidence': min(0.9, len(group) / 20)  # Simple confidence based on duration
                })
    
    return anomalies

def predict_disease_internal(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict disease from audio features.
    
    Args:
        features: Preprocessed audio features
        
    Returns:
        Dictionary containing prediction results
    """
    global model
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Prepare input tensor
        mel_spec = features['mel_spectrogram']
        mel_spec_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        mel_spec_tensor = mel_spec_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(mel_spec_tensor)
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
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("Starting up Respiratory Disease Classification API...")
    try:
        if load_model():
            logger.info("API ready for predictions")
        else:
            logger.warning("Model loading failed, but API is still available")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        logger.info("API starting without model (will load on first request)")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Respiratory Disease Classification API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if 'device' in globals() else "unknown"
    }

@app.post("/predict_disease")
async def predict_disease(file: UploadFile = File(...)):
    """
    Predict respiratory disease from uploaded audio file.
    
    Args:
        file: Audio file (.wav, .mp3, etc.)
        
    Returns:
        JSON response with prediction results
    """
    try:
        # Validate file type
        if not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Read file content
        audio_data = await file.read()
        
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        logger.info(f"Processing audio file: {file.filename}, size: {len(audio_data)} bytes")
        
        # Preprocess audio
        features = preprocess_audio(audio_data)
        
        # Try to load model if not already loaded
        if model is None:
            logger.info("Model not loaded, attempting to load...")
            if not load_model():
                raise HTTPException(status_code=500, detail="Failed to load model")
        
        # Predict disease
        prediction = predict_disease_internal(features)
        
        # Detect anomalies
        anomalies = detect_anomalies(features)
        
        # Prepare response
        response = {
            "success": True,
            "filename": file.filename,
            "prediction": {
                "disease": prediction['predicted_disease'],
                "confidence": prediction['confidence'],
                "class_probabilities": prediction['class_probabilities']
            },
            "anomalies": anomalies,
            "audio_info": {
                "duration": features['duration'],
                "sample_rate": features['sample_rate']
            }
        }
        
        logger.info(f"Prediction completed: {prediction['predicted_disease']} (confidence: {prediction['confidence']:.3f})")
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/predict_annotation")
async def predict_annotation(data: dict):
    """
    Predict disease from annotation events.
    
    Args:
        data: Dictionary containing events and duration
        
    Returns:
        JSON response with prediction results
    """
    try:
        events = data.get('events', [])
        duration = data.get('duration', 0)
        
        if not events:
            raise HTTPException(status_code=400, detail="No events provided")
        
        # Simple annotation-based prediction logic
        # Count different event types
        crackle_count = sum(1 for event in events if event['type'] == 'crackle')
        wheeze_count = sum(1 for event in events if event['type'] == 'wheeze')
        
        # Simple rule-based prediction
        if wheeze_count > crackle_count and wheeze_count > 2:
            predicted_disease = "Asthma"
            confidence = min(0.9, 0.6 + (wheeze_count * 0.1))
        elif crackle_count > wheeze_count and crackle_count > 2:
            predicted_disease = "Pneumonia"
            confidence = min(0.9, 0.6 + (crackle_count * 0.1))
        elif crackle_count > 0 and wheeze_count > 0:
            predicted_disease = "COPD"
            confidence = min(0.9, 0.7 + ((crackle_count + wheeze_count) * 0.05))
        else:
            predicted_disease = "Healthy"
            confidence = 0.8
        
        # Format events for response
        formatted_events = []
        for event in events:
            formatted_events.append({
                'start': event['timestamp'],
                'end': event['timestamp'] + event['duration'],
                'label': event['type'],
                'confidence': 0.8
            })
        
        response = {
            "success": True,
            "filename": "annotation_data",
            "disease": predicted_disease,
            "confidence": confidence,
            "events": formatted_events,
            "audio_info": {
                "duration": duration,
                "sample_rate": 22050
            }
        }
        
        logger.info(f"Annotation prediction completed: {predicted_disease} (confidence: {confidence:.3f})")
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during annotation prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
