"""
FastAPI backend for Respiratory Disease Classification
Simplified version for deployment testing
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import librosa
import numpy as np
import io
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

def preprocess_audio(audio_data: bytes) -> dict:
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

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Respiratory Disease Classification API",
        "version": "1.0.0",
        "status": "running",
        "models_loaded": False
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": False,
        "device": "cpu"
    }

@app.post("/predict_disease")
async def predict_disease_endpoint(file: UploadFile = File(...)):
    """
    Predict respiratory disease from uploaded audio file.
    Currently returns dummy predictions for testing.
    
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
        
        # Dummy prediction for testing
        prediction = {
            'predicted_disease': 'COPD',
            'confidence': 0.85,
            'class_probabilities': {
                'COPD': 0.85,
                'Healthy': 0.10,
                'Pneumonia': 0.03,
                'URTI': 0.02
            }
        }
        
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
    Currently returns dummy predictions for testing.
    
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
        
        # Dummy prediction for testing
        prediction = {
            'predicted_disease': 'COPD',
            'confidence': 0.92,
            'events': [
                {
                    'start': 1.5,
                    'end': 3.2,
                    'label': 'wheeze',
                    'confidence': 0.78
                },
                {
                    'start': 5.8,
                    'end': 7.1,
                    'label': 'crackle',
                    'confidence': 0.65
                }
            ]
        }
        
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
