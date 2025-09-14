"""
FastAPI backend for Respiratory Disease Classification
Lightweight version optimized for Render free tier
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import librosa
import numpy as np
import pandas as pd
import io
import json
import logging
import os
from typing import Dict, List, Any
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Respiratory Disease Classification API",
    description="Lightweight API for classifying respiratory diseases - optimized for free tier",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
scaler1 = None
scaler2 = None

def load_lightweight_models():
    """
    Load lightweight models optimized for free tier.
    """
    global model1, model2, scaler1, scaler2
    
    try:
        # Try to load lightweight models (if they exist)
        model1_path = 'models/model1_lightweight.joblib'
        model2_path = 'models/model2_lightweight.joblib'
        scaler1_path = 'models/scaler1_lightweight.joblib'
        scaler2_path = 'models/scaler2_lightweight.joblib'
        
        if os.path.exists(model1_path) and os.path.exists(scaler1_path):
            model1 = joblib.load(model1_path)
            scaler1 = joblib.load(scaler1_path)
            logger.info("✅ Model 1 (Lightweight Disease Classifier) loaded successfully")
        else:
            logger.warning("❌ Model 1 not found, will use dummy predictions")
        
        if os.path.exists(model2_path) and os.path.exists(scaler2_path):
            model2 = joblib.load(model2_path)
            scaler2 = joblib.load(scaler2_path)
            logger.info("✅ Model 2 (Lightweight Annotation Model) loaded successfully")
        else:
            logger.warning("❌ Model 2 not found, will use dummy predictions")
        
        return model1 is not None, model2 is not None
        
    except Exception as e:
        logger.error(f"Error loading lightweight models: {str(e)}")
        return False, False

def extract_lightweight_features(audio_data: bytes) -> np.ndarray:
    """
    Extract lightweight features from audio for free tier compatibility.
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
        
        # Extract lightweight features
        features = []
        
        # Basic audio features
        features.extend([
            np.mean(audio),
            np.std(audio),
            np.max(audio),
            np.min(audio),
            np.median(audio)
        ])
        
        # Spectral features
        mfccs = librosa.feature.mfcc(y=audio, sr=TARGET_SR, n_mfcc=13)
        features.extend([
            np.mean(mfccs),
            np.std(mfccs),
            np.max(mfccs),
            np.min(mfccs)
        ])
        
        # Mel spectrogram features
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=TARGET_SR, n_mels=N_MELS)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        features.extend([
            np.mean(mel_spec_db),
            np.std(mel_spec_db),
            np.max(mel_spec_db),
            np.min(mel_spec_db)
        ])
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        features.extend([
            np.mean(zcr),
            np.std(zcr)
        ])
        
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=TARGET_SR)
        features.extend([
            np.mean(spectral_centroids),
            np.std(spectral_centroids)
        ])
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=TARGET_SR)
        features.extend([
            np.mean(spectral_rolloff),
            np.std(spectral_rolloff)
        ])
        
        return np.array(features).reshape(1, -1)
        
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Feature extraction failed: {str(e)}")

def create_annotation_features_from_csv(annotation_data: str) -> np.ndarray:
    """
    Create lightweight annotation features from CSV data.
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
        
        # Create lightweight features
        features = []
        
        if annotations:
            # Basic statistics
            durations = [ann['end'] - ann['start'] for ann in annotations]
            crackle_counts = [ann['crackles'] for ann in annotations]
            wheeze_counts = [ann['wheezes'] for ann in annotations]
            
            features.extend([
                len(annotations),  # Total number of annotations
                np.mean(durations) if durations else 0,  # Average duration
                np.std(durations) if durations else 0,   # Duration std
                np.sum(crackle_counts),  # Total crackles
                np.sum(wheeze_counts),   # Total wheezes
                np.mean(crackle_counts) if crackle_counts else 0,  # Avg crackles per annotation
                np.mean(wheeze_counts) if wheeze_counts else 0,    # Avg wheezes per annotation
            ])
            
            # Time distribution features
            start_times = [ann['start'] for ann in annotations]
            end_times = [ann['end'] for ann in annotations]
            
            features.extend([
                np.mean(start_times) if start_times else 0,
                np.std(start_times) if start_times else 0,
                np.mean(end_times) if end_times else 0,
                np.std(end_times) if end_times else 0,
            ])
        else:
            # No annotations - fill with zeros
            features = [0] * 11
        
        return np.array(features).reshape(1, -1)
        
    except Exception as e:
        logger.error(f"Error creating annotation features: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Annotation processing failed: {str(e)}")

# Load models on startup
models_loaded = load_lightweight_models()

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Respiratory Disease Classification API (Lightweight)",
        "version": "2.0.0",
        "status": "running",
        "models_loaded": {
            "model1_disease_classifier": models_loaded[0],
            "model2_doctor_assisted": models_loaded[1]
        },
        "optimized_for": "Render Free Tier"
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
        "optimized_for": "Render Free Tier",
        "memory_efficient": True
    }

@app.post("/predict_disease")
async def predict_disease_endpoint(file: UploadFile = File(...)):
    """
    Predict respiratory disease from uploaded audio file using lightweight Model 1.
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
        
        # Extract features
        features = extract_lightweight_features(audio_data)
        
        if models_loaded[0] and scaler1 is not None:
            # Use trained model
            features_scaled = scaler1.transform(features)
            probabilities = model1.predict_proba(features_scaled)[0]
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]
        else:
            # Use dummy prediction
            probabilities = np.random.dirichlet(np.ones(len(DISEASE_CLASSES)))
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]
            logger.warning("Using dummy prediction - model not loaded")
        
        # Create class probabilities dictionary
        class_probabilities = {
            DISEASE_CLASSES[i]: float(probabilities[i])
            for i in range(len(DISEASE_CLASSES))
        }
        
        # Prepare response
        response = {
            "success": True,
            "filename": file.filename,
            "prediction": DISEASE_CLASSES[predicted_class],
            "confidence": float(confidence),
            "class_probabilities": class_probabilities,
            "model_type": "lightweight",
            "audio_info": {
                "duration": DURATION,
                "sample_rate": TARGET_SR
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
    Predict disease from doctor's annotation data using lightweight Model 2.
    """
    try:
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
        
        if models_loaded[1] and scaler2 is not None:
            # Use trained model
            features_scaled = scaler2.transform(annotation_features)
            probabilities = model2.predict_proba(features_scaled)[0]
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]
        else:
            # Use dummy prediction
            probabilities = np.random.dirichlet(np.ones(len(DISEASE_CLASSES)))
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]
            logger.warning("Using dummy prediction - model not loaded")
        
        # Create class probabilities dictionary
        class_probabilities = {
            DISEASE_CLASSES[i]: float(probabilities[i])
            for i in range(len(DISEASE_CLASSES))
        }
        
        # Prepare response
        response = {
            "success": True,
            "filename": file.filename,
            "disease": DISEASE_CLASSES[predicted_class],
            "confidence": float(confidence),
            "class_probabilities": class_probabilities,
            "model_type": "lightweight",
            "annotation_info": {
                "num_annotations": len(annotation_data.decode('utf-8').split('\n')) - 1,
                "model_type": "Lightweight Doctor-Assisted Annotation Model"
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
