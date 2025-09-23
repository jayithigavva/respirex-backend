import os
import io
import logging
import replicate
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RespireX API",
    description="API for respiratory disease classification using Replicate",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Replicate configuration
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    logger.warning("⚠️ REPLICATE_API_TOKEN not found. Set it in Railway environment variables.")
    replicate_client = None
else:
    replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)
    logger.info("✅ Replicate client initialized successfully")

# Disease classes
DISEASE_CLASSES = [
    "Healthy", "COPD", "Pneumonia", "Asthma", "Bronchiectasis", 
    "Bronchiolitis", "LRTI", "URTI"
]

# Response models
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    model_used: str
    processing_time: float

class AnnotationResponse(BaseModel):
    wheeze: bool
    crackle: bool
    confidence: float
    model_used: str
    processing_time: float

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    logger.info("🚀 RespireX API starting up...")
    logger.info("📡 Using Replicate for ML inference")
    if replicate_client:
        logger.info("✅ Replicate client ready")
        else:
        logger.warning("⚠️ Replicate client not initialized - check API token")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RespireX API - Respiratory Disease Classification",
        "version": "1.0.0",
        "status": "running",
        "inference_provider": "Replicate"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        status = {
        "status": "healthy",
            "message": "RespireX API is running on Railway",
            "replicate_connected": replicate_client is not None,
            "timestamp": "2024-01-01T00:00:00Z"
        }
        logger.info("✅ Health check passed")
        return status
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Service unavailable")

@app.post("/predict", response_model=PredictionResponse)
async def predict_disease(file: UploadFile = File(...)):
    """
    Predict respiratory disease from audio file using Replicate
    """
    if not replicate_client:
        raise HTTPException(
            status_code=500, 
            detail="Replicate client not initialized. Check REPLICATE_API_TOKEN."
        )
    
    try:
        # Validate file type
        if not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Read file content
        content = await file.read()
        logger.info(f"📁 Processing audio file: {file.filename} ({len(content)} bytes)")
        
        # For now, return a mock response until we deploy models to Replicate
        # TODO: Replace with actual Replicate model call
        import time
        start_time = time.time()
        
        # Mock prediction (replace with actual Replicate call)
        prediction = "Healthy"
        confidence = 0.85
        
        processing_time = time.time() - start_time
        
        logger.info(f"🎯 Prediction: {prediction} (confidence: {confidence:.2f})")
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            model_used="respirex-disease-classifier",
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-annotation", response_model=AnnotationResponse)
async def predict_annotation(file: UploadFile = File(...)):
    """
    Predict respiratory annotations (wheeze, crackle) from audio file using Replicate
    """
    if not replicate_client:
        raise HTTPException(
            status_code=500, 
            detail="Replicate client not initialized. Check REPLICATE_API_TOKEN."
        )
    
    try:
        # Validate file type
        if not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Read file content
        content = await file.read()
        logger.info(f"📁 Processing audio file for annotation: {file.filename} ({len(content)} bytes)")
        
        # For now, return a mock response until we deploy models to Replicate
        # TODO: Replace with actual Replicate model call
        import time
        start_time = time.time()
        
        # Mock annotation (replace with actual Replicate call)
        wheeze = False
        crackle = False
        confidence = 0.78
        
        processing_time = time.time() - start_time
        
        logger.info(f"🎯 Annotation: wheeze={wheeze}, crackle={crackle} (confidence: {confidence:.2f})")
        
        return AnnotationResponse(
            wheeze=wheeze,
            crackle=crackle,
            confidence=confidence,
            model_used="respirex-annotation-classifier",
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"❌ Annotation error: {e}")
        raise HTTPException(status_code=500, detail=f"Annotation failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)