from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import random
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Disease classes
DISEASE_CLASSES = [
    "Healthy", "COPD", "Pneumonia", "Asthma", "Bronchiectasis", 
    "Bronchiolitis", "LRTI", "URTI"
]

# Pydantic models for request validation
class AnnotationEvent(BaseModel):
    type: str  # 'crackle' or 'wheeze'
    timestamp: float
    duration: float

class AnnotationRequest(BaseModel):
    events: list[AnnotationEvent]
    duration: float

@app.get("/")
async def root():
    return {"message": "Respirex API is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "Respirex API is running",
        "models": "Dummy models for demo",
        "optimization": "Render Free Tier Compatible"
    }

@app.post("/predict_disease")
async def predict_disease(file: UploadFile = File(...)):
    """Predict respiratory disease from audio file using dummy model"""
    try:
        # Read file
        content = await file.read()
        
        # Dummy prediction based on file size
        random.seed(len(content))
        predicted_class = random.choice(DISEASE_CLASSES)
        confidence = random.uniform(0.75, 0.95)
        
        # Create class probabilities dictionary
        class_probabilities = {}
        for disease in DISEASE_CLASSES:
            if disease == predicted_class:
                class_probabilities[disease] = confidence
            else:
                class_probabilities[disease] = (1 - confidence) / (len(DISEASE_CLASSES) - 1)
        
        return {
            "success": True,
            "filename": file.filename,
            "prediction": predicted_class,
            "confidence": float(confidence),
            "class_probabilities": class_probabilities,
            "audio_info": {
                "duration": 10.0,
                "sample_rate": 22050
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "filename": file.filename if file else "unknown"
        }

@app.post("/predict_annotation")
async def predict_annotation(annotation_data: AnnotationRequest):
    """Predict disease from doctor's button presses (crackles/wheezes)"""
    try:
        # Extract annotation data from request
        events = annotation_data.events
        duration = annotation_data.duration
        
        if not events:
            return {
                "success": False,
                "error": "No annotation events provided"
            }
        
        # Dummy prediction based on events
        random.seed(len(str(events)))
        predicted_class = random.choice(DISEASE_CLASSES)
        confidence = random.uniform(0.80, 0.95)
        
        return {
            "success": True,
            "disease": predicted_class,
            "confidence": float(confidence),
            "annotation_summary": {
                "total_events": len(events),
                "crackles": len([e for e in events if e.type == 'crackle']),
                "wheezes": len([e for e in events if e.type == 'wheeze']),
                "duration": duration
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)