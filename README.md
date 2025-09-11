# Respiratory Disease Classification API

A FastAPI backend service for classifying respiratory diseases from audio files using deep learning.

## Features

- **Audio Classification**: Predict respiratory diseases from audio files (WAV, MP3, M4A, FLAC)
- **Anomaly Detection**: Identify wheezes and crackles with timestamps
- **RESTful API**: Clean FastAPI endpoints with automatic documentation
- **Cloud Ready**: Configured for deployment on Render

## API Endpoints

### Health Check
```
GET /health
```
Returns the health status of the API and model loading status.

### Prediction
```
POST /predict
```
Upload an audio file and get disease prediction with anomaly detection.

**Request**: Multipart form data with audio file
**Response**:
```json
{
  "success": true,
  "filename": "audio.wav",
  "prediction": {
    "disease": "COPD",
    "confidence": 0.85,
    "class_probabilities": {
      "COPD": 0.85,
      "Healthy": 0.10,
      "Pneumonia": 0.05
    }
  },
  "anomalies": [
    {
      "type": "wheeze",
      "start_time": 2.5,
      "end_time": 4.2,
      "confidence": 0.78
    }
  ],
  "audio_info": {
    "duration": 10.0,
    "sample_rate": 22050
  }
}
```

## Setup and Installation

### Prerequisites
- Python 3.8+
- pip

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-backend-repo-url>
   cd respiratory-disease-backend
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your trained model**
   - Place your trained model file as `model.pth` in the backend directory
   - Update `DISEASE_CLASSES` in `app.py` to match your model's classes

4. **Run the development server**
   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 10000
   ```

5. **Access the API**
   - API: http://localhost:10000
   - Interactive docs: http://localhost:10000/docs

## Deployment on Render

### Step 1: Prepare Repository

1. **Create a new GitHub repository** for your backend
2. **Push your code** to the repository
3. **Ensure your model file** (`model.pth`) is included in the repository

### Step 2: Deploy on Render

1. **Go to [Render](https://render.com)** and sign up/login
2. **Click "New +"** and select **"Web Service"**
3. **Connect your GitHub repository**
4. **Configure the service**:
   - **Name**: `respiratory-disease-api`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port 10000`
   - **Health Check Path**: `/health`

5. **Deploy**
   - Click **"Create Web Service"**
   - Wait for deployment to complete (5-10 minutes)
   - Note your service URL (e.g., `https://respiratory-disease-api.onrender.com`)

### Step 3: Update Frontend

After deployment, update your frontend's API URL:

1. **In your frontend repository**, update the API URL:
   ```typescript
   // In your frontend code
   const API_BASE_URL = 'https://your-service-name.onrender.com'
   ```

2. **Redeploy your frontend** on Vercel

## Model Requirements

Your trained model should:
- Be saved as `model.pth` using PyTorch's `torch.save()`
- Have the same architecture as defined in `RespiratoryCNN` class
- Output logits for the number of disease classes
- Be compatible with the preprocessing pipeline

## Environment Variables

You can set these environment variables in Render:

- `PYTHON_VERSION`: Python version (default: 3.11.0)
- `MODEL_PATH`: Path to model file (default: model.pth)

## Troubleshooting

### Common Issues

1. **Model not loading**
   - Ensure `model.pth` is in the repository
   - Check model architecture matches `RespiratoryCNN`
   - Verify model was saved with `torch.save(model.state_dict(), 'model.pth')`

2. **Audio processing errors**
   - Check audio file format (supports WAV, MP3, M4A, FLAC)
   - Ensure audio file is not corrupted
   - Verify file size is reasonable (< 50MB)

3. **Deployment issues**
   - Check build logs in Render dashboard
   - Ensure all dependencies are in `requirements.txt`
   - Verify Python version compatibility

### Performance Optimization

- **Model Loading**: The model loads on startup, which may take 30-60 seconds
- **Memory Usage**: Ensure your Render plan has sufficient memory
- **Timeout**: Large audio files may timeout (30s limit)

## API Documentation

Once deployed, visit `https://your-service-url.onrender.com/docs` for interactive API documentation.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review Render deployment logs
3. Test locally first before deploying