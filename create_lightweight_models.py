"""
Create lightweight models for Render free tier deployment
Uses scikit-learn instead of PyTorch for smaller memory footprint
"""

import pandas as pd
import numpy as np
import librosa
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_lightweight_features(audio_path: str) -> np.ndarray:
    """Extract lightweight features from audio file."""
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=22050)
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
        
        # Pad or truncate to 10 seconds
        target_length = int(10.0 * 22050)
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
        mfccs = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=13)
        features.extend([
            np.mean(mfccs),
            np.std(mfccs),
            np.max(mfccs),
            np.min(mfccs)
        ])
        
        # Mel spectrogram features
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=22050, n_mels=128)
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
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=22050)
        features.extend([
            np.mean(spectral_centroids),
            np.std(spectral_centroids)
        ])
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=22050)
        features.extend([
            np.mean(spectral_rolloff),
            np.std(spectral_rolloff)
        ])
        
        return np.array(features)
        
    except Exception as e:
        logger.error(f"Error extracting features from {audio_path}: {str(e)}")
        return np.zeros(25)  # Return zero features if error

def create_model1_lightweight():
    """Create lightweight Model 1 for disease classification."""
    logger.info("Creating lightweight Model 1...")
    
    # Load data
    audio_dir = '../Respiratory_Sound_Database/audio_and_txt_files'
    csv_path = '../Respiratory_Sound_Database/patient_diagnosis.csv'
    
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        return
    
    # Load diagnosis data - it's just patient_id,disease format
    diagnosis_df = pd.read_csv(csv_path, header=None, names=['patient_id', 'disease'])
    logger.info(f"Loaded {len(diagnosis_df)} diagnosis records")
    
    # Extract features and labels
    features_list = []
    labels_list = []
    
    for idx, row in diagnosis_df.iterrows():
        audio_file = f"{row['patient_id']}.wav"
        audio_path = os.path.join(audio_dir, audio_file)
        
        if os.path.exists(audio_path):
            features = extract_lightweight_features(audio_path)
            features_list.append(features)
            labels_list.append(row['disease'])
        else:
            logger.warning(f"Audio file not found: {audio_path}")
    
    if not features_list:
        logger.error("No audio files found!")
        return
    
    X = np.array(features_list)
    y = np.array(labels_list)
    
    logger.info(f"Created feature matrix: {X.shape}")
    logger.info(f"Labels: {np.unique(y, return_counts=True)}")
    
    # Split data - use random split to avoid stratification issues
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Model 1 Accuracy: {accuracy:.4f}")
    logger.info("Classification Report:")
    logger.info(classification_report(y_test, y_pred))
    
    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model1_lightweight.joblib')
    joblib.dump(scaler, 'models/scaler1_lightweight.joblib')
    
    logger.info("✅ Model 1 saved successfully")
    
    return model, scaler, accuracy

def create_model2_lightweight():
    """Create lightweight Model 2 for annotation-based prediction."""
    logger.info("Creating lightweight Model 2...")
    
    # Load annotation data
    csv_path = '../data/raw/data.csv'
    
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} annotation records")
    
    # Group by filename to get per-file features
    file_features = []
    file_labels = []
    
    for filename in df['filename'].unique():
        file_data = df[df['filename'] == filename]
        
        # Create lightweight features
        features = []
        
        # Basic statistics
        durations = file_data['end'] - file_data['start']
        crackle_counts = file_data['crackles']
        wheeze_counts = file_data['weezels']
        
        features.extend([
            len(file_data),  # Total number of annotations
            np.mean(durations) if len(durations) > 0 else 0,  # Average duration
            np.std(durations) if len(durations) > 0 else 0,   # Duration std
            np.sum(crackle_counts),  # Total crackles
            np.sum(wheeze_counts),   # Total wheezes
            np.mean(crackle_counts) if len(crackle_counts) > 0 else 0,  # Avg crackles per annotation
            np.mean(wheeze_counts) if len(wheeze_counts) > 0 else 0,    # Avg wheezes per annotation
        ])
        
        # Time distribution features
        start_times = file_data['start']
        end_times = file_data['end']
        
        features.extend([
            np.mean(start_times) if len(start_times) > 0 else 0,
            np.std(start_times) if len(start_times) > 0 else 0,
            np.mean(end_times) if len(end_times) > 0 else 0,
            np.std(end_times) if len(end_times) > 0 else 0,
        ])
        
        # Get disease label (should be same for all rows of same file)
        disease = file_data['disease'].iloc[0]
        
        file_features.append(features)
        file_labels.append(disease)
    
    X = np.array(file_features)
    y = np.array(file_labels)
    
    logger.info(f"Created feature matrix: {X.shape}")
    logger.info(f"Labels: {np.unique(y, return_counts=True)}")
    
    # Split data - use random split to avoid stratification issues with sparse classes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Model 2 Accuracy: {accuracy:.4f}")
    logger.info("Classification Report:")
    logger.info(classification_report(y_test, y_pred))
    
    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model2_lightweight.joblib')
    joblib.dump(scaler, 'models/scaler2_lightweight.joblib')
    
    logger.info("✅ Model 2 saved successfully")
    
    return model, scaler, accuracy

def main():
    """Create both lightweight models."""
    logger.info("Creating lightweight models for Render free tier...")
    
    # Create Model 1
    try:
        model1, scaler1, acc1 = create_model1_lightweight()
        logger.info(f"✅ Model 1 created with accuracy: {acc1:.4f}")
    except Exception as e:
        logger.error(f"❌ Failed to create Model 1: {str(e)}")
    
    # Create Model 2
    try:
        model2, scaler2, acc2 = create_model2_lightweight()
        logger.info(f"✅ Model 2 created with accuracy: {acc2:.4f}")
    except Exception as e:
        logger.error(f"❌ Failed to create Model 2: {str(e)}")
    
    logger.info("🎉 Lightweight model creation completed!")

if __name__ == "__main__":
    main()
