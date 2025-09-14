"""
Simple but effective Model 1 creation
Focus on core features that work reliably
"""

import pandas as pd
import numpy as np
import librosa
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_simple_features(audio_path: str) -> np.ndarray:
    """Extract simple but effective audio features."""
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=22050, duration=10.0)
        
        # Basic preprocessing
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
        audio_norm = librosa.util.normalize(audio_trimmed)
        
        # Pad or truncate to exactly 10 seconds
        target_length = int(10.0 * 22050)
        if len(audio_norm) > target_length:
            audio_norm = audio_norm[:target_length]
        elif len(audio_norm) < target_length:
            audio_norm = np.pad(audio_norm, (0, target_length - len(audio_norm)), mode='constant')
        
        features = []
        
        # 1. Basic time-domain features
        features.extend([
            np.mean(audio_norm),
            np.std(audio_norm),
            np.max(audio_norm),
            np.min(audio_norm),
            np.median(audio_norm),
            np.var(audio_norm),
            np.sqrt(np.mean(audio_norm**2)),  # RMS
        ])
        
        # 2. MFCC features (mean and std of each coefficient)
        mfccs = librosa.feature.mfcc(y=audio_norm, sr=sr, n_mfcc=13, hop_length=512)
        for i in range(13):
            features.extend([
                float(np.mean(mfccs[i])),
                float(np.std(mfccs[i]))
            ])
        
        # 3. Mel spectrogram features
        mel_spec = librosa.feature.melspectrogram(y=audio_norm, sr=sr, n_mels=128, hop_length=512)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        features.extend([
            float(np.mean(mel_spec_db)),
            float(np.std(mel_spec_db)),
            float(np.max(mel_spec_db)),
            float(np.min(mel_spec_db)),
        ])
        
        # 4. Spectral features
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio_norm)
        features.extend([
            float(np.mean(zcr)),
            float(np.std(zcr)),
        ])
        
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_norm, sr=sr)
        features.extend([
            float(np.mean(spectral_centroids)),
            float(np.std(spectral_centroids)),
        ])
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_norm, sr=sr)
        features.extend([
            float(np.mean(spectral_rolloff)),
            float(np.std(spectral_rolloff)),
        ])
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_norm, sr=sr)
        features.extend([
            float(np.mean(spectral_bandwidth)),
            float(np.std(spectral_bandwidth)),
        ])
        
        # 5. Tempo
        tempo, _ = librosa.beat.beat_track(y=audio_norm, sr=sr)
        features.append(float(tempo))
        
        # 6. Chroma features
        chroma = librosa.feature.chroma_stft(y=audio_norm, sr=sr)
        features.extend([
            float(np.mean(chroma)),
            float(np.std(chroma)),
        ])
        
        # 7. Tonnetz features
        tonnetz = librosa.feature.tonnetz(y=audio_norm, sr=sr)
        features.extend([
            float(np.mean(tonnetz)),
            float(np.std(tonnetz)),
        ])
        
        # 8. Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_norm, sr=sr)
        features.extend([
            float(np.mean(spectral_contrast)),
            float(np.std(spectral_contrast)),
        ])
        
        # 9. Energy features
        energy = librosa.feature.rms(y=audio_norm, frame_length=2048, hop_length=512)[0]
        features.extend([
            float(np.mean(energy)),
            float(np.std(energy)),
            float(np.max(energy)),
        ])
        
        return np.array(features, dtype=float)
        
    except Exception as e:
        logger.error(f"Error extracting features from {audio_path}: {str(e)}")
        return np.zeros(50, dtype=float)  # Return zero features if error

def create_simple_model1():
    """Create simple but effective Model 1."""
    logger.info("Creating simple but effective Model 1...")
    
    # Load data
    audio_dir = '../Respiratory_Sound_Database/audio_and_txt_files'
    csv_path = '../Respiratory_Sound_Database/patient_diagnosis.csv'
    
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        return
    
    # Load diagnosis data
    diagnosis_df = pd.read_csv(csv_path, header=None, names=['patient_id', 'disease'])
    logger.info(f"Loaded {len(diagnosis_df)} diagnosis records")
    
    # Extract features and labels
    features_list = []
    labels_list = []
    
    # Get all available audio files
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    logger.info(f"Found {len(audio_files)} audio files")
    
    # Create mapping from patient_id to available audio files
    patient_to_files = {}
    for audio_file in audio_files:
        patient_id = audio_file.split('_')[0]
        if patient_id not in patient_to_files:
            patient_to_files[patient_id] = []
        patient_to_files[patient_id].append(audio_file)
    
    logger.info(f"Found audio files for {len(patient_to_files)} patients")
    
    for idx, row in diagnosis_df.iterrows():
        patient_id = str(row['patient_id'])
        
        if patient_id in patient_to_files:
            # Use the first available audio file for this patient
            audio_file = patient_to_files[patient_id][0]
            audio_path = os.path.join(audio_dir, audio_file)
            
            features = extract_simple_features(audio_path)
            features_list.append(features)
            labels_list.append(row['disease'])
            logger.info(f"Processed {audio_file} -> {row['disease']}")
        else:
            logger.warning(f"No audio files found for patient {patient_id}")
    
    if not features_list:
        logger.error("No audio files found!")
        return
    
    X = np.array(features_list)
    y = np.array(labels_list)
    
    logger.info(f"Created feature matrix: {X.shape}")
    logger.info(f"Labels: {np.unique(y, return_counts=True)}")
    
    # Handle NaN values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Split data - use random split to avoid stratification issues with sparse classes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Try multiple models and pick the best one
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=500,
            max_depth=25,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=10,
            random_state=42
        )
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        mean_cv_score = np.mean(cv_scores)
        logger.info(f"{name} CV Score: {mean_cv_score:.4f} (+/- {np.std(cv_scores) * 2:.4f})")
        
        if mean_cv_score > best_score:
            best_score = mean_cv_score
            best_model = model
            best_name = name
    
    logger.info(f"Best model: {best_name} with CV score: {best_score:.4f}")
    
    # Train the best model
    best_model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = best_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Simple Model 1 Accuracy: {accuracy:.4f}")
    logger.info("Classification Report:")
    logger.info(classification_report(y_test, y_pred))
    
    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/model1_simple_improved.joblib')
    joblib.dump(scaler, 'models/scaler1_simple_improved.joblib')
    
    logger.info("✅ Simple Model 1 saved successfully")
    
    return best_model, scaler, accuracy

def main():
    """Create simple Model 1."""
    logger.info("Creating simple but effective Model 1...")
    
    try:
        model, scaler, acc = create_simple_model1()
        logger.info(f"✅ Simple Model 1 created with accuracy: {acc:.4f}")
    except Exception as e:
        logger.error(f"❌ Failed to create Simple Model 1: {str(e)}")
    
    logger.info("🎉 Simple model creation completed!")

if __name__ == "__main__":
    main()
