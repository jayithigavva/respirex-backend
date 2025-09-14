"""
Enhanced Model 1 creation with improved accuracy
Uses advanced audio features and better preprocessing
"""

import pandas as pd
import numpy as np
import librosa
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_enhanced_features(audio_path: str) -> np.ndarray:
    """Extract comprehensive audio features for better accuracy."""
    try:
        # Load audio with higher quality settings
        audio, sr = librosa.load(audio_path, sr=22050, duration=10.0)
        
        # Advanced preprocessing
        # Remove silence
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
        
        # Normalize
        audio_norm = librosa.util.normalize(audio_trimmed)
        
        # Pad or truncate to exactly 10 seconds
        target_length = int(10.0 * 22050)
        if len(audio_norm) > target_length:
            audio_norm = audio_norm[:target_length]
        elif len(audio_norm) < target_length:
            audio_norm = np.pad(audio_norm, (0, target_length - len(audio_norm)), mode='constant')
        
        features = []
        
        # 1. Time-domain features
        features.extend([
            np.mean(audio_norm),           # Mean amplitude
            np.std(audio_norm),            # Standard deviation
            np.max(audio_norm),            # Maximum amplitude
            np.min(audio_norm),            # Minimum amplitude
            np.median(audio_norm),         # Median amplitude
            np.var(audio_norm),            # Variance
            np.sqrt(np.mean(audio_norm**2)), # RMS energy
        ])
        
        # 2. Spectral features
        # MFCCs with more coefficients
        mfccs = librosa.feature.mfcc(y=audio_norm, sr=sr, n_mfcc=20, hop_length=512)
        features.extend([
            np.mean(mfccs, axis=1).tolist(),  # Mean of each MFCC coefficient
            np.std(mfccs, axis=1).tolist(),  # Std of each MFCC coefficient
        ])
        features = [item for sublist in features if isinstance(sublist, list) for item in sublist]
        
        # 3. Mel spectrogram features
        mel_spec = librosa.feature.melspectrogram(y=audio_norm, sr=sr, n_mels=128, hop_length=512)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        features.extend([
            np.mean(mel_spec_db),
            np.std(mel_spec_db),
            np.max(mel_spec_db),
            np.min(mel_spec_db),
            np.median(mel_spec_db),
        ])
        
        # 4. Spectral features
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio_norm)
        features.extend([
            np.mean(zcr),
            np.std(zcr),
            np.max(zcr),
        ])
        
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_norm, sr=sr)
        features.extend([
            np.mean(spectral_centroids),
            np.std(spectral_centroids),
            np.max(spectral_centroids),
        ])
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_norm, sr=sr)
        features.extend([
            np.mean(spectral_rolloff),
            np.std(spectral_rolloff),
            np.max(spectral_rolloff),
        ])
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_norm, sr=sr)
        features.extend([
            np.mean(spectral_bandwidth),
            np.std(spectral_bandwidth),
        ])
        
        # 5. Rhythm features
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=audio_norm, sr=sr)
        features.append(tempo)
        
        # 6. Chroma features
        chroma = librosa.feature.chroma_stft(y=audio_norm, sr=sr)
        features.extend([
            np.mean(chroma),
            np.std(chroma),
        ])
        
        # 7. Tonnetz features
        tonnetz = librosa.feature.tonnetz(y=audio_norm, sr=sr)
        features.extend([
            np.mean(tonnetz),
            np.std(tonnetz),
        ])
        
        # 8. Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_norm, sr=sr)
        features.extend([
            np.mean(spectral_contrast),
            np.std(spectral_contrast),
        ])
        
        # 9. Advanced statistical features
        # Kurtosis and skewness
        from scipy.stats import kurtosis, skew
        features.extend([
            kurtosis(audio_norm),
            skew(audio_norm),
        ])
        
        # 10. Energy features
        # Short-time energy
        frame_length = 2048
        hop_length = 512
        energy = librosa.feature.rms(y=audio_norm, frame_length=frame_length, hop_length=hop_length)[0]
        features.extend([
            np.mean(energy),
            np.std(energy),
            np.max(energy),
        ])
        
        return np.array(features)
        
    except Exception as e:
        logger.error(f"Error extracting features from {audio_path}: {str(e)}")
        return np.zeros(100)  # Return zero features if error

def create_enhanced_model1():
    """Create enhanced Model 1 with better accuracy."""
    logger.info("Creating enhanced Model 1 with improved accuracy...")
    
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
            
            features = extract_enhanced_features(audio_path)
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
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature selection to remove redundant features
    selector = SelectKBest(score_func=f_classif, k=min(50, X_train_scaled.shape[1]))
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    logger.info(f"Selected {X_train_selected.shape[1]} best features")
    
    # Try multiple models and pick the best one
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='accuracy')
        mean_cv_score = np.mean(cv_scores)
        logger.info(f"{name} CV Score: {mean_cv_score:.4f} (+/- {np.std(cv_scores) * 2:.4f})")
        
        if mean_cv_score > best_score:
            best_score = mean_cv_score
            best_model = model
            best_name = name
    
    logger.info(f"Best model: {best_name} with CV score: {best_score:.4f}")
    
    # Train the best model
    best_model.fit(X_train_selected, y_train)
    
    # Evaluate
    y_pred = best_model.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Enhanced Model 1 Accuracy: {accuracy:.4f}")
    logger.info("Classification Report:")
    logger.info(classification_report(y_test, y_pred))
    
    # Save model, scaler, and selector
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/model1_enhanced.joblib')
    joblib.dump(scaler, 'models/scaler1_enhanced.joblib')
    joblib.dump(selector, 'models/selector1_enhanced.joblib')
    
    logger.info("✅ Enhanced Model 1 saved successfully")
    
    return best_model, scaler, selector, accuracy

def main():
    """Create enhanced Model 1."""
    logger.info("Creating enhanced Model 1 with improved accuracy...")
    
    try:
        model, scaler, selector, acc = create_enhanced_model1()
        logger.info(f"✅ Enhanced Model 1 created with accuracy: {acc:.4f}")
    except Exception as e:
        logger.error(f"❌ Failed to create Enhanced Model 1: {str(e)}")
    
    logger.info("🎉 Enhanced model creation completed!")

if __name__ == "__main__":
    main()
