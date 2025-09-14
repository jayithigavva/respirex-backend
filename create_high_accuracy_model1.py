"""
High-Accuracy Model 1 Creation
Uses advanced ML techniques for medical-grade accuracy
"""

import pandas as pd
import numpy as np
import librosa
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
import joblib
import logging
from scipy.stats import kurtosis, skew
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_advanced_features(audio_path: str) -> np.ndarray:
    """Extract comprehensive advanced audio features."""
    try:
        # Load audio with high quality
        audio, sr = librosa.load(audio_path, sr=22050, duration=10.0)
        
        # Advanced preprocessing
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
        audio_norm = librosa.util.normalize(audio_trimmed)
        
        # Pad or truncate to exactly 10 seconds
        target_length = int(10.0 * 22050)
        if len(audio_norm) > target_length:
            audio_norm = audio_norm[:target_length]
        elif len(audio_norm) < target_length:
            audio_norm = np.pad(audio_norm, (0, target_length - len(audio_norm)), mode='constant')
        
        features = []
        
        # 1. Advanced time-domain features
        features.extend([
            np.mean(audio_norm),
            np.std(audio_norm),
            np.max(audio_norm),
            np.min(audio_norm),
            np.median(audio_norm),
            np.var(audio_norm),
            np.sqrt(np.mean(audio_norm**2)),  # RMS
            kurtosis(audio_norm),
            skew(audio_norm),
            np.percentile(audio_norm, 25),
            np.percentile(audio_norm, 75),
            np.percentile(audio_norm, 90),
            np.percentile(audio_norm, 95),
        ])
        
        # 2. Advanced MFCC features
        mfccs = librosa.feature.mfcc(y=audio_norm, sr=sr, n_mfcc=20, hop_length=512)
        for i in range(20):
            features.extend([
                float(np.mean(mfccs[i])),
                float(np.std(mfccs[i])),
                float(np.max(mfccs[i])),
                float(np.min(mfccs[i])),
                float(np.median(mfccs[i])),
            ])
        
        # 3. Delta and delta-delta MFCCs
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        for i in range(20):
            features.extend([
                float(np.mean(delta_mfccs[i])),
                float(np.std(delta_mfccs[i])),
                float(np.mean(delta2_mfccs[i])),
                float(np.std(delta2_mfccs[i])),
            ])
        
        # 4. Mel spectrogram features
        mel_spec = librosa.feature.melspectrogram(y=audio_norm, sr=sr, n_mels=128, hop_length=512)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        features.extend([
            float(np.mean(mel_spec_db)),
            float(np.std(mel_spec_db)),
            float(np.max(mel_spec_db)),
            float(np.min(mel_spec_db)),
            float(np.median(mel_spec_db)),
            float(np.percentile(mel_spec_db, 25)),
            float(np.percentile(mel_spec_db, 75)),
        ])
        
        # 5. Spectral features
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio_norm)
        features.extend([
            float(np.mean(zcr)),
            float(np.std(zcr)),
            float(np.max(zcr)),
            float(np.min(zcr)),
        ])
        
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_norm, sr=sr)
        features.extend([
            float(np.mean(spectral_centroids)),
            float(np.std(spectral_centroids)),
            float(np.max(spectral_centroids)),
            float(np.min(spectral_centroids)),
        ])
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_norm, sr=sr)
        features.extend([
            float(np.mean(spectral_rolloff)),
            float(np.std(spectral_rolloff)),
            float(np.max(spectral_rolloff)),
            float(np.min(spectral_rolloff)),
        ])
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_norm, sr=sr)
        features.extend([
            float(np.mean(spectral_bandwidth)),
            float(np.std(spectral_bandwidth)),
            float(np.max(spectral_bandwidth)),
            float(np.min(spectral_bandwidth)),
        ])
        
        # Spectral flatness
        spectral_flatness = librosa.feature.spectral_flatness(y=audio_norm)
        features.extend([
            float(np.mean(spectral_flatness)),
            float(np.std(spectral_flatness)),
        ])
        
        # 6. Tempo and rhythm features
        tempo, _ = librosa.beat.beat_track(y=audio_norm, sr=sr)
        features.append(float(tempo))
        
        # 7. Chroma features
        chroma = librosa.feature.chroma_stft(y=audio_norm, sr=sr)
        features.extend([
            float(np.mean(chroma)),
            float(np.std(chroma)),
            float(np.max(chroma)),
            float(np.min(chroma)),
        ])
        
        # 8. Tonnetz features
        tonnetz = librosa.feature.tonnetz(y=audio_norm, sr=sr)
        features.extend([
            float(np.mean(tonnetz)),
            float(np.std(tonnetz)),
            float(np.max(tonnetz)),
            float(np.min(tonnetz)),
        ])
        
        # 9. Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_norm, sr=sr)
        features.extend([
            float(np.mean(spectral_contrast)),
            float(np.std(spectral_contrast)),
            float(np.max(spectral_contrast)),
            float(np.min(spectral_contrast)),
        ])
        
        # 10. Energy features
        energy = librosa.feature.rms(y=audio_norm, frame_length=2048, hop_length=512)[0]
        features.extend([
            float(np.mean(energy)),
            float(np.std(energy)),
            float(np.max(energy)),
            float(np.min(energy)),
            float(np.median(energy)),
        ])
        
        # 11. Advanced statistical features
        # Peak detection
        peaks, _ = find_peaks(audio_norm, height=np.std(audio_norm))
        features.extend([
            len(peaks),
            float(np.mean(np.diff(peaks))) if len(peaks) > 1 else 0,
            float(np.std(np.diff(peaks))) if len(peaks) > 1 else 0,
        ])
        
        # 12. Frequency domain features
        fft = np.fft.fft(audio_norm)
        fft_magnitude = np.abs(fft)
        features.extend([
            float(np.mean(fft_magnitude)),
            float(np.std(fft_magnitude)),
            float(np.max(fft_magnitude)),
            float(np.argmax(fft_magnitude)),  # Dominant frequency
        ])
        
        return np.array(features, dtype=float)
        
    except Exception as e:
        logger.error(f"Error extracting features from {audio_path}: {str(e)}")
        return np.zeros(200, dtype=float)  # Return zero features if error

def create_high_accuracy_model1():
    """Create high-accuracy Model 1 using advanced techniques."""
    logger.info("Creating high-accuracy Model 1 with advanced ML techniques...")
    
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
            
            features = extract_advanced_features(audio_path)
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
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features using RobustScaler (better for outliers)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature selection using multiple methods
    # Method 1: SelectKBest
    selector_kbest = SelectKBest(score_func=f_classif, k=min(100, X_train_scaled.shape[1]))
    X_train_kbest = selector_kbest.fit_transform(X_train_scaled, y_train)
    X_test_kbest = selector_kbest.transform(X_test_scaled)
    
    # Method 2: PCA for dimensionality reduction
    pca = PCA(n_components=min(50, X_train_kbest.shape[1]), random_state=42)
    X_train_pca = pca.fit_transform(X_train_kbest)
    X_test_pca = pca.transform(X_test_kbest)
    
    logger.info(f"Selected {X_train_kbest.shape[1]} features with SelectKBest")
    logger.info(f"Reduced to {X_train_pca.shape[1]} components with PCA")
    logger.info(f"PCA explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # Create ensemble of advanced models
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=1000,
            max_depth=30,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=1000,
            max_depth=30,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=15,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        ),
        'SVM': SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            random_state=42,
            probability=True
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(200, 100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=1000,
            random_state=42
        )
    }
    
    # Train individual models and evaluate
    model_scores = {}
    trained_models = {}
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_pca, y_train, cv=5, scoring='accuracy')
        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)
        
        logger.info(f"{name} CV Score: {mean_cv_score:.4f} (+/- {std_cv_score * 2:.4f})")
        
        # Train on full training set
        model.fit(X_train_pca, y_train)
        trained_models[name] = model
        model_scores[name] = mean_cv_score
    
    # Create voting ensemble of best models
    best_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    logger.info(f"Best models: {[name for name, score in best_models]}")
    
    voting_models = [(name, trained_models[name]) for name, _ in best_models]
    ensemble = VotingClassifier(estimators=voting_models, voting='soft')
    
    logger.info("Training ensemble model...")
    ensemble.fit(X_train_pca, y_train)
    
    # Evaluate ensemble
    y_pred = ensemble.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"High-Accuracy Model 1 Accuracy: {accuracy:.4f}")
    logger.info("Classification Report:")
    logger.info(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info("Confusion Matrix:")
    logger.info(cm)
    
    # Save the ensemble model and preprocessing components
    os.makedirs('models', exist_ok=True)
    joblib.dump(ensemble, 'models/model1_high_accuracy.joblib')
    joblib.dump(scaler, 'models/scaler1_high_accuracy.joblib')
    joblib.dump(selector_kbest, 'models/selector1_high_accuracy.joblib')
    joblib.dump(pca, 'models/pca1_high_accuracy.joblib')
    
    logger.info("✅ High-Accuracy Model 1 saved successfully")
    
    return ensemble, scaler, selector_kbest, pca, accuracy

def main():
    """Create high-accuracy Model 1."""
    logger.info("Creating high-accuracy Model 1 with advanced ML techniques...")
    
    try:
        model, scaler, selector, pca, acc = create_high_accuracy_model1()
        logger.info(f"✅ High-Accuracy Model 1 created with accuracy: {acc:.4f}")
    except Exception as e:
        logger.error(f"❌ Failed to create High-Accuracy Model 1: {str(e)}")
    
    logger.info("🎉 High-accuracy model creation completed!")

if __name__ == "__main__":
    main()
