import os
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Change this to your dataset folder path
DATASET_PATH = "dataset"

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, duration=3, offset=0.5)

    mfcc = np.mean(
        librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T,
        axis=0
    )

    chroma = np.mean(
        librosa.feature.chroma_stft(y=audio, sr=sample_rate).T,
        axis=0
    )

    spectral_contrast = np.mean(
        librosa.feature.spectral_contrast(y=audio, sr=sample_rate).T,
        axis=0
    )

    energy = np.mean(
        librosa.feature.rms(y=audio).T,
        axis=0
    )

    features = np.hstack([mfcc, chroma, spectral_contrast, energy])

    return features

features = []
labels = []

# Dataset structure assumed:
# dataset/emotion_name/audio.wav

for emotion in os.listdir(DATASET_PATH):
    emotion_folder = os.path.join(DATASET_PATH, emotion)
    
    for file in os.listdir(emotion_folder):
        file_path = os.path.join(emotion_folder, file)
        
        try:
            feature = extract_features(file_path)
            features.append(feature)
            labels.append(emotion)
        except:
            continue

X = np.array(features)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
with open("emotion_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")
