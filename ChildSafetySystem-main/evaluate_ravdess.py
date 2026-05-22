import os
import pickle
import librosa
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
with open("emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

# Path to RAVDESS dataset
dataset_path = "RAVDESS"

emotion_map = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear"
}

X_test = []
y_test = []

# Read all actor folders
for actor in os.listdir(dataset_path):

    actor_path = os.path.join(dataset_path, actor)

    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):

        if file.endswith(".wav"):

            parts = file.split("-")

            emotion_code = parts[2]

            if emotion_code not in emotion_map:
                continue

            emotion = emotion_map[emotion_code]

            file_path = os.path.join(actor_path, file)

            # Load audio
            audio, sample_rate = librosa.load(file_path, sr=22050)

            audio, _ = librosa.effects.trim(audio)
            audio = librosa.util.normalize(audio)

            # Feature extraction
            mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
            chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
            spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sample_rate).T, axis=0)
            energy = np.mean(librosa.feature.rms(y=audio).T, axis=0)

            features = np.hstack([mfcc, chroma, spectral_contrast, energy])

            X_test.append(features)
            y_test.append(emotion)

X_test = np.array(X_test)

# Predict
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")
plt.title("Confusion Matrix for Emotion Classification")

plt.savefig("confusion_matrix.png", dpi=300)

plt.show()