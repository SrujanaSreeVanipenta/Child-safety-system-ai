import pickle
import numpy as np
import librosa
import sounddevice as sd
from scipy.io.wavfile import write

# Load trained model
with open("emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

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
    return features.reshape(1, -1)

# Record audio
duration = 4
sample_rate = 22050

print("Speak now...")
recording = sd.rec(int(duration * sample_rate),
                   samplerate=sample_rate,
                   channels=1)
sd.wait()

write("temp.wav", sample_rate, recording)

# Extract features
features = extract_features("temp.wav")

# Get prediction probabilities
proba = model.predict_proba(features)[0]
classes = model.classes_

print("\nPrediction Probabilities:")
for emotion, prob in zip(classes, proba):
    print(f"{emotion}: {prob:.3f}")

max_prob = np.max(proba)
prediction = classes[np.argmax(proba)]

print("\nHighest Confidence:", round(max_prob, 3))

distress_emotions = ["fear", "angry"]

if prediction in distress_emotions and max_prob >= 0.4:
    print("Predicted Emotion:", prediction)
    print("⚠ Distress detected!")
elif max_prob < 0.4:
    print("Low confidence prediction. Treating as Normal.")
else:
    print("Predicted Emotion:", prediction)
    print("No distress detected.")