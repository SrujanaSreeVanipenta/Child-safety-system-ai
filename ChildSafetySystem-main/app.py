from flask import Flask, render_template, request
import time
import os
from threading import Thread
from alert import send_voice_alert, send_manual_sos
import pickle
import numpy as np
import librosa
from datetime import datetime
import pytz
#import sounddevice as sd
#from scipy.io.wavfile import write
import speech_recognition as sr

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Load trained model
with open("emotion_model.pkl", "rb") as f:
    model = pickle.load(f)


# ---------------- Background Alert Function ----------------

# ---------------- Feature Extraction ----------------
def extract_features(file_path):

    audio, sample_rate = librosa.load(file_path, sr=22050)

    audio, _ = librosa.effects.trim(audio)
    audio = librosa.util.normalize(audio)

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


# ---------------- Home Route ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- Detection Route ----------------
@app.route("/detect", methods=["POST"])
def detect_emotion():

    audio_file = request.files["audio"]

    file_path = "temp.wav"

    audio_file.save(file_path)

    latitude = request.form.get("latitude")
    longitude = request.form.get("longitude")

    # -------- Speech Recognition --------
    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(file_path) as source:

            audio_data = recognizer.record(source)

        detected_text = recognizer.recognize_google(audio_data, language="en-IN")

        print("Detected speech:", detected_text)

    except Exception as e:
        print("Speech recognition error:", e)
        detected_text = "Speech not recognized"

    # -------- Keyword Detection --------
    emergency_keywords = ["help", "save me", "danger", "please help"]

    keyword_distress = False

    for word in emergency_keywords:
        if word in detected_text.lower():
            keyword_distress = True
            break

    # Load audio file first
    audio, sample_rate = librosa.load(file_path, sr=22050)

    # Only process audio AFTER confirming speech exists
    audio, _ = librosa.effects.trim(audio)
    audio = librosa.effects.preemphasis(audio)
    audio = librosa.util.normalize(audio)

    # -------- Feature Extraction --------
    features = extract_features(file_path)

    proba = model.predict_proba(features)[0]

    classes = model.classes_

    print("Emotion probabilities:", dict(zip(classes, proba)))

    max_prob = np.max(proba)

    prediction = classes[np.argmax(proba)]

    # Stabilize weak predictions
    if max_prob < 0.50:
        prediction = "neutral"

    distress_emotions = ["fear", "angry"]

    emotion_distress = prediction in distress_emotions and max_prob >= 0.4

    status = "No Distress"

    # Create location link
    location_link = f"https://maps.google.com/?q={latitude},{longitude}"

    # -------- Final Decision --------
    if emotion_distress or keyword_distress:

        status = "⚠ Distress Detected!"

        Thread(
            target=send_voice_alert,
            args=(prediction, detected_text, location_link)
        ).start()

    prob_dict = {}

    for emotion_name, prob in zip(classes, proba):
        prob_dict[str(emotion_name)] = round(float(prob), 3)

    return render_template(
        "result.html",
        emotion=prediction,
        confidence=round(float(max_prob), 3),
        probabilities=prob_dict,
        speech_text=detected_text,
        status=status
    )


# ---------------- Manual SOS Route ----------------
@app.route("/sos")
def sos():
    
    latitude = request.args.get("latitude")
    longitude = request.args.get("longitude")

    location_link = f"https://maps.google.com/?q={latitude},{longitude}"

    status = "🚨 Manual SOS Triggered!"

    Thread(
        target=send_manual_sos,
        args=(location_link,)
    ).start()

    return render_template(
        "result.html",
        emotion="N/A",
        confidence="Not predictable",
        probabilities={},
        speech_text="Manual emergency triggered by User",
        status=status
    )


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)