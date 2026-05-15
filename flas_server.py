from flask import Flask, request, jsonify
import numpy as np
import librosa
import base64
import soundfile as sf
from tensorflow.keras.models import load_model
import logging
import noisereduce as nr
from scipy.signal import butter, lfilter
import os
from datetime import datetime

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load trained LSTM model
model = load_model("lstm_verse_classifier.keras")

# Constants
RATE = 16000
N_MFCC = 13
MAX_LEN = 120
SAVE_AUDIO = True
AUDIO_DIR = "received_audio"
CONFIDENCE_THRESHOLD = 0.4

# Create folder to save audio
os.makedirs(AUDIO_DIR, exist_ok=True)

# Verse texts for each Surah
verse_texts_by_surah = {
    "al-ikhlas": {
        1: "قُلْ هُوَ اللَّهُ أَحَدٌ",
        2: "اللَّهُ الصَّمَدُ",
        3: "لَمْ يَلِدْ وَلَمْ يُولَدْ",
        4: "وَلَمْ يَكٜنْ لَهُ كُفٌوًا أَحَدٌ"
    },
    "al-fatihah": {
        1: "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
        2: "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ",
        3: "الرَّحْمَٰنِ الرَّحِيمِ",
        4: "مَالِكِ يَوْمِ الدِّينِ",
        5: "إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ",
        6: "اهْدِنَا الصِّرَاطَ الْمُسْتَقِيمَ",
        7: "صِرَاطَ الَّذِينَ أَنْعَمْتَ عَلَيْهِمْ غَيْرِ الْمَغْضُوبِ عَلَيْهِمْ وَلَا الضَّالِّينَ"
    },
    "an-nas": {
        1: "قُلْ أَعُوذُ بِرَبِّ النَّاسِ",
        2: "مَلِكِ النَّاسِ",
        3: "إِلَّهِ النَّاسِ",
        4: "مِنْ شَرِّ الوَسْوَاسِ الْخَنَّاسِ",
        5: "الَّذِيْ يُوسْوِٖسُ فِي صُدُورِ النَّاسِ",
        6: "مِنَ الجِنَّةِ وَالنَّاسِ"
    }
}

# High-pass filter to remove low-frequency rumble
def highpass_filter(audio, sr, cutoff=100, order=5):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return lfilter(b, a, audio)

# Prediction pipeline
def predict_from_audio(y):
    if not np.all(np.isfinite(y)) or np.max(np.abs(y)) < 1e-5:
        logging.warning("⚠️ Very low signal or silent audio received")
        return None, 0.0

    #y_trimmed, _ = librosa.effects.trim(y, top_db=45)
    #if len(y_trimmed) == 0:
    #    logging.warning("⚠️ Silence trimming removed all audio")
    #    return None, 0.0

    # Use raw audio without trimming
    y_trimmed = y

    # Optional: warn if amplitude is super low
    if np.max(np.abs(y_trimmed)) < 1e-5:
        logging.warning("⚠️ Possibly too quiet signal (not trimmed)")

    y_filtered = highpass_filter(y_trimmed, sr=RATE)
    reduced = nr.reduce_noise(y=y_filtered, sr=RATE, prop_decrease=0.8)

    mfcc = librosa.feature.mfcc(y=reduced, sr=RATE, n_mfcc=N_MFCC).T
    mfcc = np.nan_to_num(mfcc)

    if mfcc.shape[0] < 5:
        logging.warning(f"⚠️ MFCC too short: shape={mfcc.shape}")
        return None, 0.0

    if mfcc.shape[0] < MAX_LEN:
        mfcc = np.pad(mfcc, ((0, MAX_LEN - mfcc.shape[0]), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:MAX_LEN, :]

    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    mfcc = np.expand_dims(mfcc, axis=0)

    prediction = model.predict(mfcc, verbose=0)
    verse_index = int(np.argmax(prediction)) + 1
    confidence = float(np.max(prediction))
    return verse_index, confidence

@app.route("/predict", methods=["POST"])
def predict():
    try:
        audio_base64 = request.form.get("audio_base64")
        surah = request.form.get("surah", "").lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if not audio_base64 or not surah:
            return jsonify({
                "status": "error",
                "message": "Missing audio or surah name"
            }), 400

        if surah not in verse_texts_by_surah:
            return jsonify({
                "status": "error",
                "message": "Surah not supported"
            }), 400

        audio_bytes = base64.b64decode(audio_base64)
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        logging.info(f"[AUDIO] Received {len(audio_np)} samples from Surah: {surah}")

        if SAVE_AUDIO:
            filename = f"{surah}_{timestamp}.wav"
            sf.write(os.path.join(AUDIO_DIR, filename), audio_np, RATE)

        verse_index, confidence = predict_from_audio(audio_np)
        logging.info(f"[PREDICTION] Surah: {surah}, Index: {verse_index}, Confidence: {confidence:.3f}")

        if verse_index is None or confidence < CONFIDENCE_THRESHOLD:
            if SAVE_AUDIO:
                fail_filename = f"FAILED_{surah}_{timestamp}.wav"
                sf.write(os.path.join(AUDIO_DIR, fail_filename), audio_np, RATE)

            return jsonify({
                "status": "low_confidence",
                "message": "Low confidence or unreadable audio",
                "confidence": confidence,
                "timestamp": timestamp
            }), 200

        return jsonify({
            "status": "success",
            "verse_index": verse_index,
            "confidence": confidence,
            "verse_text": verse_texts_by_surah[surah].get(verse_index, ""),
            "timestamp": timestamp
        })

    except Exception as e:
        logging.exception("Error during prediction")
        return jsonify({
            "status": "error",
            "message": f"Error: {str(e)}"
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=False)
