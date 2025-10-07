import gradio as gr
import joblib
import numpy as np
import os
import re
import subprocess
import sys
import traceback

MODEL_PATH = "chord_classifier.pkl"

# --- Note parsing ---
NAME_TO_PC = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11
}
NOTE_TOKEN_RE = re.compile(r"[A-Ga-g](?:#|b)?")

# --- Feature vector construction ---
def build_feature_vector(notes_str: str):
    """
    Builds a 24-feature vector (12 note presence + 12 interval presence)
    to match the model's training representation.
    """
    tokens = NOTE_TOKEN_RE.findall(notes_str)
    pcs = sorted({NAME_TO_PC.get(t.upper(), None) for t in tokens if t})
    pcs = [p for p in pcs if p is not None]
    if len(pcs) == 0:
        return np.zeros(24)

    # 12-note presence
    note_vec = np.zeros(12)
    for p in pcs:
        note_vec[p] = 1

    # 12-interval presence (root-relative distances)
    intervals = np.zeros(12)
    for i in range(len(pcs)):
        for j in range(i + 1, len(pcs)):
            diff = abs(pcs[j] - pcs[i]) % 12
            intervals[diff] = 1

    # Concatenate note and interval features
    return np.concatenate([note_vec, intervals])

# --- Model handling ---
def retrain_model():
    """Always retrain model on startup for consistency across environments."""
    print("Retraining chord classifier (forced rebuild)...")
    try:
        subprocess.run([sys.executable, "train_chord_model.py"], check=True)
        print("✅ Model retrained and saved successfully.")
    except subprocess.CalledProcessError as e:
        print("❌ Error retraining model:", e)
        traceback.print_exc()
        sys.exit(1)

def load_model():
    retrain_model()
    return joblib.load(MODEL_PATH)

clf = load_model()

# --- Prediction logic ---
def chord_bot(message: str, history: list[tuple[str, str]]):
    vec = build_feature_vector(message)
    if np.sum(vec[:12]) < 2:
        return "⚠️ Please enter at least 2 distinct notes (e.g., C E G)"
    try:
        pred = clf.predict([vec])[0]
        return f"🎵 Identified chord: **{pred}**"
    except Exception as e:
        traceback.print_exc()
        return f"❌ Prediction error: {str(e)}"

# --- Gradio app ---
chatbot = gr.ChatInterface(
    fn=chord_bot,
    title="🎶 ML Chord Bot (Auto-Recovering)",
    description="Automatically retrains model on startup and uses interval features for robust chord recognition."
)

# --- Launch ---
if __name__ == "__main__":
    host = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    chatbot.launch(server_name=host, server_port=port, share=False)
