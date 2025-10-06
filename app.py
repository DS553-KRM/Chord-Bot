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

def notes_to_vector(notes_str: str):
    tokens = NOTE_TOKEN_RE.findall(notes_str)
    pcs = [NAME_TO_PC.get(t.upper(), None) for t in tokens]
    pcs = [p for p in pcs if p is not None]
    vec = np.zeros(12)
    for p in pcs:
        vec[p] = 1
    return vec

# --- Model handling ---
def retrain_model():
    """Always retrain model on startup for consistency across environments."""
    print("🧠 Retraining chord classifier (forced rebuild)...")
    try:
        subprocess.run(
            [sys.executable, "train_chord_model.py"],
            check=True
        )
        print("✅ Model retrained and saved successfully.")
    except subprocess.CalledProcessError as e:
        print("❌ Error retraining model:", e)
        traceback.print_exc()
        sys.exit(1)

def load_model():
    # Always retrain on startup
    retrain_model()
    return joblib.load(MODEL_PATH)

clf = load_model()

# --- Prediction logic ---
def chord_bot(message: str, history: list[tuple[str, str]]):
    vec = notes_to_vector(message)
    if np.sum(vec) < 2:
        return "⚠️ Please enter at least 2 distinct notes (e.g., C E G)"

    try:
        pred = clf.predict([vec])[0]
        return f"🎵 Identified chord: **{pred}**"
    except Exception as e:
        traceback.print_exc()
        return f"❌ Prediction error: {str(e)}"

chatbot = gr.ChatInterface(
    fn=chord_bot,
    title="🎶 ML Chord Bot (Auto-Recovering)",
    description="Automatically retrains model on startup to avoid pickle mismatches."
)

if __name__ == "__main__":
    import os

   host = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
   port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))  # default to 7860

   chatbot.launch(server_name=host, server_port=port, share=False)

