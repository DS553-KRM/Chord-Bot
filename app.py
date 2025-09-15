import gradio as gr
import joblib
import numpy as np
import os
import re
import subprocess

NAME_TO_PC = {
    "C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,"E":4,"F":5,"F#":6,"Gb":6,
    "G":7,"G#":8,"Ab":8,"A":9,"A#":10,"Bb":10,"B":11
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

MODEL_PATH = "chord_classifier.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        print("⚠️ chord_classifier.pkl not found. Training model...")
        subprocess.run(["python", "train_chord_model.py"], check=True)
    return joblib.load(MODEL_PATH)

clf = load_model()

def chord_bot(message: str, history: list[tuple[str,str]]):
    vec = notes_to_vector(message)
    if np.sum(vec) < 2:
        return "⚠️ Please enter at least 2 distinct notes (e.g., C E G)"
    label = clf.predict([vec])[0]
    return f"🎵 Identified chord: **{label}**"

chatbot = gr.ChatInterface(
    fn=chord_bot,
    title="🎶 ML Chord Bot",
    description="Enter 2+ notes (e.g., C E G or Db F Ab C). Powered by a trained RandomForest classifier."
)

if __name__ == "__main__":
    chatbot.launch()

#hi ryan
