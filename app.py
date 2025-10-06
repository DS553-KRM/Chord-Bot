import gradio as gr
import joblib
import numpy as np

# -------------------------------------
# 🔤 Note definitions
# -------------------------------------
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]

# -------------------------------------
# 🧠 Load trained model
# -------------------------------------
clf = joblib.load("chord_classifier.pkl")

# -------------------------------------
# 🧩 Helper functions
# -------------------------------------
def normalize_note(n):
    n = n.strip().upper()
    n = n.replace("DB", "C#").replace("EB", "D#").replace("GB", "F#").replace("AB", "G#").replace("BB", "A#")
    return n if n in NOTE_NAMES else None

def notes_to_pcs(notes):
    pcs = [NOTE_NAMES.index(n) for n in notes if n in NOTE_NAMES]
    return sorted(set(pcs))

def notes_to_vector(pcs):
    vec = np.zeros(12)
    for p in pcs:
        vec[p] = 1
    return vec

def intervals_from_pcs(pcs):
    """Compute normalized interval profile for pitch classes."""
    n = len(pcs)
    if n < 2:
        return np.zeros(12)
    intervals = [(pcs[j] - pcs[i]) % 12 for i in range(n) for j in range(i + 1, n)]
    profile = np.zeros(12)
    for i in intervals:
        profile[i] += 1
    return profile / np.max(profile)

def encode_features(pcs):
    """Combine pitch-class vector + interval features."""
    return np.concatenate([notes_to_vector(pcs), intervals_from_pcs(pcs)])

# -------------------------------------
# 🎶 Chord prediction logic
# -------------------------------------
def predict_chord(notes):
    # Clean and map to pitch classes
    clean_notes = [normalize_note(n) for n in notes]
    clean_notes = [n for n in clean_notes if n]
    pcs = notes_to_pcs(clean_notes)
    if not pcs:
        return "⚠️ Invalid notes. Try again with something like `C E G`."

    # Identify chord quality
    features = encode_features(pcs).reshape(1, -1)
    proba = clf.predict_proba(features)[0]
    chord_quality = clf.classes_[np.argmax(proba)]
    conf = np.max(proba)

    # Identify chord root — lowest pitch, adjusted to modulo 12
    # (Finds the most plausible root by aligning interval pattern)
    best_root = None
    min_distance = float("inf")
    for i in pcs:
        rotated = sorted([(p - i) % 12 for p in pcs])
        distance = np.sum(np.diff(rotated))  # measure compactness
        if distance < min_distance:
            min_distance = distance
            best_root = i

    root_note = NOTE_NAMES[best_root]
    return f"🎵 Identified chord: {root_note} {chord_quality} (confidence: {conf:.2f})"

# -------------------------------------
# 💬 Chatbot wrapper
# -------------------------------------
def chord_bot(message, history):
    notes = message.split()
    return predict_chord(notes)

# -------------------------------------
# 🚀 Launch Gradio app
# -------------------------------------
chatbot = gr.ChatInterface(
    fn=chord_bot,
    title="🎶 Local Chord Identifier",
    description="Enter notes like `C E G` or `F A C E` to identify the chord!",
)

if __name__ == "__main__":
    chatbot.launch(server_name="0.0.0.0", server_port=7861, share=True)

