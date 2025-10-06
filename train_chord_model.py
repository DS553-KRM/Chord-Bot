import numpy as np
import itertools
import joblib
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# === Chord definitions ===
CHORD_TYPES = {
    "major": [0, 4, 7],
    "minor": [0, 3, 7],
    "dominant7": [0, 4, 7, 10],
    "major7": [0, 4, 7, 11],
    "minor7": [0, 3, 7, 10],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
}

# === Feature encoding ===
def notes_to_vector(pcs):
    vec = np.zeros(12)
    for p in pcs:
        vec[p] = 1
    return vec

def intervals_from_pcs(pcs):
    n = len(pcs)
    if n < 2:
        return np.zeros(12)
    intervals = [(pcs[j] - pcs[i]) % 12 for i in range(n) for j in range(i + 1, n)]
    profile = np.zeros(12)
    for i in intervals:
        profile[i] += 1
    return profile / np.max(profile)

def encode_features(pcs):
    return np.concatenate([notes_to_vector(pcs), intervals_from_pcs(pcs)])

# === Dataset generation ===
X, y = [], []

for name, intervals in CHORD_TYPES.items():
    for root in range(12):
        base = [(root + i) % 12 for i in intervals]
        for _ in range(100):
            pcs = sorted(set(base))
            # random inversion (rotate chord order)
            shift = random.randint(0, len(pcs) - 1)
            pcs = pcs[shift:] + pcs[:shift]
            # add small Gaussian noise to improve generalization
            vec = encode_features(pcs)
            vec += np.random.normal(0, 0.02, size=vec.shape)
            X.append(vec)
            y.append(name)

X, y = np.array(X), np.array(y)
print(f"🎵 Generated {len(X)} samples across {len(set(y))} chord types.")

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

clf = RandomForestClassifier(
    n_estimators=400, max_depth=25, random_state=42, n_jobs=-1
)

clf = RandomForestClassifier(
    n_estimators=400, max_depth=25, random_state=42, n_jobs=-1
)
clf.fit(X_train, y_train)

print("📊 Evaluating model...")
print(classification_report(y_test, clf.predict(X_test)))

joblib.dump(clf, "chord_classifier.pkl")
print("✅ Model trained and saved as chord_classifier.pkl")

