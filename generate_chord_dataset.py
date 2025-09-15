import csv

PITCHES_SHARP = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

NAME_TO_PC = {
    "C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,"E":4,"F":5,"F#":6,"Gb":6,
    "G":7,"G#":8,"Ab":8,"A":9,"A#":10,"Bb":10,"B":11
}

CHORD_FORMULAS = {
    "maj": [0,4,7],
    "min": [0,3,7],
    "dim": [0,3,6],
    "aug": [0,4,8],
    "7":   [0,4,7,10],
    "maj7":[0,4,7,11],
    "m7":  [0,3,7,10],
    "mMaj7":[0,3,7,11],
    "dim7":[0,3,6,9],
    "m7b5":[0,3,6,10],
    "6":   [0,4,7,9],
    "m6":  [0,3,7,9],
    "sus2":[0,2,7],
    "sus4":[0,5,7],
}

def generate_chord_vectors():
    data = []
    for root_pc in range(12):
        for quality, intervals in CHORD_FORMULAS.items():
            pcs = [(root_pc + i) % 12 for i in intervals]
            vec = [1 if i in pcs else 0 for i in range(12)]
            root_name = PITCHES_SHARP[root_pc]
            chord_name = root_name + quality
            data.append((vec, chord_name))
    return data

def save_to_csv(filename="chords_dataset.csv"):
    data = generate_chord_vectors()
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"pc_{p}" for p in PITCHES_SHARP] + ["label"]
        writer.writerow(header)
        for vec, label in data:
            writer.writerow(vec + [label])
    print(f"✅ Saved {len(data)} chords to {filename}")

if __name__ == "__main__":
    save_to_csv()
