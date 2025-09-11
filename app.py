# app.py
import gradio as gr
from typing import List, Tuple
import re

# ----- Pitch utilities -----
SHARP_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
FLAT_NAMES  = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]

NAME_TO_PC = {
    # naturals
    "C":0, "D":2, "E":4, "F":5, "G":7, "A":9, "B":11,
    # sharps
    "C#":1, "D#":3, "F#":6, "G#":8, "A#":10,
    # flats
    "Db":1, "Eb":3, "Gb":6, "Ab":8, "Bb":10,
    # unicode ♯/♭
    "C♯":1, "D♯":3, "F♯":6, "G♯":8, "A♯":10,
    "D♭":1, "E♭":3, "G♭":6, "A♭":8, "B♭":10,
}

# Common chord templates: set of intervals from root (0 always present)
CHORD_TEMPLATES = {
    (0,4,7): ("major triad", ""),
    (0,3,7): ("minor triad", "m"),
    (0,3,6): ("diminished triad", "dim"),
    (0,4,8): ("augmented triad", "+"),
    (0,2,7): ("sus2", "sus2"),
    (0,5,7): ("sus4", "sus4"),
    (0,4,7,10): ("dominant 7th", "7"),
    (0,4,7,11): ("major 7th", "maj7"),
    (0,3,7,10): ("minor 7th", "m7"),
    (0,3,6,10): ("half-diminished (m7♭5)", "m7b5"),
    (0,3,6,9): ("diminished 7th", "dim7"),
    (0,3,7,11): ("minor major 7th", "m(maj7)"),
    (0,4,7,9): ("major 6th", "6"),
    (0,3,7,9): ("minor 6th", "m6"),
}

ADD_TONES = {
    2: ("add9", "add9"),  # treat 2 as add9
    9: ("add9", "add9"),
    11: ("add11", "add11"),
    6: ("add13", "add13"),
}

SUS_OVERLAPS = {(0,2,7), (0,5,7)}

NOTE_TOKEN_RE = re.compile(r"[A-Ga-g](?:#|b|♯|♭)?")


def pc_name(pc: int, prefer_flats: bool) -> str:
    return (FLAT_NAMES if prefer_flats else SHARP_NAMES)[pc % 12]


def parse_notes(user_text: str) -> Tuple[List[int], bool, List[str]]:
    """Return (pitch_classes, prefer_flats, original_tokens)."""
    tokens = NOTE_TOKEN_RE.findall(user_text)
    tokens = [t.upper().replace("♯", "#").replace("♭", "b") for t in tokens]
    prefer_flats = any("B" in t and len(t)>1 for t in tokens) or any("b" in t for t in tokens)
    pcs = []
    for t in tokens:
        if t in NAME_TO_PC:
            pcs.append(NAME_TO_PC[t])
    # dedupe while preserving order
    seen = set()
    pcs_unique = []
    for p in pcs:
        if p not in seen:
            pcs_unique.append(p)
            seen.add(p)
    return pcs_unique, prefer_flats, tokens


def intervals_from_root(pcs: List[int], root: int) -> Tuple[int,...]:
    ints = sorted(((p - root) % 12) for p in pcs)
    if 0 not in ints:
        ints = (0,) + tuple(i for i in ints)
    return tuple(sorted(set(ints)))


def describe_chord(pcs: List[int], prefer_flats: bool) -> str:
    if len(pcs) < 3:
        return "Please provide at least 3 distinct note names (e.g., C E G or C, Eb, G)."

    matches = []
    for root in pcs:  # try each included pitch as potential root
        base_ints = intervals_from_root(pcs, root)
        # Try exact template match first (triads/sevenths/6ths/etc.)
        if base_ints in CHORD_TEMPLATES:
            qual_name, suffix = CHORD_TEMPLATES[base_ints]
            matches.append((root, qual_name, suffix, []))
            continue
        # Try triad with added tones (add9/add11/add13)
        # Identify a triad subset and extra tones
        for triad in [(0,4,7), (0,3,7), (0,3,6), (0,4,8)]:
            triad_set = set(triad)
            if triad_set.issubset(set(base_ints)):
                extras = sorted(set(base_ints) - triad_set)
                add_suffixes = []
                for e in extras:
                    if e in ADD_TONES:
                        add_suffixes.append(ADD_TONES[e][1])
                if add_suffixes:
                    qual_name, suffix = CHORD_TEMPLATES.get(triad, ("triad",""))
                    matches.append((root, f"{qual_name} with {'/'.join(add_suffixes)}", suffix + ("" if not add_suffixes else ("("+" ".join(add_suffixes)+")")), extras))
        # sus chords (if match) possibly with added tones
        for sus in [(0,2,7), (0,5,7)]:
            if set(sus).issubset(set(base_ints)):
                extras = sorted(set(base_ints) - set(sus))
                add_suffixes = []
                for e in extras:
                    if e in ADD_TONES:
                        add_suffixes.append(ADD_TONES[e][1])
                qual_name, suffix = CHORD_TEMPLATES[sus]
                matches.append((root, qual_name + (" with "+"/".join(add_suffixes) if add_suffixes else ""), suffix + ("" if not add_suffixes else ("("+" ".join(add_suffixes)+")")), extras))

    if not matches:
        # fallback: show interval set relative to lowest note as a hint
        root_guess = min(pcs)
        ints = intervals_from_root(pcs, root_guess)
        return (
            "I couldn't confidently name that chord. Interval set from {}: {}.\n"
            "Try removing extensions/duplicates or check note spelling (sharps vs flats)."
        ).format(pc_name(root_guess, prefer_flats), ",".join(str(i) for i in ints))

    # Rank matches: prefer templates with no ambiguous extras, prefer 7th/6th over triad with adds, then by root being lowest provided note
    def rank(m):
        root, qual_name, suffix, extras = m
        score = 0
        if "7" in suffix or "6" in suffix:
            score += 2
        if not extras:
            score += 1
        if root == min(pcs):
            score += 0.5
        return -score

    matches.sort(key=rank)

    # Build a readable response listing top 1-3 candidates
    top = matches[:3]
    lines = []
    for i,(root, qual_name, suffix, extras) in enumerate(top, start=1):
        name = pc_name(root, prefer_flats)
        symbol = name + (suffix if suffix else "")
        spelled = ", ".join(pc_name(p, prefer_flats) for p in sorted(set(pcs)))
        lines.append(f"{i}. {symbol}  — {qual_name} (notes: {spelled})")

    # Inversion hint (best-effort): if lowest note isn't the chosen root, suggest slash chord
    chosen_root = top[0][0]
    lowest = min(pcs)
    if lowest != chosen_root:
        lines[0] += f"  — likely {pc_name(chosen_root, prefer_flats)}/{pc_name(lowest, prefer_flats)}"

    return "\n".join(lines)


def answer(message: str, history: List[Tuple[str,str]]):
    pcs, prefer_flats, tokens = parse_notes(message)
    if not tokens:
        return (
            "Tell me 3+ notes (e.g., 'C E G' or 'Db, F, Ab, C'). "
            "I support #/b and unicode ♯/♭."
        )
    return describe_chord(pcs, prefer_flats)


def example_inputs():
    return [
        "C E G",
        "D F# A C",
        "C Eb G Bb",
        "F A C D",
        "G Bb D F",
        "Db F Ab C",
        "C D G",
        "A C E G#",
        "E G# B D",
        "C Eb Gb A"
    ]

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎵 Note → Chord Chatbot
    Type 3 or more note names and I'll guess the chord (triads, 7ths, 6ths, sus, and common add tones).
    - Accepted formats: `C E G`, `Db, F, Ab, C`, `G-B-D-F`, etc.
    - Sharps/flats: `#`, `b`, or unicode `♯` `♭`.
    """)
    chat = gr.ChatInterface(
        fn=answer,
        examples=[[e] for e in example_inputs()],
        title="Chord Identifier",
        retry_btn=None,
        undo_btn="Delete last turn",
        clear_btn="Clear",
        textbox=gr.Textbox(placeholder="e.g., C E G or Db, F, Ab", label="Your notes"),
    )

if __name__ == "__main__":
    demo.launch()