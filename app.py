import gradio as gr
from transformers import pipeline

# Load FLAN-T5 small locally
generator = pipeline("text2text-generation", model="google/flan-t5-small")

def chord_bot(message: str, history: list[tuple[str, str]]):
    """
    Use FLAN-T5 with few-shot prompting to identify chords given a list of notes.
    """
    chord_prompt = f"""
You are a chord identifier. Given 3 or more notes, return the chord name.

Examples:
Input: C E G
Output: C major

Input: D F# A
Output: D major

Input: D F# A C
Output: D7 (dominant 7)

Input: C Eb G
Output: C minor

Input: C Eb G Bb
Output: Cm7 (minor 7)

Now, identify this chord:
Input: {message}
Output:
"""

    # Generate answer
    response = generator(
        chord_prompt,
        max_new_tokens=32,
        temperature=0.0,  # deterministic
        top_p=0.95
    )

    # Post-process
    raw_text = response[0]["generated_text"]
    if "Output:" in raw_text:
        answer = raw_text.split("Output:")[-1].strip()
    else:
        answer = raw_text.strip()
    return answer

# Gradio Chat UI
chatbot = gr.ChatInterface(
    fn=chord_bot,
    title="🎶 FLAN-T5 Chord Bot",
    description="Type 3+ notes (e.g., C E G or Db F Ab C) and I'll identify the chord."
)

if __name__ == "__main__":
    chatbot.launch()
