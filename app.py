import gradio as gr
from transformers import pipeline

# Load FLAN-T5 small locally
generator = pipeline("text2text-generation", model="google/flan-t5-small")

def chord_bot(prompt: str) -> str:
    """
    Use FLAN-T5 with few-shot prompting to explain chord progressions in musical terms.
    """
    chord_prompt = f"""
    Given a group of notes, identify what chord those notes make

    Example:
    Input: C E G
    Output: This is a C major chord

    Example:
    Input: D F# A C#
    Output: This is a D major 7 chord.

    Now explain:
    Input: {prompt}
    Output:
    """

    # Generate explanation
    response = generator(
        chord_prompt,
        max_new_tokens=150,
        temperature=0.9,
        top_p=0.95
    )

    # Post-process to clean output
    raw_text = response[0]["generated_text"]
    answer = raw_text.split("Output:")[-1].strip()
    return answer

# Gradio UI
iface = gr.Interface(
    fn=chord_bot,
    inputs=gr.Textbox(lines=2, placeholder="Type a chord progression (e.g., C D E G)"),
    outputs="text",
    title="🎶 Locally-Hosted Chord Bot",
    description="This version runs FLAN-T5 locally with few-shot prompting to explain chord progressions."
)

if __name__ == "__main__":
    iface.launch()
