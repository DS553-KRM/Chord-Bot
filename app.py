import gradio as gr
from transformers import pipeline

# Load FLAN-T5 small for instruction following
generator = pipeline("text2text-generation", model="google/flan-t5-small")

def chord_bot(prompt: str) -> str:
    """
    Use Flan-T5 to explain a chord progression in musical terms.
    """
    chord_prompt = f"Explain this chord progression in musical terms and how it could be used in a song: {prompt}"
    response = generator(chord_prompt, max_new_tokens=128)
    return response[0]["generated_text"]

# Gradio UI
iface = gr.Interface(
    fn=chord_bot,
    inputs=gr.Textbox(lines=2, placeholder="Type a chord progression (e.g., C D E G)"),
    outputs="text",
    title="🎶 Locally-Hosted Chord Bot",
    description="This version runs FLAN-T5 locally, with instruction-tuned prompting to explain chord progressions."
)

if __name__ == "__main__":
    iface.launch()
