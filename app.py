import gradio as gr
from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")

def chord_bot(prompt: str) -> str:
    """
    Generate a response using the local Hugging Face model.
    This simulates a chord/music assistant, but could be extended with
    domain-specific prompts or fine-tuned models.
    """
    response = generator(
        prompt,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
    return response[0]["generated_text"]

# Gradio UI
iface = gr.Interface(
    fn=chord_bot,
    inputs=gr.Textbox(lines=2, placeholder="Type a chord progression or music-related question..."),
    outputs="text",
    title="🎶 Locally-Hosted Chord Bot",
    description="This version of Chord Bot runs a local Hugging Face Transformers pipeline inside the Space container."
)

if __name__ == "__main__":
    iface.launch()
