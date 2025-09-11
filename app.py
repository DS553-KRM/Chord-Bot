import gradio as gr
from transformers import pipeline

# Load a lightweight local model (works on Hugging Face Spaces free tier)
generator = pipeline("text-generation", model="distilgpt2")

def chord_bot(prompt: str) -> str:
    """
    Uses prompt engineering to guide GPT-2 toward music/chord explanations.
    """
    # Add strong context to steer the model
    chord_prompt = f"""
    You are a helpful music assistant. 
    Explain the following chord progression in musical terms and suggest how it could be used in a song:

    {prompt}
    """

    response = generator(
        chord_prompt,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
    return response[0]["generated_text"]

# Gradio UI
iface = gr.Interface(
    fn=chord_bot,
    inputs=gr.Textbox(lines=2, placeholder="Type a chord progression (e.g., C D E G)"),
    outputs="text",
    title="🎶 Locally-Hosted Chord Bot",
    description="This version runs GPT-2 locally, with prompt engineering to explain chord progressions."
)

if __name__ == "__main__":
    iface.launch()
