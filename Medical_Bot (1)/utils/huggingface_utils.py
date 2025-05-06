# huggingface_utils.py

from transformers import pipeline

# Load a lightweight model
generator = pipeline("text-generation", model="distilgpt2")

def generate_response(prompt):
    try:
        result = generator(prompt, max_new_tokens=100)
        return result[0]["generated_text"]
    except Exception as e:
        return f"Error generating response: {str(e)}"


