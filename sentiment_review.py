import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


from huggingface_hub import login
login(token="hf_XXXXXX")  # Replace with your Hugging Face token
# Check if GPU is available and set device accordingly

def llm_response(prompt, model_name="meta-llama/Llama-2-7b-chat-hf"):
    """
    Function to perform sentiment analysis using a LLaMA-based model.
    
    Parameters:
    - prompt (str): The input prompt for the model.
    - model_name (str): The Hugging Face model name or path.
    
    Returns:
    - str: The generated response from the model.
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Create pipeline for text generation
    text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Generate response
    response = text_generator(prompt, max_length=100, do_sample=True, temperature=0.7, truncation=True)
    return response[0]["generated_text"]

# Define the prompt for sentiment classification
prompt = """Classify the sentiment of the following review as positive, negative, or neutral:
"The banana pudding was delicious."
"""

# Call the llm_response function
response = llm_response(prompt)
print("Sentiment:", response)
