from unsloth import FastLanguageModel
import torch

def load_model(model_dir, max_seq_length=4096, dtype=None, load_in_4bit=True):
    """
    Load the trained model and tokenizer from the specified directory.
    
    Args:
        model_dir (str): Path to the directory containing the trained model.
        max_seq_length (int): Maximum sequence length for the model.
        dtype (torch.dtype or None): Data type for model weights.
        load_in_4bit (bool): Whether to load the model in 4-bit precision.
    
    Returns:
        model (FastLanguageModel): The loaded language model.
        tokenizer (Tokenizer): The corresponding tokenizer.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    # Prepare the model for inference
    model = FastLanguageModel.for_inference(model)
    return model, tokenizer

def generate_reply(model, tokenizer, prompt, max_length=4096, temperature=0.6, top_p=0.9):
    """
    Generate a reply for the given prompt using the loaded model.
    
    Args:
        model (FastLanguageModel): The loaded language model.
        tokenizer (Tokenizer): The corresponding tokenizer.
        prompt (str): The input prompt for text generation.
        max_length (int): Maximum length of the generated response.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling probability.
    
    Returns:
        reply (str): The generated text response.
    """
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode the generated tokens to string
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reply

def main():
    # Path to the trained model directory
    model_dir = "outputs/checkpoint-120"
    
    # Load the model and tokenizer
    print("Loading the model...")
    model, tokenizer = load_model(model_dir)
    print("Model loaded successfully.")
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    
    print("\nYou can now enter prompts to generate text. Type 'exit' or 'quit' to terminate.")
    
    while True:
        # Accept user input
        prompt = input("\nEnter your prompt: ")
        
        if prompt.lower() in ["exit", "quit"]:
            print("Exiting the text generator. Goodbye!")
            break
        
        # Generate and display the reply
        print("\nGenerating reply...")
        reply = generate_reply(model, tokenizer, prompt)
        print(f"\nGenerated Reply:\n{reply}")

if __name__ == "__main__":
    main()
