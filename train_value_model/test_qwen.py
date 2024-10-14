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
    # Load the model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    # # Load the tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}/tokenizer")

    # # Apply the LoRA adapters
    # model = PeftModel.from_pretrained(model, f"{model_dir}/adapter")

    # Prepare the model for inference
    model = FastLanguageModel.for_inference(model)
    return model, tokenizer

def generate_reply(model, tokenizer, prompt, max_new_tokens=512, temperature=0.6, top_p=0.9):
    """
    Generate a reply for the given prompt using the loaded model.
    
    Args:
        model (FastLanguageModel): The loaded language model.
        tokenizer (Tokenizer): The corresponding tokenizer.
        prompt (str): The input prompt for text generation.
        max_new_tokens (int): Maximum number of new tokens to generate.
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
            max_new_tokens=max_new_tokens,  # Use max_new_tokens to avoid warnings
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode the generated tokens to string
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reply

def verify_special_tokens(tokenizer, special_tokens):
    """
    Verify that the special tokens are correctly added to the tokenizer.
    
    Args:
        tokenizer (Tokenizer): The tokenizer to verify.
        special_tokens (list): List of special tokens to check.
    """
    print("\nVerifying special tokens:")
    for token in special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id == tokenizer.unk_token_id:
            print(f"❌ Token '{token}' not found in tokenizer.")
        else:
            print(f"✅ Token '{token}' found with ID {token_id}.")

def main():
    # Path to the trained model directory
    model_dir = "outputs/checkpoint-120/merged_model"  # Ensure this path is correct
    
    # Define the special tokens added during training
    special_tokens = [
        '<|reserved_special_token_10|>', 
        '<|reserved_special_token_11|>', 
        '<|reserved_special_token_12|>', 
        '<|reserved_special_token_13|>'
    ]
    
    # Load the model and tokenizer
    print("Loading the model and tokenizer...")
    model, tokenizer = load_model(model_dir)
    print("Model and tokenizer loaded successfully.")
    
    # Verify special tokens
    verify_special_tokens(tokenizer, special_tokens)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    
    # Create a directory to save generated replies
    os.makedirs("generated_replies", exist_ok=True)
    
    print("\nYou can now enter prompts to generate text. Type 'exit' or 'quit' to terminate.")
    
    while True:
        # Accept user input
        prompt = input("\nEnter your prompt: ")
        
        if prompt.lower() in ["exit", "quit"]:
            print("Exiting the text generator. Goodbye!")
            break
        
        # Generate and display the reply
        print("\nGenerating reply...")
        try:
            reply = generate_reply(model, tokenizer, prompt)
            print(f"\nGenerated Reply:\n{reply}")
            
            # Save the prompt and reply to a file
            with open("generated_replies/log.txt", "a") as f:
                f.write(f"Prompt: {prompt}\nReply: {reply}\n\n")
        except Exception as e:
            print(f"An error occurred during generation: {e}")

if __name__ == "__main__":
    main()
