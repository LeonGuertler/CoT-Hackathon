# check_tokenizer_size.py
from unsloth import FastLanguageModel

def count_tokens(model_dir):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )
    return len(tokenizer)

if __name__ == "__main__":
    model_dir = "outputs/checkpoint-120"  # Ensure this path is correct
    token_count = count_tokens(model_dir)
    print(f"Tokenizer Vocabulary Size: {token_count}")



"""exit()



from unsloth import FastLanguageModel

def count_tokens(model_dir):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )
    return len(tokenizer)

model_dir = "outputs/checkpoint-10"  # Replace with your actual checkpoint path
token_count = count_tokens(model_dir)
print(f"Tokenizer Vocabulary Size: {token_count}")


import torch
from unsloth import FastLanguageModel

def get_embedding_size(model_dir):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )
    embedding = model.base_model.model.model.embed_tokens.weight
    return embedding.shape

model_dir = "outputs/checkpoint-10"  # Replace with your actual checkpoint path
embedding_size = get_embedding_size(model_dir)
print(f"Model Embedding Matrix Size: {embedding_size}")
"""