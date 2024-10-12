import torch
from torch.nn.utils.rnn import pad_sequence
from prepare import prepare_data
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from utils import HDF5Dataset


# hyperparameters
batch_size = 2
gradient_accumulation_steps = 2



half_precision_training = True




# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")


# Retrieve the pad token ID from the tokenizer
pad_token_id = tokenizer.convert_tokens_to_ids("<|reserved_special_token_13|>")


model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to device and enable FP16 precision if CUDA is available
model.to(device)
if device.type == "cuda" and half_precision_training:
    model.half()


# prepare training data
tokenized_data_folder = prepare_data(tokenizer=tokenizer)

# Define a collate function for padding
def collate_fn(batch):

    # Extract token IDs and labels
    token_ids = [torch.tensor(sample["ids"], dtype=torch.long) for sample in batch]
    labels = torch.tensor([sample["label"] for sample in batch], dtype=torch.float32)


    # Pad sequences to the length of the longest sequence in the batch using the pad token ID
    padded_input_ids = pad_sequence(token_ids, batch_first=True, padding_value=pad_token_id)

    # Create attention masks: 1 for real tokens, 0 for padding tokens
    attention_masks = (padded_input_ids != pad_token_id).long()

    return padded_input_ids, attention_masks, labels

# load the dataloader
train_loader = torch.utils.data.DataLoader(
    load_from_disk(tokenized_data_folder)["train"], 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=4,  # Adjust based on your CPU cores
    collate_fn=collate_fn,
    pin_memory=True  # If using GPU
)

val_loader = torch.utils.data.DataLoader(
    load_from_disk(tokenized_data_folder)["val"], 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=4,  # Adjust based on your CPU cores
    collate_fn=collate_fn,
    pin_memory=True  # If using GPU
)

for X, mask, y in train_loader: