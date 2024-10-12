import os
import torch
from torch.nn.utils.rnn import pad_sequence
from prepare import prepare_data
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from torch.optim import AdamW
import numpy as np
import wandb
from torch.cuda.amp import autocast
from transformers import get_linear_schedule_with_warmup


# Hyperparameters
batch_size = 12 #24
gradient_accumulation_steps = 32
start_lr = 1e-8
top_lr = 1e-4
end_lr = 1e-10
num_epochs = 5
half_precision_training = False
freeze_weights = False

beta1 = 0.9
beta2 = 0.95

max_grad_norm = 1.0



# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "Qwen/Qwen2.5-0.5B"
# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16) #attn_implementation = "flash_attention_2", dtype=torch.bfloat16)

# Add the special tokens to the tokenizer
special_tokens_dict = {
    'additional_special_tokens': [
      '<|reserved_special_token_10|>', 
      '<|reserved_special_token_11|>', 
      '<|reserved_special_token_12|>', 
      '<|reserved_special_token_13|>'
    ]
}
num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)

# Resize model embeddings to match the new tokenizer size
model.resize_token_embeddings(len(tokenizer))

# init bruf
wandb.init(
  project="COT",
  name=f"Value Model: {model_name}"
)


# Retrieve the pad token ID from the tokenizer
pad_token_id = tokenizer.convert_tokens_to_ids("<|reserved_special_token_13|>")

class RMSNorm(torch.nn.Module):
    """
    RMSNorm (https://arxiv.org/abs/1910.07467), implementation from
    https://github.com/meta-llama/llama3/blob/main/llama/model.py
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """Apply RMSNorm"""
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# replace the model
class CustomLMHead(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.l = torch.nn.Linear(hidden_size, hidden_size)
        self.reg = RMSNorm(dim=hidden_size)
        self.linear = torch.nn.Linear(hidden_size, 3)

    def forward(self, x):
        x = self.l(x)
        x = self.reg(x)
        x = self.linear(x)
        return x

# Replace the model's lm_head with CustomLMHead
hidden_size = model.config.hidden_size
model.lm_head = CustomLMHead(hidden_size)




# Move model to device and enable FP16 precision if specified
model.to(device)

# put into train model
model.train()

# Prepare training data
tokenized_data_folder = prepare_data(
  tokenizer=tokenizer,
  model_name=model_name
)

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

# Load the datasets
train_dataset = load_from_disk(tokenized_data_folder)["train"]
val_dataset = load_from_disk(tokenized_data_folder)["val"]

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True if device.type == "cuda" else False,
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True if device.type == "cuda" else False,
)

# Set up optimizer and loss function
optimizer = AdamW(model.parameters(), lr=learning_rate, beta1=beta1, beta2=beta2)
criterion = torch.nn.CrossEntropyLoss()


# LR scheduling
total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.05), num_training_steps=total_steps)

scaler = torch.cuda.amp.GradScaler()


# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    loss.backward()    
    
    for i, (X, mask, y) in enumerate(train_loader):
        X = X.to(device)
        mask = mask.to(device)
        y = y.to(device)
        with autocast():
          outputs = model(X, attention_mask=mask)
        logits = outputs.logits

        # Remove the last dimension if it's of size 1
        logits = logits.squeeze(-1)  # Now shape: (batch_size, sequence_length)

        # logits shape: (batch_size, sequence_length, num_classes)
        lengths = mask.sum(dim=1) - 1  # Indices of last tokens
        last_logits = logits[torch.arange(logits.size(0)), lengths]  # Shape: (batch_size, num_classes)


        # Create batch indices
        batch_indices = torch.arange(X.size(0), device=X.device)

        # Extract the logits corresponding to the last non-padded tokens
        last_logits = logits[batch_indices, last_token_indices]  # Shape: (batch_size,)

        # Compute the loss using the extracted logits
        # If y contains values in {-1, 0, 1}, shift them to {0, 1, 2}
        targets = (y + 1).long()
        loss = criterion(last_logits.float(), targets)



        loss = loss / gradient_accumulation_steps  # Normalize loss for gradient accumulation
        running_loss += loss.item()
        scaler.scale(loss).backward()
        
        if (i + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], Loss: {running_loss:.4f}, y_pred: {logits[:, -1].reshape(-1).tolist()}, y_true: {y.tolist()}")
            wandb.log(
              {
                "epoch": epoch,
                "step": i,
                "samples": batch_size*i,
                "loss": running_loss,
                "learning_rate": scheduler.get_last_lr()[0],
              }
            )
            running_loss = 0.0


    # checkpoint
    # TODO store the model
        
        
    
    # # Validation loop
    # model.eval()
    # val_loss = 0.0
    # with torch.no_grad():
    #     for X_val, mask_val, y_val in val_loader:
    #         X_val = X_val.to(device)
    #         mask_val = mask_val.to(device)
    #         y_val = y_val.to(device)
            
    #         outputs_val = model(X_val, attention_mask=mask_val)
    #         logits = outputs_val.logits
    #         # Calculate the lengths of each sequence (number of non-padded tokens)
    #         lengths = mask.sum(dim=1).to(torch.int64)  # Shape: (batch_size,)

    #         # Subtract 1 to get the indices of the last non-padded tokens
    #         last_token_indices = lengths - 1  # Shape: (batch_size,)

    #         # Create batch indices
    #         batch_indices = torch.arange(X_val.size(0), device=X.device)

    #         # Extract the logits corresponding to the last non-padded tokens
    #         last_logits = logits[batch_indices, last_token_indices]  # Shape: (batch_size,)

    #         # Compute the loss using the extracted logits
    #         loss_val = criterion(last_logits.float(), (y_val+1).long())
    #         val_loss += loss_val.item()
    
    # avg_val_loss = val_loss / len(val_loader)
    # print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")
