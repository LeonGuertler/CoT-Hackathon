import os
import torch
from torch.nn.utils.rnn import pad_sequence
from prepare import prepare_data
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
from datasets import load_from_disk
from torch.optim import AdamW
import wandb

# ------------------------------------
# Hyperparameters
# ------------------------------------
batch_size = 2  # Adjust as per your GPU memory
gradient_accumulation_steps = 32
start_lr = 1e-8
top_lr = 1e-4
end_lr = 1e-10
num_epochs = 5
freeze_weights = False

beta1 = 0.9
beta2 = 0.95

max_grad_norm = 1.0

# Learning rate (using top_lr as the main learning rate)
learning_rate = top_lr

# ------------------------------------
# Device Configuration
# ------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------
# Model and Tokenizer Setup
# ------------------------------------
model_name = "Qwen/Qwen2.5-0.5B"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float32  # Ensure model uses float32
)

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

# ------------------------------------
# Custom RMSNorm and LM Head
# ------------------------------------
class RMSNorm(torch.nn.Module):
    """
    RMSNorm implementation.
    Reference: https://arxiv.org/abs/1910.07467
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight

class CustomLMHead(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.l = torch.nn.Linear(hidden_size, hidden_size)
        self.reg = RMSNorm(dim=hidden_size)
        self.linear = torch.nn.Linear(hidden_size, 3)  # Assuming 3 classes

    def forward(self, x):
        x = self.l(x)
        x = self.reg(x)
        x = self.linear(x)
        return x

# Replace the model's lm_head with CustomLMHead
hidden_size = model.config.hidden_size
model.lm_head = CustomLMHead(hidden_size)

# Optionally freeze model weights except for the custom head
if freeze_weights:
    for name, param in model.named_parameters():
        if not name.startswith('lm_head'):
            param.requires_grad = False

# ------------------------------------
# Initialize Weights & Biases
# ------------------------------------
wandb.init(
    project="COT",
    name=f"Value Model: {model_name}"
)

# ------------------------------------
# Retrieve Pad Token ID
# ------------------------------------
pad_token_id = tokenizer.convert_tokens_to_ids("<|reserved_special_token_13|>")

# ------------------------------------
# Move Model to Device
# ------------------------------------
model.to(device)
model.train()

# ------------------------------------
# Prepare Training Data
# ------------------------------------
tokenized_data_folder = prepare_data(
    tokenizer=tokenizer,
    model_name=model_name
)

# ------------------------------------
# Define Collate Function
# ------------------------------------
def collate_fn(batch):
    """
    Collate function to pad sequences and create attention masks.
    """
    # Extract token IDs and labels
    token_ids = [torch.tensor(sample["ids"], dtype=torch.long) for sample in batch]
    labels = torch.tensor([sample["label"]+1 for sample in batch], dtype=torch.long)  # Long for CrossEntropyLoss

    # Pad sequences to the length of the longest sequence in the batch using the pad token ID
    padded_input_ids = pad_sequence(token_ids, batch_first=True, padding_value=pad_token_id)

    # Create attention masks: 1 for real tokens, 0 for padding tokens
    attention_masks = (padded_input_ids != pad_token_id).long()

    return padded_input_ids, attention_masks, labels

# ------------------------------------
# Load Datasets
# ------------------------------------
train_dataset = load_from_disk(tokenized_data_folder)["train"]
val_dataset = load_from_disk(tokenized_data_folder)["val"]

# ------------------------------------
# Create DataLoaders
# ------------------------------------
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



class Trainer():
  def __init__(
    train_loader,
    val_loader,
    model,
    learning_rate,
    beta1,
    beta2,
    gradient_accumulation_steps,
    num_epochs
  ):
    self.train_data_iter = iter(train_loader)
    self.val_data_iter = iter(val_loader)
    self.model = model


    self.gradient_accumulation_steps = gradient_accumulation_steps
    self.num_epochs = num_epochs

    # init training objects
    self.optimizer = AdamW(
      self.model.parameters(),
      lr=learning_rate,
      betas=(beta1, beta2)
    )
    self.criterion = torch.nn.CrossEntropyLoss()


    # ------------------------------------
    # Learning Rate Scheduler
    # ------------------------------------
    total_steps = (len(train_loader) // gradient_accumulation_steps) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        self.optimizer,
        num_warmup_steps=int(total_steps * 0.05),
        num_training_steps=total_steps
    )






  def _save_model(self, epoch):
      checkpoint = {
        "model": self.model.state_dict(),
        "optimizer": self.optimizer.state_dict()
      }
      checkpoint_path = f"checkpoints/ckpt_epoch.pt"
      print(f"saving checkpoint to {checkpoint_path}")
      torch.save(checkpoint, checkpoint_path)


def _run_step(self):
    optimizer.zero_grad()
    
    accumulated_loss = 0
    for i in range(gradient_accumulation_steps):
      # get the next batch
      x, y = next(train_data_iter):
      x = x.to(gpu_id if self.gpu_id is not None else self.model.device)
      y = y.to(gpu_id if self.gpu_id is not None else self.model.device)



def run_training_loop(self):
  for epoch in range(self.num_epochs):
    print(f"Starting training epoch: {epoch}")
    start_time = time.time()

    loss = self._run_step()
    end_time = time.time()
    ## print and log the result only on the first GPU after aggregation
    print(f"All GPU(s): step {iter_num}: loss {lossf:.4f}, lr {lr:.1e}, dt {end_time-start_time:.1f}s")

    # save model
    if (self.gpu_id==0 or self.gpu_id==None):
        self._save_model(epoch)


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (X, mask, y) in enumerate(train_loader):
        X = X.to(device)
        mask = mask.to(device)
        y = y.to(device)

        # Forward pass
        outputs = model(X, attention_mask=mask)
        logits = outputs.logits  # Shape: (batch_size, seq_length, num_classes)

        # Get indices of the last non-padded token for each sequence
        lengths = mask.sum(dim=1) - 1  # Shape: (batch_size,)

        # Gather the logits at the last token positions
        last_logits = logits[torch.arange(logits.size(0)), lengths, :]  # Shape: (batch_size, num_classes)

        # Compute loss
        loss = criterion(last_logits, y)

        # Normalize loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        running_loss += loss.item()

        # Backward pass
        loss.backward()

        # Gradient accumulation step
        if (i + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Logging
            avg_loss = running_loss * gradient_accumulation_steps / gradient_accumulation_steps  # Simplified
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {avg_loss:.4f}")
            wandb.log({
                "epoch": epoch + 1,
                "step": i + 1,
                "samples": batch_size * (i + 1),
                "loss": running_loss * gradient_accumulation_steps,  # Multiply back to get actual loss
                "learning_rate": scheduler.get_last_lr()[0],
            })
            running_loss = 0.0

    # ------------------------------------
    # Validation Loop
    # ------------------------------------
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_val, mask_val, y_val in val_loader:
            X_val = X_val.to(device)
            mask_val = mask_val.to(device)
            y_val = y_val.to(device)

            outputs_val = model(X_val, attention_mask=mask_val)
            logits_val = outputs_val.logits

            lengths_val = mask_val.sum(dim=1) - 1
            last_logits_val = logits_val[torch.arange(logits_val.size(0)), lengths_val, :]

            loss_val = criterion(last_logits_val, y_val)
            val_loss += loss_val.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")
    wandb.log({
        "epoch": epoch + 1,
        "validation_loss": avg_val_loss,
    })

    # ------------------------------------
    # Checkpointing
    # ------------------------------------
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_save_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pt")

    torch.save(model.state_dict(), model_save_path)

    print(f"Saved model checkpoint to {model_save_path}")

# ------------------------------------
# Finalize Weights & Biases Run
# ------------------------------------
wandb.finish()
