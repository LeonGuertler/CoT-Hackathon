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

from peft import get_peft_model, LoraConfig, TaskType


# Hyperparameters
batch_size = 12 #24
gradient_accumulation_steps = 32
learning_rate = 1e-4
num_epochs = 3
half_precision_training = False
freeze_weights = False

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "meta-llama/Llama-3.2-1B"
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
# class CustomLMHead(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.l = torch.nn.Linear(
#           in_features=896,
#           out_features=896,
#           bias=True
#         )
#         self.linear = torch.nn.Linear(
#             in_features=896,
#             out_features=1,
#             bias=True
#         )

#         self.reg = RMSNorm(dim=896)

#         self.activation = torch.nn.Tanh()


#     def forward(self, x):
#         # input(x)
#         x = self.l(x)
#         x = self.reg(x)
#         x = self.linear(x)
#         return self.activation(x)

class CustomLMHead(torch.nn.Module):
    def __init__(self, hidden_size, num_classes=3):
        super().__init__()
        self.l = torch.nn.Linear(hidden_size, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.l3 = torch.nn.Linear(hidden_size, hidden_size)
        self.reg = RMSNorm(dim=hidden_size)
        self.linear = torch.nn.Linear(hidden_size, num_classes)  # Adjust num_classes as needed

    def forward(self, x):
        # x = self.l(x)
        # x = self.relu(x)
        # x = self.l2(x)
        # x = self.relu(x)
        x = self.l3(x)
        x = self.reg(x)

        x = self.linear(x)
        return x

# Replace the model's lm_head with CustomLMHead
hidden_size = model.config.hidden_size
model.lm_head = CustomLMHead(hidden_size)

# Prepare LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,  # Rank of the LoRA matrices
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]  # Adjust based on model architecture
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Optionally freeze model weights except for the LoRA layers and custom head
if freeze_weights:
    for name, param in model.named_parameters():
        if 'lora' not in name and not name.startswith('lm_head'):
            param.requires_grad = False


# if freeze_weights:
#     # Freeze all model parameters
#     for param in model.parameters():
#         param.requires_grad = False

#     # Unfreeze only the parameters of the new lm_head
#     for param in model.lm_head.parameters():
#         param.requires_grad = True


"""
print(model)
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 2048)
    (layers): ModuleList(
      (0-15): 16 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (k_proj): Linear(in_features=2048, out_features=512, bias=False)
          (v_proj): Linear(in_features=2048, out_features=512, bias=False)
          (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2048, out_features=8192, bias=False)
          (up_proj): Linear(in_features=2048, out_features=8192, bias=False)
          (down_proj): Linear(in_features=8192, out_features=2048, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((2048,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=2048, out_features=128256, bias=False)

  (lm_head): CustomLMHead(
    (linear): Linear(in_features=2048, out_features=1, bias=False)
    (activation): Tanh()
  )
) 
"""



# Move model to device and enable FP16 precision if specified
model.to(device)
if half_precision_training and device.type == "cuda":
    model.half()

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
    labels = torch.tensor([sample["label"]+1 for sample in batch], dtype=torch.float32)

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
optimizer = AdamW(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    mae_tracker = []
    
    
    for i, (X, mask, y) in enumerate(train_loader):
        # input(mask.size())
        # if X.size(1) >= 1024:
        #   # left truncate
        #   X = X[:, -1023:]
        #   mask = mask[:, -1023:]
       
        X = X.to(device)
        mask = mask.to(device)
        y = y.to(device)
        with autocast(dtype=torch.bfloat16):
            # Forward pass
            outputs = model(X, attention_mask=mask)
        logits = outputs.logits  # Shape: (batch_size, seq_length, num_classes)

        # Get indices of the last non-padded token for each sequence
        lengths = mask.sum(dim=1) - 1  # Shape: (batch_size,)

        # Gather the logits at the last token positions
        last_logits = logits[torch.arange(logits.size(0)), lengths, :]  # Shape: (batch_size, num_classes)

        # Compute loss
        loss = criterion(last_logits, y.long())

        # Normalize loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        running_loss += loss.item()

        # Backward pass
        loss.backward()

        if (i + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], Loss: {running_loss:.4f}, y_pred: {logits[:, -1].view(-1).tolist()}, y_true: {y.tolist()}")
            wandb.log(
              {
                "epoch": epoch,
                "step": i,
                "samples": batch_size*i,
                "mse": running_loss,
                "mae": np.mean(mae_tracker)
              }
            )
            running_loss = 0.0
            mae_tracker = []
        
        
    
    # # Validation loop
    # model.eval()
    # val_loss = 0.0
    # with torch.no_grad():
    #     for X_val, mask_val, y_val in val_loader:
    #         X_val = X_val.to(device)
    #         mask_val = mask_val.to(device)
    #         y_val = y_val.to(device)
            
    #         outputs_val = model(X_val, attention_mask=mask_val)
    #         logits_val = outputs_val.logits
    #         loss_val = criterion(logits_val.squeeze(-1), y_val)
    #         val_loss += loss_val.item()
    
    # avg_val_loss = val_loss / len(val_loader)
    # print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")
