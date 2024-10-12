import os
import torch
from torch.nn.utils.rnn import pad_sequence
from prepare import prepare_data
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from torch.optim import AdamW

# Hyperparameters
batch_size = 2
gradient_accumulation_steps = 2
learning_rate = 5e-10
num_epochs = 10
half_precision_training = True

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# Retrieve the pad token ID from the tokenizer
pad_token_id = tokenizer.convert_tokens_to_ids("<|reserved_special_token_13|>")

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")


# replace the model
class CustomLMHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(
            in_features=2048,
            out_features=1,
            bias=False
        )

        self.activation = torch.nn.Tanh()


    def forward(self, x):
        x = self.linear(x)
        return self.activation(x)

model.lm_head = CustomLMHead()

"""
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
tokenized_data_folder = prepare_data(tokenizer=tokenizer)

# Define a collate function for padding
def collate_fn(batch):
    # Extract token IDs and labels
    token_ids = [torch.tensor(sample["ids"], dtype=torch.long) for sample in batch]
    print(batch)
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
optimizer = AdamW(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    
    for i, (X, mask, y) in enumerate(train_loader):
        X = X.to(device)
        mask = mask.to(device)
        y = y.to(device)
        
        outputs = model(X, attention_mask=mask)
        logits = outputs.logits
        
        # Compute loss
        print(logits[:, -1].view(-1), y)
        loss = criterion(
            logits[:, -1].view(-1),  # only use last token for value prediction
            y
        )


        loss = loss / gradient_accumulation_steps  # Normalize loss for gradient accumulation
        loss.backward()
        
        if (i + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        running_loss += loss.item()
        
        if (i + 1) % (gradient_accumulation_steps * 10) == 0:
            avg_loss = running_loss / (gradient_accumulation_steps * 10)
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], Loss: {avg_loss:.4f}")
            running_loss = 0.0
    
    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_val, mask_val, y_val in val_loader:
            X_val = X_val.to(device)
            mask_val = mask_val.to(device)
            y_val = y_val.to(device)
            
            outputs_val = model(X_val, attention_mask=mask_val)
            logits_val = outputs_val.logits
            loss_val = criterion(logits_val.squeeze(-1), y_val)
            val_loss += loss_val.item()
    
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")
