import wandb
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments, 
    HfArgumentParser,
    DataCollatorForLanguageModeling
)
import math
import torch
from sklearn.metrics import accuracy_score  # This will no longer be used but kept for reference
import os
from peft import get_peft_model, LoraConfig, TaskType

# Set WandB project
os.environ["WANDB_PROJECT"] = "COT"

# Load tokenizer and model
model_name = "Qwen/Qwen2.5-0.5B"  # Ensure this is the correct model name
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize the model for causal language modeling
model = AutoModelForCausalLM.from_pretrained(model_name)

# Add the special tokens to the tokenizer
special_tokens_dict = {
    'additional_special_tokens': [
        '<|reserved_special_token_10|>',
        '<|reserved_special_token_11|>',
        '<|reserved_special_token_12|>',
        '<|reserved_special_token_13|>',
        '[PAD]'
    ]
}
# Add pad token
tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.pad_token = '[PAD]'
tokenizer.pad_token_id = tokenizer.eos_token_id  # Assuming pad token is same as eos
model.resize_token_embeddings(len(tokenizer))

# Set pad token ID in model config
model.config.pad_token_id = tokenizer.pad_token_id

# Define the dataset and tokenization function
dataset = load_dataset("LeonGuertler/PRM800K_train2_base_sft_updated")
dataset = dataset["train"].train_test_split(test_size=0.01)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,          # Adjust as needed
        padding=False            # Let the data collator handle padding
    )

# Apply tokenization and remove unnecessary columns
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]  # Remove 'text' column after tokenization
)

# Initialize DataCollatorForLanguageModeling for dynamic padding and next token prediction
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Set to False for causal language modeling
)

# Define compute_metrics for perplexity
def compute_metrics(eval_pred):
    loss = eval_pred.metrics["eval_loss"]
    perplexity = math.exp(loss) if loss < 300 else float("inf")  # Avoid overflow
    return {"perplexity": perplexity}


# Define LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # Specify the task type
    inference_mode=False,         # Set to False for training
    r=16,                          # Rank of the LoRA matrices
    lora_alpha=16,                # Scaling factor
    lora_dropout=0,             # Dropout rate for LoRA layers
    target_modules=["q_proj", "v_proj"],  # Modules to apply LoRA to
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
# input(model)
for name, param in model.named_parameters():
    if 'embed_tokens' in name or 'lm_head' in name:
        param.requires_grad = False
    else:
        param.requires_grad = False

# However, LoRA layers should remain trainable
# So, override the requires_grad for LoRA parameters
for name, param in model.named_parameters():
    if "lora" in name:
        param.requires_grad = True





training_args = TrainingArguments(
    output_dir="./Qwen-0.5B",
    
    eval_strategy="steps",
    eval_steps=1,
    save_strategy="steps",
    save_steps=10,
    
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,  # Simulate larger batch size
    num_train_epochs=3,
    weight_decay=0.01,
    report_to="wandb",
    logging_dir="./Qwen-0.5B",
    logging_steps=2,  # Adjust logging frequency as needed
    run_name="Qwen-0.5B-LoRA-LM-sh",
    bf16=True,  # Use bf16 if supported; otherwise, switch to fp16
    # fp16=True,  # Uncomment if bf16 is not supported
    save_total_limit=10,  # Limit the number of saved checkpoints
    load_best_model_at_end=True,  # Load the best model when finished training
    metric_for_best_model="perplexity",  # Define your metric
    greater_is_better=False,  # Lower perplexity is better

    # **Specify Learning Rate and Warmup**
    learning_rate=1e-4,    # LoRA typically benefits from a higher learning rate
    warmup_steps=500,      # Number of warmup steps

    # Optionally specify scheduler type
    lr_scheduler_type='linear',  # Default is 'linear'
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,  # Use the language modeling collator
    compute_metrics=compute_metrics,  # Compute perplexity
)

# # # **Optionally, inspect a batch to verify masking**
# # # This step is for debugging purposes and is not required for training.
# # # If you choose to keep it, ensure tensors are on the correct device.

# dataloader = torch.utils.data.DataLoader(tokenized_datasets["train"], collate_fn=data_collator, batch_size=2)

# for batch in dataloader:
#     print(batch)
#     break

# # **Move tensors to the same device as the model**
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # # Instead of using torch.tensor (which can cause issues), directly move existing tensors
# inputs = {k: v.to(device) for k, v in batch.items()}

# # # Forward pass
# with torch.no_grad():  # Disable gradient calculation for inspection
#     outputs = model(**inputs)
# print("Model outputs:", outputs)

# # Assuming output is of type CausalLMOutputWithPast
# logits = outputs.logits
# print(f'{logits=}')

# # Get the token IDs for the most likely tokens
# predicted_token_ids = logits.argmax(dim=-1)
# print(f'{predicted_token_ids}')

# # Assuming 'tokenizer' is your model's tokenizer
# decoded_text = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)
# print(decoded_text)

# Fine-tune the model
trainer.train()

# Evaluate the model
trainer.evaluate()
