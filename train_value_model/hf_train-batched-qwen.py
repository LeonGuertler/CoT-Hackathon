import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score
import torch, os

# Initialize WandB
# wandb.init(
#     project="COT",
#     name="Value Model: Qwen/Qwen2.5-0.5B"
# )

os.environ["WANDB_PROJECT"] = "COT"

# Load tokenizer and model
model_name = "Qwen/Qwen2.5-0.5B"  # Ensure this is the correct model name
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize the model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

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
tokenizer.pad_token_id = tokenizer.eos_token_id
num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

# Set pad token ID in model config
model.config.pad_token_id = tokenizer.pad_token_id

# Define the dataset and tokenization function
dataset = load_dataset("LeonGuertler/PRM800K_train2_updated")
dataset = dataset["train"].train_test_split(test_size=0.01)

def tokenize_function(examples):
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,          # Optional: Set max_length to control padding
        padding=False            # Let the data collator handle padding
    )
    tokenized_inputs["labels"] = examples["value_label"]  # Assign labels correctly
    return tokenized_inputs

# Apply tokenization and remove unnecessary columns
tokenized_datasets = dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=["text", "value_label"]  # Remove 'text' and original 'value_label' columns
)

# Initialize DataCollatorWithPadding for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define compute_metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    metrics = {"accuracy": accuracy_score(labels, predictions)}
    return metrics

# Define training arguments with mixed precision and warmup
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=25_000,
    save_strategy="steps",  # Align save strategy with eval strategy
    save_steps=100_000,
    per_device_train_batch_size=6,
    per_device_eval_batch_size=6,
    gradient_accumulation_steps=32,  # Simulate batch size of 64
    num_train_epochs=2,
    weight_decay=0.01,
    report_to="wandb",
    logging_dir="./logs",
    logging_steps=1,
    run_name="Qwen-2.5-0.5B-mixed-precision-classifier",
    # fp16=True,  # Enable mixed precision with fp16
    # For bfloat16, use the following instead:
    bf16=True,
    # Note: Only set bf16=True if your hardware supports it.
    save_total_limit=2,  # Limit the number of saved checkpoints
    load_best_model_at_end=True,  # Load the best model when finished training
    metric_for_best_model="accuracy",  # Define your metric

    # **Specify Learning Rate and Warmup**
    learning_rate=1e-5,    # Set your desired learning rate here
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
    data_collator=data_collator,  # Use the dynamic padding collator
    compute_metrics=compute_metrics,  # Ensure this is defined
)

# **Optionally, inspect a batch to verify masking**
# This step is for debugging purposes and is not required for training.
# If you choose to keep it, ensure tensors are on the correct device.

batch = tokenized_datasets["train"][:2]
batch = data_collator(batch)
print("Input IDs:", batch["input_ids"])
print("Attention Mask:", batch["attention_mask"])
print("Labels:", batch["labels"])

# **Move tensors to the same device as the model**
device = model.device  # Get the device of the model (e.g., 'cuda:0' or 'cpu')

# Instead of using torch.tensor (which can cause issues), directly move existing tensors
inputs = {k: v.to(device) for k, v in batch.items()}

# Forward pass
with torch.no_grad():  # Disable gradient calculation for inspection
    outputs = model(**inputs)
print("Model outputs:", outputs)

# Fine-tune the model
trainer.train()

# Evaluate the model
trainer.evaluate()
