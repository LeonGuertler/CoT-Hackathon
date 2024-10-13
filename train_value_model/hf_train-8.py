import wandb
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DefaultDataCollator,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
)
from sklearn.metrics import accuracy_score
from peft import get_peft_model, LoraConfig, TaskType
import torch

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Initialize WandB
wandb.init(project="your_project_name")

# Load tokenizer
model_name = "meta-llama/Llama-3.2-1B"  # Ensure this is the correct model name
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define BitsAndBytesConfig for 8-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Enable 8-bit quantization
    # Additional parameters can be set here if needed
)

# Define PEFT (LoRA) configuration
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,      # Sequence Classification
    inference_mode=False,            # Set to True if only inference is needed
    r=8,                             # Rank of the LoRA matrices
    lora_alpha=32,                   # Scaling factor
    lora_dropout=0.1,                # Dropout rate
    target_modules=["q_proj", "v_proj"],  # Target specific modules
)

# Load model with quantization
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    quantization_config=bnb_config,  # Use the new quantization configuration
    device_map="auto",                # Automatically map the model to available devices
)

# Apply PEFT (LoRA) to the model
model = get_peft_model(model, peft_config)

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Define the dataset and tokenization function
dataset = load_dataset("LeonGuertler/PRM800K_train2")
dataset = dataset["train"].train_test_split(test_size=0.01)

def tokenize_function(examples):
    tokenized_inputs = tokenizer(
        examples["text"],
    )
    tokenized_inputs["labels"] = [label + 1 for label in examples["value_label"]]
    return tokenized_inputs

tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text", "value_label"],
)

# Use DataCollatorWithPadding for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define training arguments with mixed precision and optimized settings
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",  # Align save strategy with eval strategy
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,  # Simulate batch size of 8
    num_train_epochs=3,
    weight_decay=0.01,
    report_to="wandb",
    logging_dir="./logs",
    logging_steps=10,
    run_name="llama-3.2-1B-mixed-precision-classifier",
    fp16=True,  # Enable mixed precision with fp16
    # For bfloat16, use the following instead:
    # bf16=True,
    # Note: Only set bf16=True if your hardware supports it.
    save_total_limit=2,  # Limit the number of saved checkpoints
    load_best_model_at_end=True,  # Load the best model when finished training
    metric_for_best_model="accuracy",  # Define your metric
    dataloader_num_workers=4,  # Optimize data loading
    optim="adamw_torch",  # Use a memory-efficient optimizer
)

# Verify that some parameters are trainable
trainable_params = [p for p in model.parameters() if p.requires_grad]
print(f"Number of trainable parameters: {len(trainable_params)}")

if not trainable_params:
    raise ValueError("No trainable parameters found. Please check your PEFT configuration.")

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

# Evaluate the model
trainer.evaluate()

# Optionally, save the fine-tuned model
trainer.save_model("./fine-tuned-llama-3.2-1B")
