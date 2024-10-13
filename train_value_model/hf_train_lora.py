import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, DefaultDataCollator
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

# Initialize WandB
wandb.init(project="your_project_name")

# Load tokenizer and model
model_name = "meta-llama/Llama-3.2-1B"  # Ensure this is the correct model name
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Remove padding token logic since we're using batch size 1
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Define the dataset and tokenization function
dataset = load_dataset("LeonGuertler/PRM800K_train2")
dataset = dataset["train"].train_test_split(test_size=0.01)

def tokenize_function(examples):
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512
    )
    tokenized_inputs["labels"] = [label + 1 for label in examples["value_label"]]
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Use DefaultDataCollator since padding is not needed
data_collator = DefaultDataCollator()

# Configure LoRA
peft_config = LoraConfig(
    task_type=TaskType.SEQUENCE_CLASSIFICATION,
    inference_mode=False,
    r=8,  # Rank of the adaptation matrices
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1,  # Dropout probability
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Modules to apply LoRA
)

# Apply LoRA to the model
model = get_peft_model(model, peft_config)

# Optionally, freeze the base model's parameters to train only LoRA parameters
for param in model.base_model.parameters():
    param.requires_grad = False


# Use DefaultDataCollator since padding is not needed
data_collator = DefaultDataCollator()

# Define training arguments with mixed precision
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
)


# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,  # Add this line
)

# Fine-tune the model
trainer.train()

# Evaluate the model
trainer.evaluate()
