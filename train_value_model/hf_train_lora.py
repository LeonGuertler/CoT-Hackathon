import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, DefaultDataCollator
from peft import get_peft_model, LoraConfig, TaskType

# Initialize WandB
wandb.init(project="your_project_name")

# Load tokenizer and model
model_name = "meta-llama/Llama-3.2-1B"  # Ensure this is the correct model name
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Remove padding token logic since we're using batch size 1
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Define the dataset and tokenization function
dataset = load_dataset("LeonGuertler/PRM800K_train2_updated")
dataset = dataset["train"].train_test_split(test_size=0.01)

def tokenize_function(examples):
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        # max_length=512
    )
    tokenized_inputs["labels"] = examples["value_label"] #[label + 1 for label in examples["value_label"]]
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Use DefaultDataCollator since padding is not needed
data_collator = DefaultDataCollator()

# Configure LoRA
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,  # Updated enum member
    inference_mode=False,
    r=32,  # Rank of the adaptation matrices
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1,  # Dropout probability
    target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
)

# Apply LoRA to the model
model = get_peft_model(model, peft_config)

# Optionally, freeze the base model's parameters to train only LoRA parameters
for name, param in model.base_model.named_parameters():
    if "lora" not in name:
        param.requires_grad = False

# trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
# print("Trainable parameters:")
# for name in trainable_params:
#     print(name)

# Define compute_metrics
from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Define training arguments with mixed precision
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",  # Align save strategy with eval strategy
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=32,  # Simulate batch size of 8
    num_train_epochs=3,
    weight_decay=0.01,
    report_to="wandb",
    logging_dir="./logs",
    logging_steps=1,
    run_name="llama-3.2-1B-mixed-precision-classifier",
    fp16=True,  # Enable mixed precision with fp16
    # For bfloat16, use the following instead:
    # bf16=True,
    # Note: Only set bf16=True if your hardware supports it.
    save_total_limit=2,  # Limit the number of saved checkpoints
    load_best_model_at_end=True,  # Load the best model when finished training
    metric_for_best_model="accuracy",  # Define your metric

    # **Specify Learning Rate**
    learning_rate=1e-5,  # Set your desired learning rate here

    # **Configure Warmup**
    warmup_steps=500,  # Number of warmup steps
    # Alternatively, use warmup_ratio to specify warmup as a fraction of total steps
    # warmup_ratio=0.1,  # 10% of total steps

    # **Other Scheduler Parameters (Optional)**
    # You can also specify other scheduler parameters if needed
    # lr_scheduler_type='linear',  # Type of learning rate scheduler
    # num_warmup_steps=500,  # Explicitly set warmup steps if not using warmup_steps or warmup_ratio
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,  # Ensure this is defined
)

# Fine-tune the model
trainer.train()

# Evaluate the model
trainer.evaluate()
