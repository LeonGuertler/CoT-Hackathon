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

# # Trainer args
# parser = HfArgumentParser(TrainingArguments)
# training_args = parser.parse_json_file(json_file="training_configs/trainer_config.json")

os.environ["WANDB_PROJECT"] = "COT"

# Load tokenizer and model
model_name = "meta-llama/Llama-3.2-1B-Instruct"  # Ensure this is the correct model name
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
dataset = load_dataset("LeonGuertler/PRM800K_train2_base_sft")
dataset = dataset["train"].train_test_split(test_size=0.01)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        # max_length=512,          # Adjust as needed
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
# def compute_metrics(eval_pred):
#     input(eval_pred)
#     loss = eval_pred.metrics["eval_loss"]
#     perplexity = math.exp(loss) if loss < 300 else float("inf")  # Avoid overflow
#     return {"perplexity": perplexity}

entropy_list = []
def batch_compute_metrics(eval_pred):
    input(eval_pred)
    global entropy_list
    IGNORE_INDEX = -100
    shift_logits = eval_pred.predictions[..., :-1, :].contiguous()
    shift_labels = eval_pred.label_ids[..., 1:].contiguous()
    batch_size, seq_length, vocab_size = shift_logits.shape
    trainer.accelerator.print(f"batch_size, seq_length: {batch_size,seq_length}")
    # Flatten the tokens
    entropy = torch.nn.functional.cross_entropy(
        shift_logits.view(batch_size * seq_length, vocab_size), 
        shift_labels.view(batch_size * seq_length), 
        reduction='none'
    )
    # Append the flattened entropy for this batch
    entropy_list.append(entropy[torch.where(shift_labels.view(batch_size * seq_length) != IGNORE_INDEX)].cpu())
    # Concatenate all entropy values and compute the mean
    all_entropy = torch.cat(entropy_list, dim=0)
    mean_entropy = torch.mean(all_entropy, dim=-1)
    trainer.accelerator.print(mean_entropy)
    perplexity = torch.exp(mean_entropy)
    # empty cache
    entropy_list = []
    return {"perplexity": perplexity.item()}

# Define training arguments with mixed precision and warmup
training_args = TrainingArguments(
    output_dir="/data/shanghong/llama-3.2-1B",
    
    eval_strategy="epoch",
    save_strategy="epoch",

    per_device_train_batch_size=3,
    per_device_eval_batch_size=1,
    eval_accumulation_steps=1,
    gradient_accumulation_steps=64,  # Simulate larger batch size
    num_train_epochs=3,
    weight_decay=0.01,
    report_to="wandb",
    logging_dir="/data/shanghong/llama-3.2-1B",
    logging_steps=2,  # Adjust logging frequency as needed
    run_name="llama-3.2-1B-mixed-precision-LM-sh",
    fp16=True,  # Enable mixed precision with fp16
    # For bfloat16, use the following instead:
    # bf16=True,
    # Note: Only set bf16=True if your hardware supports it.
    save_total_limit=2,  # Limit the number of saved checkpoints
    load_best_model_at_end=True,  # Load the best model when finished training
    metric_for_best_model="perplexity",  # Define your metric
    greater_is_better=False,  # Lower perplexity is better
    # **Specify Learning Rate and Warmup**
    learning_rate=1e-6,    # Set your desired learning rate here
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
    compute_metrics=batch_compute_metrics,  # Compute perplexity
)

# Fine-tune the model
trainer.train()

# Evaluate the model
trainer.evaluate()
