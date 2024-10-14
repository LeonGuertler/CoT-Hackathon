from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from torch import nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from unsloth import is_bfloat16_supported

# Define maximum sequence length and other configurations
max_seq_length = 2048
dtype = None
load_in_4bit = True

# Load the base model and tokenizer
base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Apply Parameter-Efficient Fine-Tuning (PEFT) to the base model
peft_model = FastLanguageModel.get_peft_model(
    base_model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        # "embed_tokens", "lm_head"  # Uncomment if needed
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# Define the number of labels for classification
num_labels = 2  # Adjust based on your classification task

# Define the custom model class for sequence classification
class LlamaForSequenceClassification(nn.Module):
    def __init__(self, peft_model, num_labels):
        super().__init__()
        self.peft_model = peft_model  # PEFT-enhanced base model
        self.classifier = nn.Linear(peft_model.config.hidden_size, num_labels)
        self.max_seq_length = peft_model.max_seq_length  # Retain max_seq_length attribute

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Forward pass through the PEFT model
        outputs = self.peft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # Extract hidden states from the last layer
        hidden_states = outputs.hidden_states[-1]  # Shape: (batch_size, seq_len, hidden_size)
        # Use the hidden state of the last token for classification
        pooled_output = hidden_states[:, -1, :]  # Shape: (batch_size, hidden_size)
        logits = self.classifier(pooled_output)  # Shape: (batch_size, num_labels)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}

# Initialize the custom sequence classification model
model = LlamaForSequenceClassification(peft_model, num_labels=num_labels)

# Load the dataset
dataset = load_dataset("LeonGuertler/PRM800K_train2_updated")
train_dataset = dataset["train"]

# Tokenization function including labels
def tokenize_function(examples):
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_seq_length,
        padding=False,  # Let the data collator handle padding
    )
    tokenized_inputs["labels"] = examples["value_label"]  # Assign labels correctly
    return tokenized_inputs

# Tokenize the dataset
tokenized_datasets = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)

# Define compute_metrics function for evaluation
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=32,
    gradient_accumulation_steps=16,
    warmup_steps=200,
    max_steps=10_000,
    learning_rate=5e-5,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    evaluation_strategy="steps",
    save_steps=500,
    eval_steps=500,
)

# Initialize the Trainer with the custom model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.peft_model.save_pretrained("outputs/checkpoint-final/peft_model")
model.classifier.save_pretrained("outputs/checkpoint-final/classifier")
tokenizer.save_pretrained("outputs/checkpoint-final")
