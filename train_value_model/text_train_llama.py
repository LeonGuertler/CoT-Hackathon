from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
max_seq_length = 2048
dtype = None
load_in_4bit = True
# fourbit_models = [
#     "unsloth/Qwen2.5-0.5B-bnb-4bit",
# ]
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B", #"unsloth/Qwen2.5-0.5B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# print(len(tokenizer))
# # adjust tokenizer vocab
# special_tokens_dict = {
#     'additional_special_tokens': [
#       '<|reserved_special_token_10|>', 
#       '<|reserved_special_token_11|>', 
#       '<|reserved_special_token_12|>', 
#       '<|reserved_special_token_13|>'
#     ]
# }
# num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
# print(num_added_tokens, len(tokenizer))
# input()
# # Resize model embeddings to match the new tokenizer size
# model.resize_token_embeddings(len(tokenizer))


model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "embed_tokens", "lm_head"
        ],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)





# Define the dataset and tokenization function
dataset = load_dataset("LeonGuertler/PRM800K_train2_base_sft_updated")
dataset = dataset["train"]#.train_test_split(test_size=0.01)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=4096,          # Adjust as needed
        padding=False            # Let the data collator handle padding
    )

# Apply tokenization and remove unnecessary columns
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]  # Remove 'text' column after tokenization
)


from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 16, #8,
        gradient_accumulation_steps = 16,
        # Use num_train_epochs = 1, warmup_ratio for full training runs!
        warmup_steps = 200,
        max_steps = 2_500,
        learning_rate = 5e-5,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer_stats = trainer.train()


# 10. Save the updated tokenizer and model (including PEFT adapters)
print("Saving the tokenizer and model...")
tokenizer.save_pretrained("outputs/checkpoint-120")
model.save_pretrained("outputs/checkpoint-120")
print("Model and tokenizer saved successfully.")