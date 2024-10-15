import datasets
import torch
import transformers
import os
import logging

from datasets import load_dataset

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments, 
    HfArgumentParser,
    DataCollatorForLanguageModeling
)
from accelerate.state import PartialState

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    log_level = logging.INFO
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    os.environ["WANDB_PROJECT"] = "COT"
    
    # Load args
    # Define training arguments with mixed precision and warmup
    training_args = TrainingArguments(
        output_dir="/data/shanghong/llama-3.2-1B-sft",
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        per_device_train_batch_size=3,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=32,  # Simulate larger batch size
        num_train_epochs=3,
        weight_decay=0.01,
        report_to="wandb",
        logging_dir="/data/shanghong/llama-3.2-1B-sft",
        logging_steps=2,  # Adjust logging frequency as needed
        run_name="llama-3.2-1b-sft",
        bf16=True,  
        save_total_limit=2,  # Limit the number of saved checkpoints
        load_best_model_at_end=True,  # Load the best model when finished training
        metric_for_best_model="perplexity",  # Define your metric
        greater_is_better=False,  # Lower perplexity is better
        learning_rate=1e-6,    # Set your desired learning rate here
        warmup_steps=500,      # Number of warmup steps
        lr_scheduler_type='linear',  # Default is 'linear'
        batch_eval_metrics=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant":False}
    )
    
    model_kwargs = dict(
        revision="main",
        torch_dtype="bfloat16",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        use_cache=False if training_args.gradient_checkpointing else True,
    )

    # Load tokenizer and model
    logger.info("*** Loading pretrained model and tokenizer ***")
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens_dict = {
        'additional_special_tokens': [
            '<|reserved_special_token_10|>',
            '<|reserved_special_token_11|>',
            '<|reserved_special_token_12|>',
            '<|reserved_special_token_13|>',
            '[PAD]'
        ]
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.pad_token = '[PAD]'
    tokenizer.pad_token_id = tokenizer.eos_token_id  # Assuming pad token is same as eos
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id # Set pad token ID in model config

    # Define the dataset and tokenization function
    dataset = load_dataset("LeonGuertler/PRM800K_train2_base_sft")
    dataset = dataset["train"].train_test_split(test_size=0.01)
    logger.info(
            f"Training on the following datasets and their proportions: {[split + ' : ' + str(dset.num_rows) for split, dset in dataset.items()]}"
        )
    
    # Apply tokenization and remove unnecessary columns
    logger.info("*** Tokenizing datasets ***")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,          # Adjust as needed
            padding=False            # Let the data collator handle padding
        )

    with PartialState().local_main_process_first():
        train_dataset = dataset['train'].map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]  # Remove 'text' column after tokenization
        )
        test_dataset = dataset['test'].map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]  # Remove 'text' column after tokenization
        )

    # Initialize DataCollatorForLanguageModeling for dynamic padding and next token prediction
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Set to False for causal language modeling
    )

    entropy_list = []
    def batch_compute_metrics(eval_pred, compute_result):
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
        if compute_result:
             # Concatenate all entropy values and compute the mean
            all_entropy = torch.cat(entropy_list, dim=0)
            mean_entropy = torch.mean(all_entropy, dim=-1)
            trainer.accelerator.print(mean_entropy)
            perplexity = torch.exp(mean_entropy)
            # empty cache
            entropy_list = []
            return {"perplexity": perplexity.item()}
        else:
            return {}

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,  # Use the language modeling collator
        compute_metrics=batch_compute_metrics,  # Compute perplexity
    )

    # Fine-tune the model
    logger.info("*** Training ***")
    trainer.train()

    # Evaluate the model
    torch.cuda.empty_cache()
    logger.info("*** Evaluating ***")

    trainer.evaluate()
