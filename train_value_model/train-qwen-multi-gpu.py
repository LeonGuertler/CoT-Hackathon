import os
import time
import torch
import torch.distributed as dist
from prepare import prepare_data
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
from datasets import load_from_disk
import wandb
from contextlib import nullcontext
from omegaconf import OmegaConf

import torch
from torch.distributed import destroy_process_group
import torch.multiprocessing as mp
from torch.distributed import init_process_group

# ------------------------------------
# Custom RMSNorm and LM Head
# ------------------------------------
class RMSNorm(torch.nn.Module):
    """
    RMSNorm implementation.
    Reference: https://arxiv.org/abs/1910.07467
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight

class CustomLMHead(torch.nn.Module):
    def __init__(self, hidden_size, num_classes=3):
        super().__init__()
        self.l = torch.nn.Linear(hidden_size, hidden_size)
        self.reg = RMSNorm(dim=hidden_size)
        self.linear = torch.nn.Linear(hidden_size, num_classes)  # Adjust num_classes as needed

    def forward(self, x):
        x = self.l(x)
        x = self.reg(x)
        x = self.linear(x)
        return x

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    # Get the master address and port from SLURM environment variables
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "12355")

    # Set the environment variables for PyTorch distributed
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


# ------------------------------------
# Trainer Class
# ------------------------------------
class Trainer:
    def __init__(
        self,
        model,
        tokenizer,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        device,
        gradient_accumulation_steps,
        max_grad_norm,
        num_epochs=3,
        checkpoint_dir="checkpoints",
        use_wandb=True,
        wandb_project="COT",
        wandb_run_name="Value Model",
        gpu_id=None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps // torch.cuda.device_count() if torch.cuda.is_available() else gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.use_wandb = use_wandb
        self.gpu_id = gpu_id

        print(f"GPU_id: {self.gpu_id}")
        self.scaler = None
        self.ctx = self._setup_ctx()

        if gpu_id is not None: # using ddp
            self.dist = True
            self.DDP_model = DDP(self.model, device_ids=[gpu_id])
        else:
            self.dist = False
            self.DDP_model = model

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if self.use_wandb and (self.gpu_id==0 or not self.dist):
            wandb.init(
                project=wandb_project,
                name=wandb_run_name
            )

        self.model.to(self.device)
        if torch.cuda.device_count() > 1:
            self.model = DDP(self.model, device_ids=[self.gpu_id]) if gpu_id is not None else DDP(self.model)
    def _setup_ctx(self):
        """Get the context manager"""
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        self._setup_scaler(dtype)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
        return ctx
    def _setup_scaler(self, dtype=torch.float16):
        """Setup the scaler"""
        # self.scaler = torch.cuda.amp.GradScaler(enabled=dtype == torch.float16)
        self.scaler = torch.amp.GradScaler(self.model.device, enabled=dtype == torch.float16)


    def _save_model(self, epoch):
        checkpoint = {
            "model_state_dict": self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": epoch
        }
        checkpoint_path = os.path.join(self.checkpoint_dir, f"ckpt_epoch_{epoch}.pt")
        print(f"Saving checkpoint to {checkpoint_path}")
        torch.save(checkpoint, checkpoint_path)

    def _validate(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, mask_val, y_val in self.val_loader:
                X_val = X_val.to(self.device)
                mask_val = mask_val.to(self.device)
                y_val = y_val.to(self.device)

                outputs_val = self.model(X_val, attention_mask=mask_val)
                logits_val = outputs_val.logits

                lengths_val = mask_val.sum(dim=1) - 1
                last_logits_val = logits_val[torch.arange(logits_val.size(0)), lengths_val, :]

                loss_val = self.criterion(last_logits_val, y_val)
                val_loss += loss_val.item()

        avg_val_loss = val_loss / len(self.val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        if self.use_wandb:
            wandb.log({"Validation Loss": avg_val_loss})

        self.model.train()
        return avg_val_loss

    def run_training_loop(self):
        for epoch in range(1, self.num_epochs + 1):
            print(f"Starting epoch {epoch}/{self.num_epochs}")
            epoch_start_time = time.time()
            running_loss = 0.0

            for i, (X, mask, y) in enumerate(self.train_loader, 1):
                start_time = time.time()
                X = X.to(self.device)
                mask = mask.to(self.device)
                y = y.to(self.device)

                if self.dist and hasattr(self.DDP_model, 'no_sync'):
                    context_manager = self.DDP_model.no_sync() if i != self.gradient_accumulation_steps else nullcontext()
                else:
                    context_manager = nullcontext()

                with context_manager:
                    with self.ctx:
                        outputs = self.model(X, attention_mask=mask)
                        logits = outputs.logits  # Shape: (batch_size, seq_length, num_classes)

                        lengths = mask.sum(dim=1) - 1  # Shape: (batch_size,)
                        last_logits = logits[torch.arange(logits.size(0)), lengths, :]  # Shape: (batch_size, num_classes)

                        loss = self.criterion(last_logits, y)


                # Scale loss to simulate larger effective batch size
                loss = loss / self.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
                # loss.backward()
                running_loss += loss.item()

                if i % self.gradient_accumulation_steps == 0:

                    # Unscale the gradients of the optimizer's assigned params in-place
                    self.scaler.unscale_(self.optimizer)
                    # Clip the gradients with normalization
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.scaler.step(self.optimizer)
                    self.scheduler.step()
                    self.scaler.update()
                    self.optimizer.zero_grad()


                    avg_loss = running_loss * self.gradient_accumulation_steps / self.gradient_accumulation_steps
                    current_lr = self.scheduler.get_last_lr()[0]
                    end_time = time.time()
                    print(f"Epoch [{epoch}/{self.num_epochs}], Step [{i}/{len(self.train_loader)}], Loss: {avg_loss:.4f}, LR: {current_lr:.1e}, dt: {end_time-start_time}")
                    start_time = time.time()
                    if self.use_wandb and (self.gpu_id==0 or not self.dist):
                        wandb.log({
                            "Epoch": epoch,
                            "Step": i,
                            "Loss": avg_loss, #running_loss * self.gradient_accumulation_steps* (torch.cuda.device_count() if torch.cuda.is_available() else 1),
                            "Samples": i* self.gradient_accumulation_steps * (torch.cuda.device_count() if torch.cuda.is_available() else 1),
                            "Learning Rate": current_lr,
                        })
                    running_loss = 0.0

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            print(f"Epoch {epoch} completed in {epoch_duration:.2f}s")

            # Validation
            # self._validate()

            # Checkpointing
            self._save_model(epoch)

        if self.use_wandb and (self.gpu_id==0 or not self.dist):
            wandb.finish()
# ------------------------------------
# Define Collate Function
# ------------------------------------
def collate_fn(batch):
    """
    Collate function to pad sequences and create attention masks.
    """
    # Extract token IDs and labels
    token_ids = [torch.tensor(sample["ids"], dtype=torch.long) for sample in batch]
    labels = torch.tensor([sample["label"] + 1 for sample in batch], dtype=torch.long)  # Long for CrossEntropyLoss

    pad_token_id = 151668
    # Pad sequences to the length of the longest sequence in the batch using the pad token ID
    padded_input_ids = pad_sequence(token_ids, batch_first=True, padding_value=pad_token_id)

    # Create attention masks: 1 for real tokens, 0 for padding tokens
    attention_masks = (padded_input_ids != pad_token_id).long()

    return padded_input_ids, attention_masks, labels
# ------------------------------------
# Main Training Script
# ------------------------------------
def main():
    # ------------------------------------
    # Hyperparameters
    # ------------------------------------
    batch_size = 8  # Adjust as per your GPU memory
    gradient_accumulation_steps = 32
    start_lr = 1e-8
    top_lr = 1e-4
    end_lr = 1e-10
    num_epochs = 5
    freeze_weights = False

    beta1 = 0.9
    beta2 = 0.95

    max_grad_norm = 1.0

    # Learning rate (using top_lr as the main learning rate)
    learning_rate = top_lr

    model_name = "Qwen/Qwen2.5-0.5B"

    # ------------------------------------
    # Device Configuration
    # ------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, tokenizer = build_model(model_name)
    # ------------------------------------
    # Retrieve Pad Token ID
    # ------------------------------------
    pad_token_id = tokenizer.convert_tokens_to_ids("<|reserved_special_token_13|>")
    # ------------------------------------
    # Prepare Training Data
    # ------------------------------------
    # Assuming 'prepare_data' is a function that tokenizes and saves the dataset to disk
    tokenized_data_folder = prepare_data(
        tokenizer=tokenizer,
        model_name=model_name
    )

    

    # ------------------------------------
    # Load Datasets
    # ------------------------------------
    dataset = load_from_disk(tokenized_data_folder)
    train_dataset = dataset["train"]
    val_dataset = dataset["val"]

    # ------------------------------------
    # Create DataLoaders
    # ------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True if device.type == "cuda" else False,
    )

    # ------------------------------------
    # Initialize Optimizer and Scheduler
    # ------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2)
    )

    total_steps = (len(train_loader) // gradient_accumulation_steps) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.05),
        num_training_steps=total_steps
    )

    # ------------------------------------
    # Define Loss Function
    # ------------------------------------
    criterion = torch.nn.CrossEntropyLoss()

    world_size = torch.cuda.device_count()

    if world_size <= 1:
      # single GPU/CPU training
      build_single_gpu_training(
          model_name=model_name,
          model=model,
          tokenizer=tokenizer,
          train_loader=train_loader, 
          val_loader=val_loader, 
          optimizer=optimizer, 
          scheduler=scheduler,
          criterion=criterion,
          gradient_accumulation_steps=gradient_accumulation_steps,
          max_grad_norm=max_grad_norm,
      )

    else:
      # multi-GPU training
      mp.spawn(
        build_multi_gpu_training,
        args=(
            world_size,
            model_name,
            model,
            tokenizer,
            train_loader, 
            val_loader, 
            optimizer, 
            scheduler,
            criterion,
            gradient_accumulation_steps,
            max_grad_norm
        ),
        nprocs=world_size,
        join=True
      )

      # Additional cleanup to prevent leaked semaphores
      for process in mp.active_children():
        process.terminate()
        process.join()



    # # ------------------------------------
    # # Initialize Trainer
    # # ------------------------------------
    # trainer = Trainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     optimizer=optimizer,
    #     scheduler=scheduler,
    #     criterion=criterion,
    #     device=device,
    #     gradient_accumulation_steps=gradient_accumulation_steps,
    #     max_grad_norm=max_grad_norm,
    #     num_epochs=num_epochs,
    #     checkpoint_dir="checkpoints",
    #     use_wandb=True,
    #     wandb_project="COT",
    #     wandb_run_name=f"Value Model: {model_name}",
    #     gpu_id=0 if torch.cuda.is_available() else None
    # )

    # # ------------------------------------
    # # Start Training
    # # ------------------------------------
    # trainer.run_training_loop()


def init_print_overried():
    '''
    Overriding the print function is useful when running DDP. 
    This way, only rank 0 prints to the console.
    '''
    import builtins as __builtin__
    
    original_print = __builtin__.print

    def print(*args, **kwargs):
        if os.getenv('GLOBAL_RANK') == '0':
            original_print(*args, **kwargs)

    __builtin__.print = print

    return original_print

def restore_print_override(original_print):
    '''
    Restore the original print function.
    '''
    import builtins as __builtin__
    __builtin__.print = original_print

def build_model(model_name):
    # ------------------------------------
    # Model and Tokenizer Setup
    # ------------------------------------
    

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16  # Ensure model uses float32
    )

    # Add the special tokens to the tokenizer
    special_tokens_dict = {
        'additional_special_tokens': [
            '<|reserved_special_token_10|>',
            '<|reserved_special_token_11|>',
            '<|reserved_special_token_12|>',
            '<|reserved_special_token_13|>'
        ]
    }
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)

    # Resize model embeddings to match the new tokenizer size
    model.resize_token_embeddings(len(tokenizer))

    # ------------------------------------
    # Replace the model's lm_head with CustomLMHead
    # ------------------------------------
    hidden_size = model.config.hidden_size
    model.lm_head = CustomLMHead(hidden_size)

    # # Optionally freeze model weights except for the custom head
    # if freeze_weights:
    #     for name, param in model.named_parameters():
    #         if not name.startswith('lm_head'):
    #             param.requires_grad = False

    return model, tokenizer


def build_single_gpu_training(
      model_name,
      model,
      tokenizer,
      train_loader, 
      val_loader, 
      optimizer, 
      scheduler,
      criterion,
      gradient_accumulation_steps,
      max_grad_norm,
  ):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # model, tokenizer = build_model()
  model.to(device)
  model.train()
  print("Model build")
  trainer = Trainer(
      model=model,
      tokenizer=tokenizer,
      train_loader=train_loader,
      val_loader=val_loader,
      optimizer=optimizer,
      scheduler=scheduler,
      criterion=criterion,
      device=device,
      gradient_accumulation_steps=gradient_accumulation_steps,
      max_grad_norm=max_grad_norm,
      checkpoint_dir="checkpoints",
      use_wandb=True,
      wandb_project="COT",
      wandb_run_name=f"Value Model: {model_name}",
      gpu_id=None #0 if torch.cuda.is_available() else None
  )

  trainer.run_training_loop()

def build_multi_gpu_training(
      rank, 
      world_size,
      model_name,
      model,
      tokenizer,
      train_loader, 
      val_loader, 
      optimizer, 
      scheduler,
      criterion,
      gradient_accumulation_steps,
      max_grad_norm,
  ):
    os.environ["GLOBAL_RANK"] = str(rank)
    original_print = init_print_overried()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    try:
      print("Rank: ", rank, "World Size: ", world_size)
      ddp_setup(rank=rank, world_size=world_size)
      # model, tokenizer = build_model()
      model.to(device)
      model.train()
      print(f"Rank {rank} Model built")

      trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        checkpoint_dir="checkpoints",
        use_wandb=True,
        wandb_project="COT",
        wandb_run_name=f"Value Model: {model_name}",
        gpu_id=rank #0 if torch.cuda.is_available() else None
      )
      print(f"Rank {rank} Trainer built")
      trainer.run_training_loop()

    finally:
      # clean up
      destroy_process_group()

      # restore the print function
      restore_print_override(original_print)



if __name__ == "__main__":
    main()
