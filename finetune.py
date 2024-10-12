"""
Final Evaluation Code
"""
import torch
import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_utils import compare_answers, load_math

from vllm import LLM, SamplingParams
from eval_utils import compare_answers, load_math
import tqdm
import time
import prompts as math_prompts

def load_math(num_samples=None, seed=None, split="test"):
    """
    Load the MATH eval set
    https://huggingface.co/datasets/lighteval/MATH

    Args:
        num_samples (Optional[int]): Number of samples to load. If None, load all.
        seed (Optional[int]): Seed for random sampling.

    Yields:
        Tuple[str, str]: (question, answer)
    """
    dataset = load_dataset("LeonGuertler/PRM800K_train2_base_sft", split=split)

    if seed is not None:
        torch.manual_seed(seed)

    if num_samples is not None:
        dataset = dataset.shuffle(seed=seed).select(range(num_samples))

    for sample in dataset:
        yield (
            sample["test"]
        )

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to device and enable FP16 precision if CUDA is available
model.to(device)
# if device.type == "cuda":
#     model.half()



# ds = load_dataset("lighteval/MATH", "all", split="train", trust_remote_code=True)
ds = load_dataset("LeonGuertler/PRM800K_train2_base_sft", split="train")


# class LoRALinear(nn.Module):
#     def __init__(self, original_layer: nn.Linear, r=8, lora_alpha=32, lora_dropout=0.1):
#         super(LoRALinear, self).__init__()
#         self.original_layer = original_layer  # Store the original nn.Linear layer
#         self.r = r  # Rank of the LoRA approximation
#         self.lora_alpha = lora_alpha  # Scaling factor for LoRA
#         self.scaling = lora_alpha / r  # Scale factor
#         self.lora_dropout = nn.Dropout(p=lora_dropout)  # Dropout for LoRA
        
#         # Low-rank matrices
#         in_features, out_features = original_layer.weight.T.shape
#         self.lora_A = nn.Parameter(torch.randn(in_features, r) * 0.01)
#         self.lora_B = nn.Parameter(torch.randn(r, out_features) * 0.01)

#     def forward(self, x):
#         # Perform the forward pass of the original linear layer
#         original_output = self.original_layer(x)
 
#         # Compute the LoRA output
#         lora_out = self.lora_dropout(x @ self.lora_A) @ self.lora_B
        
#         # Add the LoRA output to the original output, scaled by self.scaling
#         return original_output + lora_out * self.scaling

# use_lora = model_cfg.get("use_lora", False)
# if use_lora:
#     targets = model_cfg.get("lora_targets", [])
#     # Freeze original model parameters except LoRA layers
#     for name, param in model.named_parameters():
#         if "lora" not in name:
#             param.requires_grad = False

#     # Iterate over the model modules to find the target layers and replace them with LoRA layers
#     for target in targets:
#         for name, module in model.named_modules():
#             if target in name and isinstance(module, nn.Linear):
                
#                 # Replace the module with a LoRALayer (wrap the original Linear layer)
#                 lora_layer = LoRALinear(module)
                
#                 # We need to set the new LoRA layer into the model hierarchy
#                 parent_name, attr_name = name.rsplit('.', 1)  # Split the module name to get parent and attribute
#                 parent_module = dict(model.named_modules())[parent_name]  # Access the parent module
                
#                 setattr(parent_module, attr_name, lora_layer)


## TRAIN MODEL
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
loss_fn = torch.nn.CrossEntropyLoss()

## LR Scheduler
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

for epoch in range(1):
    model.train()
    for i, batch in enumerate(tqdm.tqdm(torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True))):
        for j in range(2): # Accumulate gradients
            optimizer.zero_grad()
            inputs = tokenizer([batch["text"][j*2 + k] for k in range(2)], return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
            loss = model(**inputs, labels=inputs.input_ids).loss
            loss.backward()
            optimizer.step()
            # scheduler.step()

            if i % 100 == 0:
                print(f"Epoch {epoch}, Iteration {i}, Loss: {loss.item()}")

model.save_pretrained("sft_llama3.1-1b-instruct")
BATCH_SIZE = 56
DEBUG = False
llm = LLM(model="sft_llama3.1-1b-instruct", dtype="float32")
STOP_SEQUENCES = ["<|eot_id|>"]
sampling_params = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=4000)

def generate_replies(problems, stop_sequences=None): 
    prompts = [math_prompts.PROMPT.replace(r"{problem}",problem) for problem in problems]
    outputs = llm.generate(prompts, sampling_params)

    generated_texts = [output.outputs[0].text for output in outputs]
    for output, generated_text in zip(outputs, generated_texts):
        if generated_text == None:
            print(f"{output.outputs[0]=}")
            print(f"{generated_text=}")
    print(f"{outputs[0].outputs[0]=}")
    print(f"{generated_texts[0]=}")
    solutions=[]
    for prompt, generated_text in zip(prompts, generated_texts):
        # Extract the solution part by removing the prompt
        solution = generated_text.strip()

        # Optional: Truncate at stop sequences if provided
        if stop_sequences:
            for stop_seq in stop_sequences:
                idx = solution.find(stop_seq)
                if idx != -1:
                    solution = solution[:idx].strip()

        solutions.append(solution)
    return solutions

def main(num_samples=None, seed=None):
    """
    Main evaluation loop.

    Args:
        num_samples (Optional[int]): Number of samples to evaluate. If None, evaluate all.
        seed (Optional[int]): Seed for reproducibility.
    """
    def batch(ds, batch_size=BATCH_SIZE):
        batch = ([], [])
        for i,e  in enumerate(ds):
            batch[0].append(e[0])
            batch[1].append(e[1])
            if i % batch_size == 0 and i > 0:
                yield batch
                batch = ([], [])


    total, correct = 0, 0
    for i, (problems, solutions) in enumerate(batch(tqdm.tqdm(load_math(num_samples=num_samples, seed=seed)))):
        # Get model answer
        model_answer = generate_replies(problems,stop_sequences=STOP_SEQUENCES)

        # # Evaluate model answer
        # is_correct = compare_answers(y_true=solution, y_pred=model_answer)
        # correct += is_correct
        # total += 1
        for problem, solution, model_answer in zip(problems, solutions, model_answer):
            is_correct = compare_answers(y_true=solution, y_pred=model_answer)
            correct += is_correct
            total += 1

        # Optional: Print progress
        if (i + 1) % 10 == 0 or (i + 1) == num_samples:
            print(f"Evaluated {i + 1}/{num_samples if num_samples else 'all'} samples. Current Accuracy: {correct/total:.2%}")
        if DEBUG:
            print(f"Problem: {problem}\nModel Answer: {model_answer}\nTrue Answer: {solution}\nCorrect: {is_correct}\n")
            if i>1:
                break

    print(f"Final Accuracy: {correct/total:.2%}")

if __name__ == "__main__":

    # create the prompt 
    train_iterator = load_math(split="train")
    start = time.time()

    main(num_samples=1000, seed=42)
