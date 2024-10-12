"""
Final Evaluation Code
"""
import torch
import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_utils import compare_answers, load_math


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
    dataset = load_dataset("lighteval/MATH", "all", split=split, trust_remote_code=True)

    if seed is not None:
        torch.manual_seed(seed)

    if num_samples is not None:
        dataset = dataset.shuffle(seed=seed).select(range(num_samples))

    for sample in dataset:
        yield (
            sample["problem"],
            sample["solution"]
        )

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to device and enable FP16 precision if CUDA is available
model.to(device)
# if device.type == "cuda":
#     model.half()



ds = load_dataset("lighteval/MATH", "all", split="train", trust_remote_code=True)


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
            inputs = tokenizer([batch["problem"][j*2 + k] + batch["solution"][j*2 + k] for k in range(2)], return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
            loss = model(**inputs, labels=inputs.input_ids).loss
            loss.backward()
            optimizer.step()
            # scheduler.step()

            if i % 100 == 0:
                print(f"Epoch {epoch}, Iteration {i}, Loss: {loss.item()}")

model.eval()


def generate_reply(problem, custom_prompt=None, max_new_tokens=4000, temperature=1.0, top_p=0.95, stop_sequences=None):
    """
    Generate a reply/solution to a given math problem.

    Args:
        problem (str): The math problem to solve.
        custom_prompt (str, optional): A custom prompt to prepend. Defaults to a standard instruction.
        max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 200.
        temperature (float, optional): Sampling temperature. Lower values make output more deterministic. Defaults to 0.0.
        top_p (float, optional): Nucleus sampling probability. Defaults to 0.95.
        stop_sequences (List[str], optional): Sequences at which to stop generation. Defaults to None.

    Returns:
        str: The generated solution.
    """
    if custom_prompt:
        prompt = f"{custom_prompt}\nQuestion{problem}\nSolution:"
    else:
        prompt = f"Question{problem}\nSolution:\n"

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=(temperature > 0.0 or top_p < 1.0),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=None  # You can implement custom stopping criteria if needed
        )

    # Decode the generated tokens
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the solution part
    solution = generated_text[len(prompt):].strip()

    # Optional: Truncate at stop sequences if provided
    if stop_sequences:
        for stop_seq in stop_sequences:
            idx = solution.find(stop_seq)
            if idx != -1:
                solution = solution[:idx].strip()

    return solution

DEBUG=False

def main(num_samples=None, seed=None, custom_prompt=None):
    """
    Main evaluation loop.

    Args:
        num_samples (Optional[int]): Number of samples to evaluate. If None, evaluate all.
        seed (Optional[int]): Seed for reproducibility.
        custom_prompt (str, optional): Custom prompt to use for generation.
    """

    total, correct = 0, 0
    for i, (problem, solution) in enumerate(load_math(num_samples=num_samples, seed=seed)):
        # Get model answer
        model_answer = generate_reply(problem, custom_prompt=custom_prompt)

        # Evaluate model answer
        is_correct = compare_answers(y_true=solution, y_pred=model_answer)
        correct += is_correct
        total += 1

        # Optional: Print progress
        if (i + 1) % 10 == 0 or (i + 1) == num_samples:
            print(f"Evaluated {i + 1}/{num_samples if num_samples else 'all'} samples. Current Accuracy: {correct/total:.2%}")
        if DEBUG:
            print(f"Problem: {problem}\nModel Answer: {model_answer}\nTrue Answer: {solution}\nCorrect: {is_correct}\n")
            break

    print(f"Final Accuracy: {correct/total:.2%}")

if __name__ == "__main__":

    # create the prompt 
    train_iterator = load_math(split="train")
    prompt = "Please solve the following math questions and make sure to wrap your answer into the $\boxed{answer}$"

    for i, (question, answer) in enumerate(train_iterator):
        prompt += f"Question: {question}\nSolution: {answer}\n"
        if i >= 4:
            break

    main(num_samples=100, seed=42, custom_prompt=prompt)

