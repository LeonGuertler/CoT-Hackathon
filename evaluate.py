"""
Final Evaluation Code
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def load_math(num_samples=None, seed=None):
    """
    Load the MATH eval set
    https://huggingface.co/datasets/lighteval/MATH

    Args:
        num_samples (Optional[int]): Number of samples to load. If None, load all.
        seed (Optional[int]): Seed for random sampling.

    Yields:
        Tuple[str, str]: (question, answer)
    """
    dataset = load_dataset("lighteval/MATH", "all", split="test", trust_remote_code=True)

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
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to device and enable FP16 precision if CUDA is available
model.to(device)
if device.type == "cuda":
    model.half()

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
        prompt = f"{custom_prompt}\n\n{problem}\n\nSolution:"
    else:
        prompt = f"Solve the following math problem:\n\n{problem}\n\nSolution:"

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

def compare_answers(y_true, y_pred):
    """
    Compare the true answer with the predicted answer.

    This is a placeholder implementation. Depending on the evaluation criteria,
    you might want to implement a more sophisticated comparison, possibly involving
    symbolic computation or exact string matching.

    Args:
        y_true (str): The ground truth solution.
        y_pred (str): The model-generated solution.

    Returns:
        int: 1 if correct, 0 otherwise.
    """
    # Simple exact match (can be improved)
    return int(y_true.strip() == y_pred.strip())

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

        print(solution)
        input(model_answer)
        # Evaluate model answer
        is_correct = compare_answers(y_true=solution, y_pred=model_answer)
        correct += is_correct
        total += 1

        # Optional: Print progress
        if (i + 1) % 10 == 0 or (i + 1) == num_samples:
            print(f"Evaluated {i + 1}/{num_samples if num_samples else 'all'} samples. Current Accuracy: {correct/total:.2%}")

    print(f"Final Accuracy: {correct/total:.2%}")

if __name__ == "__main__":
    main(num_samples=100, seed=42, custom_prompt="You are a highly intelligent math assistant.")

