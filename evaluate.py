"""
Final Evaluation Code
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from eval_utils import compare_answers, load_math
import tqdm



# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to device and enable FP16 precision if CUDA is available
model.to(device)
if device.type == "cuda":
    model.half()

def generate_replies(problems, custom_prompt=None, max_new_tokens=4000, temperature=1.0, top_p=0.95, stop_sequences=None):
    """
    Generate replies/solutions to a batch of math problems.

    Args:
        problems (List[str]): The list of math problems to solve.
        custom_prompt (str, optional): A custom prompt to prepend. Defaults to a standard instruction.
        max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 4000.
        temperature (float, optional): Sampling temperature. Lower values make output more deterministic. Defaults to 1.0.
        top_p (float, optional): Nucleus sampling probability. Defaults to 0.95.
        stop_sequences (List[str], optional): Sequences at which to stop generation. Defaults to None.

    Returns:
        List[str]: A list of generated solutions for each problem.
    """
    # Construct the prompts
    if custom_prompt:
        prompts = [f"{custom_prompt}\nQuestion{problem}\nSolution:" for problem in problems]
    else:
        prompts = [f"Question{problem}\nSolution:" for problem in problems]

    # Tokenize the inputs
    inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).to(device)

    # Generate output in a batched manner
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

    # Decode the generated tokens for each batch
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    solutions = []
    for prompt, generated_text in zip(prompts, generated_texts):
        # Extract the solution part by removing the prompt
        solution = generated_text[len(prompt):].strip()

        # Optional: Truncate at stop sequences if provided
        if stop_sequences:
            for stop_seq in stop_sequences:
                idx = solution.find(stop_seq)
                if idx != -1:
                    solution = solution[:idx].strip()

        solutions.append(solution)

    return solutions

DEBUG=False

def main(num_samples=None, seed=None, custom_prompt=None):
    """
    Main evaluation loop.

    Args:
        num_samples (Optional[int]): Number of samples to evaluate. If None, evaluate all.
        seed (Optional[int]): Seed for reproducibility.
        custom_prompt (str, optional): Custom prompt to use for generation.
    """
    def batch(ds, batch_size=8):
        batch = ([], [])
        for i,e  in enumerate(ds):
            batch[0].append(e[0])
            batch[1].append(e[1])
            if i % batch_size == 0 and i > 0:
                yield batch
                batch = ([], [])


    total, correct = 0, 0
    for i, (problems, solutions) in tqdm.tqdm(enumerate(batch(load_math(num_samples=num_samples, seed=seed)))):
        # Get model answer
        model_answer = generate_replies(problems, custom_prompt=custom_prompt)

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
            if i>10:
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

