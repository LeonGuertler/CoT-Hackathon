from vllm import LLM, SamplingParams
from eval_utils import compare_answers, load_math
import tqdm
import time
import prompts as math_prompts

BATCH_SIZE = 4
DEBUG = False
llm = LLM(model="sft_llama3.1-1b-instructoriginal", dtype="bfloat16")
STOP_SEQUENCES = ["<|eot_id|>"]
sampling_params = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=4000)

def generate_replies(problems, stop_sequences=None): 
    prompts = [math_prompts.PROMPT.replace(r"{problem}",problem) for problem in problems]
    outputs = llm.generate(prompts, sampling_params)

    generated_texts = [output.outputs[0].text for output in outputs]
    for output, generated_text in zip(outputs, generated_texts):
        if generated_text == None:
            print(f"UH OH{output.outputs[0]=}")
            print(f"UH OH{generated_text=}")
            generated_text = ""
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
