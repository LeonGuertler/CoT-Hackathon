import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from evaluation_utils import compare_answers
from tqdm import tqdm

# Hyperparameters
trace_depth = 25
num_best_traces = 10
num_trace_origins = 10


data_folder = "data/iteration_1"


beginning_of_thought = "<|reserved_special_token_10|>"
end_of_thought = "<|reserved_special_token_11|>"
beginning_of_answer = "<|reserved_special_token_12|>"



def load_math(split="train"):
    dataset = load_dataset("lighteval/MATH", "all", split=split, trust_remote_code=True)
    for sample in dataset:
        yield (sample["problem"], sample["solution"])



# load the value model
class ValueModel:
    def __init__(self):
        pass 
    def __call__(self, text):
        return np.random.uniform(0,1)
value_model = ValueModel()

# load the candidate solution generation model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
if device.type == "cuda":
    model.half()
model.eval()




def generate_next_thought(current_input):
    # append beginning of thought token
    current_input.append(tokenizer.convert_tokens_to_ids(beginning_of_thought))
    
    # Convert the input list to tensor and make sure it is on the correct device (e.g., CUDA)
    input_tensor = torch.tensor([current_input], device="cuda")


    # Generate the next sequence of tokens
    generation = model.generate(
        input_tensor, 
        num_beams=4, 
        do_sample=True, 
        max_new_tokens=100,
        eos_token_id=tokenizer.convert_tokens_to_ids(end_of_thought)  # Use eos_token_id for stopping
    )[0]
    # num_return_sequences=4,

    # evaluate the current step value
    value = value_model(generation)

    # conver generation to list
    generation = generation.cpu().tolist()
    return generation, value



def generate_answer(current_input, y_true):
    # Append the beginning of the answer token
    current_input.append(tokenizer.convert_tokens_to_ids(beginning_of_answer))
    
    # Convert input to tensor and move to GPU
    input_tensor = torch.tensor([current_input]).to("cuda")


    # Generate the next sequence of tokens
    generation = model.generate(
        input_tensor,
        num_beams=4,
        do_sample=True,
        max_new_tokens=100
    )

    generated_tokens = generation.cpu().detach().tolist()
    # Decode the generation to a human-readable string
    generated_text = tokenizer.decode(generated_tokens) #generation[0])
    
    # Compare the generated answer with the true answer
    answer_correct = compare_answers(
        y_pred=generated_text,
        y_true=y_true
    )
    
    return generation, answer_correct


def prune_reasoning_traces(reasoning_traces, pct_prune):
    """
    Prune a certain percentage of reasoning traces based on their values.
    
    Args:
        reasoning_traces (list of tuples): List of tuples where each tuple contains (trace, value).
        pct_prune (float): Percentage of reasoning traces to prune, between 0 and 1.
        
    Returns:
        List of pruned reasoning traces.
    """
    if not 0 <= pct_prune <= 1:
        raise ValueError("pct_prune should be between 0 and 1.")

    # Extract values and sort reasoning_traces by value in descending order
    sorted_traces = sorted(reasoning_traces, key=lambda x: x[1], reverse=True)

    # Calculate the number of traces to retain
    num_to_retain = int(len(sorted_traces) * (1 - pct_prune))

    # Retain the top `num_to_retain` reasoning traces
    pruned_traces = sorted_traces[:num_to_retain]

    return pruned_traces


def get_top_reasoning_traces(reasoning_traces, top_n):
    """
    Extract the top_n highest value reasoning traces
    """
    top_n = np.max([top_n, len(reasoning_traces)])

    sorted_traces = sorted(reasoning_traces, key=lambda x: x[1], reverse=True)

    return reasoning_traces[:top_n]






for q_id (question, y_true) in enumerate(load_math(split="train")):
    training_data_conf_model = {
            "X": [],
            "y": []
        }
    good_reasoning_traces = []
    # tokenize question
    question_ids = tokenizer.encode(question)

    generation_value = []
    generation, value = generate_next_thought(
        current_input=question_ids
    )
    generation_value.append((generation, value))

    for i in range(num_trace_origins):
        for ii in tqdm(range(trace_depth)):
            local_gen_val = []
            for generation, value in tqdm(generation_value):
                new_generation, new_value = generate_next_thought(
                    current_input=generation
                )
                local_gen_val.append(
                    (
                        generation, 
                        value
                    )
                )
                local_gen_val.append(
                    (
                        new_generation, 
                        value*new_value
                    )
                )
            generation_value = local_gen_val

            print(f"Num active traces: {len(generation_value)}")

            prune_reasoning_traces(
                reasoning_traces=generation_value,
                pct_prune=0.3
            )

    # for all generations, append the answer token and generate the answer. Extract the answer accuracy 
    # and the inputs up to the answer token as the dataset for the confidence model
    for generation, value in generation_value:
        full_generation, is_answer_correct = generate_answer(
            current_input=generation,
            y_true=y_true
        )

        training_data_conf_model["X"].append(generation)
        training_data_conf_model["y"].append(int(is_answer_correct))

        if is_answer_correct:
            # append the full trace to the possible reasoning traces
            good_reasoning_traces.append(
                (
                    full_generation,
                    value
                )
            )


    # filter out and append the top-n reasoning traces
    training_reasoning_traces = get_top_reasoning_traces(
        reasoning_traces=good_reasoning_traces, 
        top_n=num_best_traces
    )


    # store both
    with open(os.path.join(data_folder, "reasoning_traces.json"), "w") as f:
        
