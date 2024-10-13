import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from evaluation_utils import compare_answers
from tqdm import tqdm
import openai
import re 


# Hyperparameters
trace_depth = 5
num_best_traces = 10
num_trace_origins = 2


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
    """
    Use a few-shot version of the GPT-4 API
    """
    def __init__(self):
        # load few-shot prompt
        self.prompt = (
            "For a given math question and reasoning steps, I want you to return the probability that the reasoning step is correct (label=1). Here are some examples of correct, neutral and negative reasoning steps."
            "\nQuestion And Reasoning: The first four terms in an arithmetic sequence are $x+y$, $x-y$, $xy$, and $x/y$, in that order. What is the fifth term? Express your answer as a common fraction. <|reserved_special_token_10|>To find the fifth term, I need to identify the common difference of the arithmetic sequence and add it to the fourth term.<|reserved_special_token_11|>"
            "\nValue: 1"
            "\nQuestion And Reasoning: In triangle $ABC$, medians $AD$ and $CE$ intersect at $P$, $PE=1.5$, $PD=2$, and $DE=2.5$. What is the area of $AEDC$? <|reserved_special_token_10|>I recognize that this problem involves a triangle and its medians, which are the segments that connect each vertex to the midpoint of the opposite side.<|reserved_special_token_11|> <|reserved_special_token_10|>I also know that the three medians of a triangle are concurrent, meaning they all intersect at one point, which is called the centroid.<|reserved_special_token_11|> <|reserved_special_token_10|>The centroid has some interesting properties: it divides each median into a ratio of 2:1, and it is also the center of mass of the triangle.<|reserved_special_token_11|> <|reserved_special_token_10|>This means that if I draw a third median, say $BF$, it will also pass through $P$ and be divided into $BP:PF=2:1$.<|reserved_special_token_11|>"
            "\nValue: 1"
            "\nQuestion And Reasoning: The sides of a triangle have lengths $11, 15,$ and $k,$ where $k$ is a positive integer. For how many values of $k$ is the triangle obtuse? <|reserved_special_token_10|>To solve this problem, I need to recall the relationship between the side lengths and the angles of a triangle, using the law of cosines.<|reserved_special_token_11|> <|reserved_special_token_10|>The law of cosines says that for any triangle with sides $a, b,$ and $c$ and the angle $C$ opposite to $c,$ we have $c^2 = a^2 + b^2 - 2ab \cos C.$<|reserved_special_token_11|> <|reserved_special_token_10|>If the triangle is obtuse, then the angle $C$ is greater than $90^\circ,$ which means that $\cos C$ is negative.<|reserved_special_token_11|> <|reserved_special_token_10|>This implies that $c^2 > a^2 + b^2,$ which is equivalent to $c > \sqrt{a^2 + b^2}.$<|reserved_special_token_11|> <|reserved_special_token_10|>In this problem, I can label the sides $a = 11, b = 15,$ and $c = k.$<|reserved_special_token_11|>"
            "\nValue: -1"
            "\nQuestion And Reasoning: Three points are chosen uniformly at random on a circle. What is the probability that no two of these points form an obtuse triangle with the circle's center? <|reserved_special_token_10|>This is a problem about the angles subtended by arcs of the circle.<|reserved_special_token_11|> <|reserved_special_token_10|>For example, if we choose three points A, B, and C on the circle, then the angle subtended by arc AB at the center is twice the angle subtended by arc AB at any point on the circle.<|reserved_special_token_11|> <|reserved_special_token_10|>Similarly, the angle subtended by arc BC at the center is twice the angle subtended by arc BC at any point on the circle, and so on.<|reserved_special_token_11|>"
            "\nValue: 0"
            "\nQuestion And Reasoning: What is the area of the portion of the circle defined by $x^2-12x+y^2=28$ that lies above the $x$-axis and to the right of the line $y=6-x$? <|reserved_special_token_10|>I recognize that this is a circle with center at $(6,0)$ and radius $r=\sqrt{28+6^2}=10$.<|reserved_special_token_11|>"
            "\nValue: -1"
        )


    def __call__(self, text):
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": f"\nQuestion And Reasoning: {text}\nValue: "}
            ],
            max_tokens=50,
            n=1,
            stop=None,
            temperature=0.7,
        )
        # Extract the assistant's reply
        compl = response.choices[0].message['content']

        # use regex to extract the first float and return it. If no float return 0
         # Use regex to extract the first float from the response
        match = re.search(r"-?\d+\.?\d*", compl)
        if match:
            return float(match.group())
        return 0

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
        max_new_tokens=50,
        eos_token_id=tokenizer.convert_tokens_to_ids(end_of_thought)  # Use eos_token_id for stopping
    )[0]
    # num_return_sequences=4,

    # conver generation to list
    generation = generation.cpu().tolist()

    if generation[-1] != tokenizer.convert_tokens_to_ids(end_of_thought):
        generation.append(tokenizer.convert_tokens_to_ids(end_of_thought))

    # evaluate the current step value
    value = value_model(tokenizer.decode(generation))
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






for q_id, (question, y_true) in enumerate(load_math(split="train")):
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


    # Save reasoning traces to JSON
    reasoning_traces_filepath = os.path.join(data_folder, f"reasoning_traces_{q_id}.json")
    with open(reasoning_traces_filepath, "w") as f:
        json.dump(training_reasoning_traces, f, indent=4)

    # Save training data for confidence model
    conf_model_filepath = os.path.join(data_folder, f"conf_model_data_{q_id}.json")
    with open(conf_model_filepath, "w") as f:
        json.dump(training_data_conf_model, f, indent=4)


    print(f"Completed question {q_id}: Results saved to {data_folder}")
        
