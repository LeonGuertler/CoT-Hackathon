import torch 
import json 
from datasets import Dataset as HFDataset

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path="../data/phase2_train.jsonl"):
        super().__init__()
        self.path = path


        self.beginning_of_thought = "<|reserved_special_token_10|>" #"<b_o_t>"
        self.end_of_thought = "<|reserved_special_token_11|>" #"<e_o_t>"
        self.beginning_of_answer = "<|reserved_special_token_12|>" #"<b_o_a>"


        self._load_data()

    def _wrap_in_thought_token(self, text):
        return f"\n{self.beginning_of_thought}{text}{self.end_of_thought}\n"

    def _wrap_in_answer_token(self, text):
        return f"\n{self.beginning_of_answer}{text}"

    def _load_data(self):
        with open(self.path, "r") as f:
            data = list(f) #json.load(f)
        # c=0
        self.X, self.y = [], []
        for sub_dict_str in data:
            # conver to json 
            sub_dict = json.loads(sub_dict_str)

            # extract all X and y
            current_x_stem = sub_dict["question"]["problem"] 

            for step in sub_dict["label"]["steps"]:
                # iterate over completions
                for completion_dict in step["completions"]:
                    if completion_dict["rating"] is None:
                        continue
                        # c += 1
                        # print(c)
                    # else:
                    #     print("not none")
                    #     input(step)
                    self.X.append(
                        current_x_stem + self._wrap_in_thought_token(completion_dict["text"])
                    )
                    self.y.append(
                        completion_dict["rating"]
                    )

                if step["chosen_completion"] is None:
                    current_x_stem += self._wrap_in_thought_token(step["human_completion"])
                else:
                    current_x_stem += self._wrap_in_thought_token(step["completions"][step["chosen_completion"]]["text"])

                # here would be a good place to insert the answer and conf-pred dataset


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def to_hf_dataset(self):
        """Converts the custom dataset to a Hugging Face Dataset."""
        data = {"text": self.X, "value_label": self.y}
        hf_dataset = HFDataset.from_dict(data)
        return hf_dataset



class Dataset2(torch.utils.data.Dataset):
    def __init__(self, path="../data/phase2_train.jsonl"):
        super().__init__()
        self.path = path


        self.beginning_of_thought = "<|reserved_special_token_10|>" #"<b_o_t>"
        self.end_of_thought = "<|reserved_special_token_11|>" #"<e_o_t>"
        self.beginning_of_answer = "<|reserved_special_token_12|>" #"<b_o_a>"
        self.LABEL_MAPPING = {
            -1:0,
            0:1,
            1:1
        }

        self._load_data()


        

    def _wrap_in_thought_token(self, text):
        return f"\n{self.beginning_of_thought}{text}{self.end_of_thought}\n"

    def _wrap_in_answer_token(self, text):
        return f"\n{self.beginning_of_answer}{text}"

    def _load_data(self):
        with open(self.path, "r") as f:
            data = list(f) #json.load(f)
        # c=0
        self.X, self.y = [], []
        for sub_dict_str in data:
            # conver to json 
            sub_dict = json.loads(sub_dict_str)

            # extract all X and y
            # current_x_stem = sub_dict["question"]["problem"] 
            if sub_dict["question"]["ground_truth_solution"] is None or sub_dict["question"]["ground_truth_answer"] is None:
                continue
            try:
                solution_prefix = sub_dict["question"]["ground_truth_solution"].split(sub_dict["question"]["ground_truth_answer"])[0].replace("\\boxed{", "")
            except Exception as e:
                print(e)
                continue
            for step in sub_dict["label"]["steps"]:
                # iterate over completions
                for completion_dict in step["completions"]:
                    if completion_dict["rating"] is None:
                        continue


                    self.X.append(solution_prefix)
                    self.y.append(self.LABEL_MAPPING[completion_dict["rating"]])



    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def to_hf_dataset(self):
        """Converts the custom dataset to a Hugging Face Dataset."""
        data = {"text": self.X, "value_label": self.y}
        hf_dataset = HFDataset.from_dict(data)
        return hf_dataset



class BaseSFTDataset(Dataset):
    def __init__(self, path="../data/phase2_train.jsonl"):
        super().__init__(path)


    def _load_data(self):
        with open(self.path, "r") as f:
            data = list(f) #json.load(f)

        self.X = []
        for sub_dict_str in data:
            # conver to json 
            sub_dict = json.loads(sub_dict_str)

            # extract all X and y
            current_x_stem = sub_dict["question"]["problem"] 

            for step in sub_dict["label"]["steps"]:
                if step["chosen_completion"] is None:
                    # append the answer
                    current_x_stem += self._wrap_in_thought_token(step["human_completion"])
                else:
                    current_x_stem += self._wrap_in_thought_token(step["completions"][step["chosen_completion"]]["text"])

                # here would be a good place to insert the answer and conf-pred dataset
                current_x_stem += self._wrap_in_answer_token(sub_dict["question"]["ground_truth_answer"]) 
            self.X.append(current_x_stem)

    def __getitem__(self, idx):
        return self.X[idx]

    def to_hf_dataset(self):
        """Converts the custom dataset to a Hugging Face Dataset."""
        data = {"text": self.X}
        hf_dataset = HFDataset.from_dict(data)
        return hf_dataset





if __name__ == "__main__":
    # test 
    # d = Dataset()
    # input(len(d))

    # for X,y in d:
    #     print(X)
    #     input(y)



    d = BaseSFTDataset()
    input(len(d))

    for X in d:
        input(X)


