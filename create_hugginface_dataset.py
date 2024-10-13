from utils import Dataset, BaseSFTDataset
from datasets import Dataset as HFDataset
from huggingface_hub import HfApi

# Assuming you have hf_dataset from previous steps
def push_to_hub(hf_dataset, repo_name, repo_description=""):
    """
    Pushes a Hugging Face Dataset to the Hub.

    Args:
        hf_dataset (HFDataset): The dataset to push.
        repo_name (str): The name of the repository on Hugging Face Hub.
        repo_description (str): A short description of the dataset.
    """
    # Initialize the API
    api = HfApi()

    try:
        hf_dataset.push_to_hub(repo_name)
    except:
        # does not yet esits
        api.create_repo(
            repo_id=repo_name,
            repo_type="dataset",
            # description=repo_description,
            private=False,  # Set to True if you want the dataset to be private
        )
        hf_dataset.push_to_hub(repo_name)

    # Push the dataset
    print(f"Dataset pushed to https://huggingface.co/datasets/{repo_name}")


custom_dataset = Dataset()
hf_dataset = custom_dataset.to_hf_dataset()
print(hf_dataset)

push_to_hub(
        hf_dataset, 
        repo_name="PRM800K_train2",  # Replace 'username' with your Hugging Face username
        repo_description="Phase 2 training dataset with questions and ratings."
    )


custom_dataset = BaseSFTDataset()
hf_dataset = custom_dataset.to_hf_dataset()
print(hf_dataset)

push_to_hub(
        hf_dataset, 
        repo_name="PRM800K_train2_base_sft",  # Replace 'username' with your Hugging Face username
        repo_description="Phase 2 training dataset with questions and ratings."
    )
