import os
import torch 
import numpy as np 
from tqdm import tqdm 
from datasets import load_dataset



class ValueProcessor:
    """Preprocess the data for training."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer 

    def process(self, sample):
        """Tokenize the input text and extract the float label."""
        ids = self.tokenizer.encode(sample["text"], add_special_tokens=True)
        target = sample["value_label"]  # Ensure label is a float
        return {"ids": ids, "label": target, "len": len(ids)}

    def write_tokenized_data(self, tokenized, hdf5_filename):
        """
        Write the tokenized data and labels to an HDF5 file.
        """
        with h5py.File(hdf5_filename, 'w') as hf:
            for split, dset in tokenized.items():
                grp = hf.create_group(split)
                
                # Create datasets for labels
                labels = np.array(dset["label"], dtype=np.float32)
                grp.create_dataset("labels", data=labels, compression="gzip")

                # Store tokenized IDs as variable-length sequences
                dt = h5py.vlen_dtype(np.dtype('uint16'))
                ids_ds = grp.create_dataset("ids", (len(dset["ids"]),), dtype=dt)

                for i, sample_ids in tqdm(enumerate(dset["ids"]), total=len(dset["ids"]), desc=f"Writing {split}_ids"):
                    ids_ds[i] = sample_ids

    def prepare_hf_dataset(self, tokenized_data_folder, tokenized_dataset):
        """
        Save the tokenized dataset to disk using the datasets library.
        """
        tokenized_dataset.save_to_disk(tokenized_data_folder)


def load_data(dataset_name, shuffle=True):
    """Load the data"""
    dataset = load_dataset(dataset_name, trust_remote_code=True)["train"]


    # Create dataset split
    split_dataset = dataset.train_test_split(
        test_size=0.05, seed=489, shuffle=shuffle
    )

    # Rename test split to val
    split_dataset["val"] = split_dataset.pop("test")

    # Return the training and validation datasets
    return split_dataset

def prepare_data(tokenizer):
    """Prepare and save the dataset using the datasets library."""
    tokenized_data_folder = os.path.join("data", "value_model")
    if os.path.exists(tokenized_data_folder):
        print("Tokenized data already exists")
        return tokenized_data_folder
    else:
        os.makedirs(tokenized_data_folder, exist_ok=True)

    split_dataset = load_data(
        dataset_name="LeonGuertler/PRM800K_train2",
        shuffle=False  # Avoid data leakage
    )

    processor_object = ValueProcessor(tokenizer=tokenizer)

    try:
        # Determine the number of processors
        max_procs = min(os.cpu_count(), 12)  # Cap at 12 to reduce memory usage
        print(f"Using {max_procs} processors")

        # Tokenize the dataset
        tokenized = split_dataset.map(
            processor_object.process,
            remove_columns=["text"],
            desc="Tokenizing dataset",
            num_proc=max_procs
        )

        # Save the tokenized dataset to disk
        processor_object.prepare_hf_dataset(
            tokenized_data_folder=tokenized_data_folder, 
            tokenized_dataset=tokenized
        )

    except Exception as exc:
        print(f"Error: {exc}")
        # Clean up partially written files
        if os.path.exists(tokenized_data_folder):
            import shutil
            shutil.rmtree(tokenized_data_folder)
        raise RuntimeError("Failed to process and write data") from exc

    return tokenized_data_folder

