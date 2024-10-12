
import torch
from torch.utils.data import Dataset

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_filename, split, context_window):
        """
        Initializes the dataset by loading tokenized IDs and labels from HDF5.

        Args:
            hdf5_filename (str): Path to the HDF5 file.
            split (str): The dataset split ('train' or 'val').
            context_window (int): The maximum sequence length.
        """
        self.hdf5_file = h5py.File(hdf5_filename, 'r')
        self.split = split
        self.context_window = context_window

        # Access the datasets
        self.ids = self.hdf5_file[split]["ids"]
        self.labels = self.hdf5_file[split]["labels"]
        self.length = len(self.labels)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Retrieves the x (token IDs) and y (float label) for a given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, float]: (x, y)
        """
        # Retrieve label
        label = float(self.labels[idx])

        # Retrieve token IDs
        sample_ids = self.ids[idx]

        # Truncate or pad the sample_ids to match context_window
        if len(sample_ids) > self.context_window:
            sample_ids = sample_ids[:self.context_window]
        else:
            padding_length = self.context_window - len(sample_ids)
            sample_ids = np.pad(sample_ids, (0, padding_length), 'constant', constant_values=0)

        # Convert to torch tensor
        x = torch.tensor(sample_ids, dtype=torch.long)

        return x, label

    def close(self):
        """Closes the HDF5 file."""
        self.hdf5_file.close()
