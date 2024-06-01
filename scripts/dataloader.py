import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
from collections import defaultdict
import random

class DataFrameToDataLoader:
    def __init__(self, df, target_col, batch_size=32, shuffle=True):
        self.df = df
        self.target_col = target_col
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.features_tensor, self.labels_tensor = self._df_to_tensors()
        self.dataloader = self._create_dataloader()

    def _df_to_tensors(self):
        """
        Converts the dataframe to feature and label tensors.
        """
        features = self.df.drop(columns=self.target_col).values
        labels = self.df[self.target_col].values
        
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return features_tensor, labels_tensor

    def _create_dataset(self):
        """
        Creates a TensorDataset from the feature and label tensors.
        """
        return TensorDataset(self.features_tensor, self.labels_tensor)

    def _create_dataloader(self):
        """
        Creates a DataLoader from the TensorDataset.
        """
        dataset = self._create_dataset()
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        return dataloader


# class ScarfDataset:
#     def __init__(self, df, target_col):
#         self.target = df[target_col].values
#         self.features = df.drop(columns=target_col).values


#     def __getitem__(self, index):
#         # the dataset must return a pair of samples: the anchor and a random one from the
#         # dataset that will be used to corrupt the anchor
#         random_idx = np.random.randint(0, len(self))
#         random_sample = torch.tensor(self.features[random_idx], dtype=torch.float)
#         sample = torch.tensor(self.features[index], dtype=torch.float)
#         target = torch.tensor(self.target[index], dtype=torch.float)

#         return sample, random_sample
#     def __len__(self):
#         return len(self.target)

class ScarfDataset(Dataset):
    def __init__(self, df, target_col):
        self.target = torch.tensor(df[target_col].values, dtype=torch.float)
        self.features = torch.tensor(df.drop(columns=[target_col]).values, dtype=torch.float)
        # Create a mapping from each target class to the indices where it appears
        self.indices_by_target = defaultdict(list)
        for idx, target in enumerate(self.target.tolist()):
            self.indices_by_target[target].append(idx)

    def __getitem__(self, index):
        sample = self.features[index]
        target = self.target[index].item()  # Convert tensor to Python scalar

        # Get indices of all samples with the same target, excluding the current sample index
        same_target_indices = self.indices_by_target[target]
        
        # Ensure that there is more than one sample with the same target
        if len(same_target_indices) < 2:
            raise ValueError(f"Not enough samples with target {target} for random sampling.")

        # Choose a random index from the same class, excluding the current sample
        random_idx = index
        while random_idx == index:
            random_idx = random.choice(same_target_indices)

        random_sample = self.features[random_idx]

        return sample, random_sample

    def __len__(self):
        return len(self.target)

class ScarfToDataLoader:
    def __init__(self, df, target_col, batch_size=32, shuffle=True):
        dataset = ScarfDataset(df, target_col)
        self.dataloader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle = shuffle)

class ScarfContrastiveDataset(Dataset):
    def __init__(self, df, target_col):
        self.target = torch.tensor(df[target_col].values, dtype=torch.float32)
        self.features = torch.tensor(df.drop(columns=[target_col]).values, dtype=torch.float32)
        self.indices_by_target = defaultdict(list)
        for idx, target in enumerate(self.target.tolist()):
            self.indices_by_target[target].append(idx)

    def __getitem__(self, index):
        # Get the feature and target for the current index
        anchor = self.features[index]
        target = self.target[index].item()

        # Get indices for all samples of the same class and pick a positive sample
        positive_indices = self.indices_by_target[target]
        positive_index = index
        while positive_index == index:  # Ensure different sample
            positive_index = random.choice(positive_indices)
        positive_sample = self.features[positive_index]

        # Get one negative sample from each of the other classes
        negative_samples = []
        for other_target, indices in self.indices_by_target.items():
            if other_target != target:  # Ensure different class
                negative_index = random.choice(indices)
                negative_samples.append(self.features[negative_index])

        # Convert the list of negative samples to a tensor
        negative_samples = torch.stack(negative_samples)

        return anchor, positive_sample, negative_samples

    def __len__(self):
        return len(self.features)


