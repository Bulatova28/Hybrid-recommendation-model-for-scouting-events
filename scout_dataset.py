import torch
from torch.utils.data import Dataset


class ScoutDataset(Dataset):
    def __init__(self, x_prep, y):
        self.x = torch.tensor(x_prep, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype = torch.float32).view(-1,1)
        self.dense_idx = 8
        self.text_idx = 264

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        dense_features = self.x[index, :self.dense_idx]
        text_features = self.x[index, self.dense_idx:self.text_idx]
        sparse_features = self.x[index, self.text_idx:].long()
        target_feature = self.y[index]

        return dense_features, text_features, sparse_features, target_feature


def scout_collate_fn(scout_dataset:list[tuple]):
    transposed_dataset = list(zip(*scout_dataset))
    x_dense = torch.stack(transposed_dataset[0])
    x_text = torch.stack(transposed_dataset[1])
    x_sparse = torch.stack(transposed_dataset[2])
    y_target = torch.stack(transposed_dataset[3])

    batch_size = x_sparse.shape[0]
    features_count = x_sparse.shape[1]
    
    indices = [x_sparse[:,i] for i in range(features_count)]
    offsets = [torch.arange(batch_size) for _ in range(features_count)]

    return x_dense, x_text, offsets, indices, y_target