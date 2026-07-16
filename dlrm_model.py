import torch.nn as nn
import numpy as np
import torch


class ScoutDLRM(nn.Module):
    def __init__(self, embed_dim: int, embed_table_size: np.array, bot_mlp_size: np.array, text_mlp_size: np.array, top_mlp_size: np.array):
        super(ScoutDLRM, self).__init__()
        self.embed_table = nn.ModuleList([nn.EmbeddingBag(n, embed_dim, mode='sum', sparse=False) for n in embed_table_size])

        bottom_layers = []
        for i in range(len(bot_mlp_size)-1):
            bottom_layers.append(nn.Linear(bot_mlp_size[i], bot_mlp_size[i+1]))
            bottom_layers.append(nn.ReLU())
            bottom_layers.append(nn.Dropout(0.2))
        self.bottom_mlp = nn.Sequential(*bottom_layers)

        text_layers = []
        for i in range(len(text_mlp_size)-1):
            text_layers.append(nn.Linear(text_mlp_size[i], text_mlp_size[i+1]))
            text_layers.append(nn.ReLU())
            text_layers.append(nn.Dropout(0.2))
        self.text_mlp = nn.Sequential(*text_layers)

        top_layers = []
        for i in range(len(top_mlp_size)-1):
            top_layers.append(nn.Linear(top_mlp_size[i], top_mlp_size[i+1]))
            if i < len(top_mlp_size)-2:
                top_layers.append(nn.ReLU())
                top_layers.append(nn.Dropout(0.1))
        self.top_mlp = nn.Sequential(*top_layers)
        
        self.register_buffer('rating_values', torch.tensor([1,2,3,4,5], dtype=torch.float32))

    def interaction_layer(self, dense_vector, text_vector, sparse_vectors):
        all_vectors = [dense_vector.unsqueeze(1), text_vector.unsqueeze(1)] + [y.unsqueeze(1) for y in sparse_vectors]
        one_tensor = torch.cat(all_vectors, dim=1)
        
        tensor_dot_prod = torch.bmm(one_tensor, torch.transpose(one_tensor, 1, 2))
        batch_size, n_inter, _ = tensor_dot_prod.shape
        inter_rows, inter_cols = torch.triu_indices(n_inter, n_inter, offset=1)
        tensor_dot_prod_flat = tensor_dot_prod[:, inter_rows, inter_cols]

        return torch.cat([dense_vector, tensor_dot_prod_flat], dim=1)
    
    def forward(self, dense_x, text_x, indices, offsets):
        dense_out = self.bottom_mlp(dense_x)
        text_out = self.text_mlp(text_x)
        sparse_vectors = []
        for c, embed in enumerate(self.embed_table):
            sparse_vectors.append(embed(indices[c], offsets[c]))

        interaction = self.interaction_layer(dense_out, text_out, sparse_vectors)
        logits = self.top_mlp(interaction)
        T = 0.2
        probs = torch.softmax(logits / T, dim=1)
        weighted_avg = torch.sum(probs * self.rating_values, dim=1, keepdim=True)

        return weighted_avg, logits