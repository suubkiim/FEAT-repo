import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .dilated_conv import DilatedConvEncoder
from .mlp import MLPEncoder


def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)

    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)

    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            res[i, t:t+l] = False
    return res

def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


class TSDecoder(nn.Module):
    def __init__(self, rep_dims, output_dims):
        super().__init__()
        self.linear_1 = nn.Linear(rep_dims, int(output_dims * 2))
        self.linear_2 = nn.Linear(int(output_dims * 2), output_dims)
        self.predictor = nn.Sequential(self.linear_1, nn.GELU(), self.linear_2)

    def forward(self, x):
        predict = self.predictor(x)
        return predict


class TSEncoder(nn.Module):
    def __init__(self, context_l, input_dims, output_dims, hidden_dims=64, depth=10, temporal_mask_mode='binomial', feature_mask_mode='binomial'):
        super().__init__()
        # hyper-parameter
        self.context_l = context_l # seq length
        self.input_dims = input_dims # feature dim
        self.output_dims = output_dims # rep dim
        self.hidden_dims = hidden_dims
        self.temporal_mask_mode = temporal_mask_mode
        self.feature_mask_mode = feature_mask_mode

        # input embedding
        self.temporal_embedding = nn.Linear(input_dims, hidden_dims)
        self.feature_embedding = nn.Linear(context_l, hidden_dims)

        # temporal encoder
        self.temporal_rep_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        self.temporal_rep_dropout = nn.Dropout(p=0.1)

        # feature encoder
        self.feature_rep_extractor = MLPEncoder(hidden_dims, expansion_factor=2, output_dims=output_dims, dropout=0.1)
        self.feature_rep_dropout = nn.Dropout(p=0.1)

        # learnable gating
        self.gate = nn.Linear(output_dims * 2, 2)

        # feature to temporal mapping
        self.feature_to_temporal = nn.Linear(self.input_dims, self.context_l)

        # reconstruction decoder
        self.decoder = TSDecoder(output_dims, input_dims)


    def forward(self, x, mask=None):  # x: B x T x input_dims

        # impute nan values with 0
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0

        # temporal embed
        temporal_emb = self.temporal_embedding(x)  # B x T x Ch

        # generate & apply mask
        if mask is None:
            if self.training:
                temporal_mask = self.temporal_mask_mode
            else:
                temporal_mask = 'all_true'
        else:
            temporal_mask = mask

        # temporal mask
        if temporal_mask == 'binomial':
            temporal_mask = generate_binomial_mask(temporal_emb.size(0), temporal_emb.size(1)).to(temporal_emb.device)
        elif temporal_mask == 'continuous':
            temporal_mask = generate_continuous_mask(temporal_emb.size(0), temporal_emb.size(1)).to(temporal_emb.device)
        elif temporal_mask == 'all_true':
            temporal_mask = x.new_full((temporal_emb.size(0), temporal_emb.size(1)), True, dtype=torch.bool)
        elif temporal_mask == 'all_false':
            temporal_mask = x.new_full((temporal_emb.size(0), temporal_emb.size(1)), False, dtype=torch.bool)
        elif temporal_mask == 'mask_last':
            temporal_mask = x.new_full((temporal_emb.size(0), temporal_emb.size(1)), True, dtype=torch.bool)
            temporal_mask[:, -1] = False

        temporal_mask &= nan_mask
        temporal_emb[~temporal_mask] = 0

        # feature embed
        feature_emb = self.feature_embedding(x.transpose(1, 2)) # B X F X Ch

        # generate & apply mask
        if mask is None:
            if self.training:
                feature_mask = self.feature_mask_mode
            else:
                feature_mask = 'all_true'
        else:
            feature_mask = 'all_true'

        # feature mask
        if feature_mask == 'binomial':
            feature_mask = generate_binomial_mask(feature_emb.size(0), feature_emb.size(1)).to(feature_emb.device)
        elif feature_mask == 'continuous':
            feature_mask = generate_continuous_mask(feature_emb.size(0), feature_emb.size(1)).to(feature_emb.device)
        elif feature_mask == 'all_true':
            feature_mask = feature_emb.new_full((feature_emb.size(0), feature_emb.size(1)), True, dtype=torch.bool)
        elif feature_mask == 'all_false':
            feature_mask = feature_emb.new_full((feature_emb.size(0), feature_emb.size(1)), False, dtype=torch.bool)

        feature_emb[~feature_mask] = 0

        # dilated conv encoder
        temporal_emb = temporal_emb.transpose(1, 2)  # B x Ch x T
        temporal_rep = self.temporal_rep_dropout(self.temporal_rep_extractor(temporal_emb))  # B x Co x T
        temporal_rep = temporal_rep.transpose(1, 2)  # B x T x Co

        # mlp encoder
        feature_rep = self.feature_rep_extractor(feature_emb) # B X F X Co

        # feature to temporal mapping
        feature_to_temporal_rep = self.feature_rep_dropout(self.feature_to_temporal(feature_rep.transpose(1,2)).transpose(1,2)) # B, T, Co

        # timestamp-wise gating parameter
        gate = F.softmax(self.gate(torch.cat([temporal_rep, feature_to_temporal_rep], dim=-1)), dim=-1) # B, T, 2

        # aggregate representations by gating paramters
        reps = (temporal_rep * gate[:,:, 0:1]) + (feature_to_temporal_rep * gate[:,:,1:2]) # B, T, Co

        # reconstruction
        recon_x = self.decoder(reps) # B, T, F

        return reps, temporal_rep, feature_rep, recon_x
