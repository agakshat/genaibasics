import torch
import torch.nn as nn
import math


class Attention(nn.Module):
    def __init__(self, d_model, d_key, d_value, p_dropout):
        super().__init__()
        self.Q = nn.Linear(d_model, d_key)
        self.K = nn.Linear(d_model, d_key)
        self.V = nn.Linear(d_model, d_value)
        self.d_key = d_key
        self.attention_dropout = nn.Dropout(p_dropout)

    def forward(self, query_input, keyvalue_input, mask=None):
        """
        query_input: (batch, seq_length, d_model)
        keyvalue_input: (batch, seq_length, d_model)
        mask: (batch, seq_length, seq_length)
        """
        Q = self.Q(query_input)
        K = self.K(keyvalue_input)
        V = self.V(keyvalue_input)
        if mask is None:
            mask = torch.zeros(Q.shape[1], K.shape[1])
        scaled_dot_product = torch.matmul(Q, K.transpose(-1, -2))/math.sqrt(self.d_key)
        return torch.matmul(torch.softmax(self.attention_dropout(scaled_dot_product) + mask, dim=-1), V)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, p_dropout):
        super().__init__()
        # TODO: dont be dumb and create a 4D tensor instead of a list of 3D tensors.
        self.heads = nn.ModuleList([Attention(d_model, d_model//n_heads, d_model//n_heads, p_dropout) for _ in range(n_heads)])

    def forward(self, query_input, keyvalue_input=None, mask=None):
        """
        query_input: (batch, seq_length, d_model)
        keyvalue_input: (batch, seq_length, d_model)
        mask: (batch, seq_length, seq_length)
        """
        if keyvalue_input is None:
            keyvalue_input = query_input
        return torch.cat([head(query_input, keyvalue_input, mask) for head in self.heads], dim=-1)


class FFN(nn.Module):
    def __init__(self, d_model, d_emb, p_dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_emb)
        self.linear2 = nn.Linear(d_emb, d_model)
        self.act = torch.relu
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        return self.dropout(self.linear2(self.act(self.linear1(x))))

