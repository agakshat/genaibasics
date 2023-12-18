import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, seq_length, d_input, p_dropout, name):
        """
        d_input: length of input embedding.
        seq_length: length of the sequence.
        """
        super().__init__()
        self.dropout = nn.Dropout(p_dropout)

        pos = torch.arange(0, seq_length).unsqueeze(1) # shape: (seq_length, 1)
        denom = torch.exp(torch.arange(0, d_input, 2) * (-math.log(10000.0) / d_input)) # shape: (d_input/2,)
        
        self.pos_encoding = torch.zeros(1, seq_length, d_input) # shape: (seq_length, d_input)
        self.pos_encoding[0, :, 0::2] = torch.sin(pos * denom)
        self.pos_encoding[0, :, 1::2] = torch.cos(pos * denom)
        
        # register pos_encoding with unique name for multiple invocations.
        self.register_buffer(f'pos_encoding_{name}', self.pos_encoding)

    def forward(self, x):
        """x will be (batch, seq_length, d_input)"""
        x = x + self.pos_encoding.to(x.device)
        return self.dropout(x)


class EarlyStopping:
    def __init__(self, patience, epsilon):
        self.patience = patience
        self.epsilon = epsilon
        self.lowest_loss = torch.inf
        self.counter = 0
        self.early_stop = False
        self.save_model = False

    def __call__(self, val_loss):
        if val_loss < self.lowest_loss - self.epsilon:
            # val_loss decreased, good!
            self.lowest_loss = val_loss
            self.counter = 0
            self.save_model = True
        else:
            # val_loss did not decrease.
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            self.save_model = False
