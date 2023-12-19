import torch
import torch.nn as nn

from attention import MultiHeadAttention, FFN
from utils import PositionalEncoding


class ModelConfig:
    p_attn_dropout = 0.1
    p_ffn_dropout = 0.1
    p_pos_enc_dropout = 0.1
    num_blocks = 8
    d_model = 512
    d_ffn_emb = 1024
    n_heads = 4

class GPTDecoderBlock(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.mha_self = MultiHeadAttention(hparams.n_heads, hparams.d_model, hparams.p_attn_dropout)
        self.mha_self_norm = nn.LayerNorm(hparams.d_model)
        self.ff = FFN(hparams.d_model, hparams.d_ffn_emb, hparams.p_ffn_dropout)
        self.ff_norm = nn.LayerNorm(hparams.d_model)

    def forward(self, x):
        """
        x: (batch, seq_length, d_model)
        padding_mask: (batch, seq_length, seq_length)
        """
        causal_mask = torch.triu(-1e9 * torch.ones(x.shape[1], x.shape[1]), diagonal=0).unsqueeze(0).to(x.device) # (1, seq_length, seq_length)
        x = self.mha_self(self.mha_self_norm(x), mask=causal_mask) + x
        return self.ff(self.ff_norm(x)) + x


class GPT(nn.Module):
    """
    The OpenAI GPT architecture does not include a transformer encoder at all, since we are doing a next-token
    prediction task. So we don't want an encoder looking at the entire sentence. We just want a transformer decoder.
    """
    def __init__(self, model_config, sequence_length, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_config.d_model)
        self.positional_encoding = PositionalEncoding(sequence_length, model_config.d_model, model_config.p_pos_enc_dropout, "decoder")
        self.decoder = nn.ModuleList([GPTDecoderBlock(model_config) for _ in range(model_config.num_blocks)])
        self.linear_proj = nn.Linear(model_config.d_model, vocab_size, bias=False)
        self.linear_ln = nn.LayerNorm(model_config.d_model)

    def forward(self, input):
        input = self.embedding(input)
        input = self.positional_encoding(input)
        for block in self.decoder:
            input = block(input)
        logits = self.linear_proj(self.linear_ln(input))
        return logits
