import torch
import torch.nn as nn
from attention import MultiHeadAttention, FFN
from utils import PositionalEncoding


class TransformerEncoderBlock(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.mha = MultiHeadAttention(hparams.n_heads, hparams.d_model, hparams.p_attn_dropout)
        self.mha_norm = nn.LayerNorm(hparams.d_model)
        self.ff = FFN(hparams.d_model, hparams.d_ffn_emb, hparams.p_ffn_dropout)
        self.ff_norm = nn.LayerNorm(hparams.d_model)

    def forward(self, x, padding_mask):
        x = self.mha_norm(self.mha(x, mask=padding_mask) + x)
        return self.ff_norm(self.ff(x) + x)


class TransformerEncoder(nn.Module):
    def __init__(self, hparams, sequence_length):
        super().__init__()
        self.positional_encoding = PositionalEncoding(sequence_length, hparams.d_model, hparams.p_pos_enc_dropout, "encoder")
        self.blocks = nn.ModuleList([TransformerEncoderBlock(hparams) for _ in range(hparams.num_encoder_blocks)])

    def forward(self, x, padding_mask):
        """
        x: (batch_size, seq_length, d_model)
        padding_mask: (batch_size, seq_length, seq_length)
        returns: (batch_size, seq_length, d_model)
        """
        x = self.positional_encoding(x)
        for block in self.blocks:
            x = block(x, padding_mask)
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.mha_self = MultiHeadAttention(hparams.n_heads, hparams.d_model, hparams.p_attn_dropout)
        self.mha_self_norm = nn.LayerNorm(hparams.d_model)
        self.mha_cross = MultiHeadAttention(hparams.n_heads, hparams.d_model, hparams.p_attn_dropout)
        self.mha_cross_norm = nn.LayerNorm(hparams.d_model)
        self.ff = FFN(hparams.d_model, hparams.d_ffn_emb, hparams.p_ffn_dropout)
        self.ff_norm = nn.LayerNorm(hparams.d_model)

    def forward(self, x, encoder_output, padding_mask):
        """
        x: (batch, seq_length, d_model)
        encoder_output: (batch, seq_length, d_model)
        padding_mask: (batch, seq_length, seq_length)
        """
        causal_mask = torch.triu(-1e9 * torch.ones(x.shape[1], x.shape[1]), diagonal=0).unsqueeze(0).to(x.device) # (1, seq_length, seq_length)
        x = self.mha_self_norm(self.mha_self(x, mask=causal_mask+padding_mask) + x)
        x = self.mha_cross_norm(self.mha_cross(x, encoder_output, mask=padding_mask) + x)
        return self.ff_norm(self.ff(x) + x)


class TransformerDecoder(nn.Module):
    def __init__(self, hparams, sequence_length, vocab_size):
        super().__init__()
        self.positional_encoding = PositionalEncoding(sequence_length, hparams.d_model, hparams.p_pos_enc_dropout, "decoder")
        self.blocks = nn.ModuleList([TransformerDecoderBlock(hparams) for _ in range(hparams.num_decoder_blocks)])
        # TODO: the weights of this linear projection should be shared with the transformer embedding, 
        # according to the paper.
        self.linear_proj = nn.Linear(hparams.d_model, vocab_size)

    def forward(self, x, encoder_output, padding_mask):
        """
        x: batch_size x seq_length x d_model
        encoder_output: batch_size x seq_length x d_model
        padding_mask: batch_size x seq_length x seq_length

        returns: batch_size x seq_length x vocab_size
        """
        x = self.positional_encoding(x)
        for block in self.blocks:
            x = block(x, encoder_output, padding_mask)
        return torch.softmax(self.linear_proj(x), dim=-1)


class Transformer(nn.Module):
    def __init__(self, hparams, sequence_length, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hparams.d_model)
        self.encoder = TransformerEncoder(hparams, sequence_length)
        self.decoder = TransformerDecoder(hparams, sequence_length, vocab_size)


    def forward(self, encoder_input, decoder_input):
        """
        encoder_input: batch_size x seq_length
        decoder_input: batch_size x seq_length

        returns: batch_size x seq_length x vocab_size
        """
        encoder_padding_mask = torch.where(encoder_input == 0, -1e9, 0).unsqueeze(2) # batch_size x seq_length x 1
        decoder_padding_mask = torch.where(decoder_input == 0, -1e9, 0).unsqueeze(2) # batch_size x seq_length x 1

        mask_for_encoder_output = torch.where(encoder_input == 0, 0, 1).unsqueeze(2) # batch_size x seq_length x 1

        encoder_input_embedding = self.embedding(encoder_input) # batch_size x seq_length x d_model
        decoder_input_embedding = self.embedding(decoder_input) # batch_size x seq_length x d_model

        encoder_output = self.encoder(encoder_input_embedding, encoder_padding_mask)
        return self.decoder(decoder_input_embedding, encoder_output*mask_for_encoder_output, decoder_padding_mask)
