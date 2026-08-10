import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Injecting information about relative or absolute position of the tokens in squence."""
    def __init__(self, d_model, max_len=100):
        super().__init__()

        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        #X shape: [seq_len, batch_size, d_model]
        seq_len = x.size(0)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(1)
        return x + self.pos_embedding(positions)

class MathTransformer(nn.Module):

    def __init__(self, vocab_size, pad_token_id, d_model=64, nhead=4, num_layers=3, max_len=100):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id

        #Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        #core transformer model
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward= d_model * 4,
            dropout=0.1
        )

        #Final linear layer to project the output to the vocabulary size
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf') so that model cannot look ahead."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt):
        #src/tgt shape: [seq_len, batch_size]

        #padding mask to ignore <pad> tokens in the input sequences
        src_key_padding_mask = (src == self.pad_token_id).transpose(0, 1)
        tgt_key_padding_mask = (tgt == self.pad_token_id).transpose(0, 1)

        #embed the tokens and scale them
        src_embedding = self.token_embedding(src) * math.sqrt(self.d_model)
        tgt_embedding = self.token_embedding(tgt) * math.sqrt(self.d_model)

        #add positional encoding
        src_embedding = self.pos_encoding(src_embedding)
        tgt_embedding = self.pos_encoding(tgt_embedding)

        #generate masks
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)

        #pass through the transformer
        out = self.transformer(
            src_embedding, 
            tgt_embedding, 
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,)

        return self.fc_out(out)