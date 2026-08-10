import pytorch_lightning as pl
import torch
import torch.nn as nn
from src.model.architecture import MathTransformer

class MathLightningModule(pl.LightningModule):
    def __init__(self, vocab_size, pad_token_id, learning_rate=1e-4):
        super().__init__()
        # Save hyperparams for easy tracking in W&B
        self.save_hyperparameters()
        self.pad_token_id = pad_token_id
        
        # Instantiate our underlying PyTorch model
        self.model = MathTransformer(
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            d_model=128,
            nhead=4,
            num_layers=3
        )
        
        # We ignore the <pad> token when calculating loss so it doesn't skew results
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)

    def forward(self, src, tgt):
        return self.model(src, tgt)

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        
        # In seq2seq, the input to the decoder is all tokens except the last,
        # and the target it tries to predict is all tokens except the first.
        tgt_input = tgt[:-1, :]
        tgt_expected = tgt[1:, :]
        
        # Get model predictions
        logits = self(src, tgt_input)
        
        # Reshape for CrossEntropyLoss
        logits = logits.view(-1, logits.shape[-1])
        tgt_expected = tgt_expected.reshape(-1)
        
        # Calculate loss
        loss = self.criterion(logits, tgt_expected)
        
        # Log to W&B / TensorBoard
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # AdamW is generally the best optimizer for Transformer models
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer