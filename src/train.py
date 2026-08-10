from pathlib import Path
from functools import partial
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data.tokenizer import CharTokenizer
from src.data.dataset import MathDataset, pad_collate_fn
from src.model.lightning import MathLightningModule


def main():
    project_root = Path(__file__).parent.parent
    train_path = project_root / "data" / "raw" / "train.txt"
    val_path = project_root / "data" / "raw" / "val.txt"

    # Initialize tokenizer
    tokenizer = CharTokenizer()

    #Dataset and DataLoader
    train_dataset = MathDataset(train_path, tokenizer)
    val_dataset = MathDataset(val_path, tokenizer)

    if len(train_dataset) == 0:
        raise ValueError(f"Dataset is empty! Check if {train_path} has data in it.")
    
    print(f"Loaded {len(train_dataset)} training samples.")
    print(f"Loaded {len(val_dataset)} validation samples.")

    collate_fn = partial(pad_collate_fn, pad_token_id=tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    model = MathLightningModule(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        learning_rate=3e-4
    )

    # Setup WandB logger
    wandb_logger = WandbLogger(project="math-transformer", name="run-dmodel-128", log_model=True)

    # Setup model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="math-transformer-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="train_loss",
        mode="min"
    )

    #trainer configuration
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )

    # Start training
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()