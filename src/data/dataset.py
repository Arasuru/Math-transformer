# divides the equation into two parts, the left and right side of the equation
#Encoder ereceiving the left side of the equation and the decoder receiving the right side of the equation
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class MathDataset(Dataset):
    def __init__(self, filepath, tokenizer):
        self.tokenizer = tokenizer
        self.samples = []

        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        print(f"Loading dataset from {filepath} with {len(lines)} lines.")
    
        for line in lines:
            line = line.strip()
            if '=' in line:
                # Split the equation into left and right parts
                src_str, tgt_str = line.split('=')
                src_str = src_str + '='

                #converting characters to token IDs using the tokenizer
                src_ids = torch.tensor(
                    [self.tokenizer.sos_token_id] + 
                    self.tokenizer.encode(src_str) + 
                    [self.tokenizer.eos_token_id], 
                    dtype=torch.long
                )
                
                # Prepend <sos> and append <eos> to Target (Crucial for decoder!)
                tgt_ids = torch.tensor(
                    [self.tokenizer.sos_token_id] + 
                    self.tokenizer.encode(tgt_str) + 
                    [self.tokenizer.eos_token_id], 
                    dtype=torch.long
                )

                self.samples.append((src_ids, tgt_ids))

        print(f"Successfully processed {len(self.samples)} valid samples.")
        print(f"------------------------------\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def pad_collate_fn(batch, pad_token_id):
    """Pads a batch of sequences to the same length."""
    src_batch, tgt_batch = zip(*batch)
    
    # Pad sequences
    src_padded = pad_sequence(src_batch, batch_first=False, padding_value=pad_token_id)
    tgt_padded = pad_sequence(tgt_batch, batch_first=False, padding_value=pad_token_id)
    
    return src_padded, tgt_padded

                    