import torch
from pathlib import Path

from src.data.tokenizer import CharTokenizer
from src.model.lightning import MathLightningModule

def generate_answer(model, tokenizer, equation: str, max_len: int = 15):
    """Feeds an equation to the model and decodes the prediction character by character."""
    model.eval()
    device = model.device
    
    # 1. Prepare the Source Sequence
    if not equation.endswith('='):
        equation += '='
        
    src_ids = [tokenizer.sos_token_id] + tokenizer.encode(equation) + [tokenizer.eos_token_id]
    # Transformers expect shape (seq_len, batch_size)
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(1).to(device) 
    
    # 2. Prepare the Target Sequence (Starts with just <sos>)
    tgt_ids = [tokenizer.sos_token_id]
    
    with torch.no_grad():
        for _ in range(max_len):
            tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long).unsqueeze(1).to(device)
            
            # Pass through the model
            logits = model(src_tensor, tgt_tensor) # Shape: (seq_len, batch_size, vocab_size)
            
            # Get the prediction for the VERY LAST token in the sequence
            next_token_logits = logits[-1, 0, :]
            next_token_id = next_token_logits.argmax().item()
            
            # Stop if the model predicts <eos>
            if next_token_id == tokenizer.eos_token_id:
                break
                
            tgt_ids.append(next_token_id)
            
    # 3. Decode and Reverse the string
    # We slice [1:] to remove the <sos> token from our final output
    output_str = tokenizer.decode(tgt_ids[1:])
    reversed_answer = output_str[::-1]
    
    return reversed_answer

if __name__ == "__main__":
    checkpoints_dir = Path("checkpoints")
    
    # Automatically find the latest checkpoint file
    ckpt_files = list(checkpoints_dir.glob("*.ckpt"))
    
    if not ckpt_files:
        print(f"❌ No checkpoints found in {checkpoints_dir.absolute()}")
    else:
        # Just grab the first checkpoint it finds
        ckpt_path = ckpt_files[0]
        tokenizer = CharTokenizer()
        
        # Load model from the PyTorch Lightning checkpoint
        model = MathLightningModule.load_from_checkpoint(
            ckpt_path, 
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.pad_token_id
        )
        
        print(f"✅ Model loaded successfully from {ckpt_path.name}")
        print("Type a math equation (e.g. 124+56=) or 'q' to quit.")
        print("-" * 50)
        
        while True:
            user_input = input(">> ")
            if user_input.lower() == 'q':
                break
            
            try:
                answer = generate_answer(model, tokenizer, user_input)
                print(f"Prediction: {answer}")
            except Exception as e:
                print(f"Error: {e}")