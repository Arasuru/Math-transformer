import random
import os
from pathlib import Path

def generate_equation():
    """Generates a simple math equation and its reversed answer."""
    ops = ['+', '-', '*']
    weights = [0.33, 0.33, 0.34]  
    op = random.choices(ops, weights=weights, k=1)[0]
    
    # Generate numbers heavily weighted towards smaller lengths to help the model learn
    a = random.randint(1, 999)
    b = random.randint(1, 999)
    
    if op == '+':
        ans = a + b
    elif op == '-':
        # Ensure positive results for simplicity in early training
        a, b = max(a, b), min(a, b)
        ans = a - b
    elif op == '*':
        # Keep multiplication numbers smaller to prevent massive token lengths
        a = random.randint(1, 99)
        b = random.randint(1, 99)
        ans = a * b

    # Format the equation
    equation = f"{a}{op}{b}="
    
    # Reverse the answer string! This is the secret sauce for math transformers.
    reversed_ans = str(ans)[::-1]
    
    return f"{equation}{reversed_ans}"

def create_dataset(filepath, num_samples):
    """Creates a text file full of generated equations."""
    # Ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        for _ in range(num_samples):
            f.write(generate_equation() + '\n')

    print(f"Dataset created at {filepath} with {num_samples} samples.")

if __name__ == "__main__":
    curr_dir = Path(__file__).parent
    project_root = curr_dir.parent.parent

    train_path = project_root / "data" / "raw" / "train.txt"
    val_path = project_root / "data" / "raw" / "val.txt"

    # Generate 100,000 samples for training
    create_dataset(train_path, 150000)
    # Generate 5,000 samples for validation
    create_dataset(val_path, 10000)