import json
import random
from pathlib import Path
from tqdm import tqdm

from src.data.tokenizer import CharTokenizer
from src.model.lightning import MathLightningModule
from src.api.inference import generate_answer


def main():
    print("🚀 Starting Model Evaluation...")

    project_root = Path(__file__).parent.parent
    val_path = project_root / "data" / "raw" / "val.txt"
    checkpoints_dir = project_root / "checkpoints"

    # latest checkpoint
    ckpt_files = list(checkpoints_dir.glob("*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError("No checkpoints found. Train the model first!")
    ckpt_path = ckpt_files[0]

    # Load Model and Tokenizer
    tokenizer = CharTokenizer()
    model = MathLightningModule.load_from_checkpoint(
        ckpt_path, vocab_size=tokenizer.vocab_size, pad_token_id=tokenizer.pad_token_id
    )
    model.eval()

    # Read Validation Data
    with open(val_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if "=" in line]

    # We evaluate on a random subset of 1000 samples so it doesn't take hours
    if len(lines) > 1000:
        lines = random.sample(lines, 1000)

    results = {
        "+": {"correct": 0, "total": 0},
        "-": {"correct": 0, "total": 0},
        "*": {"correct": 0, "total": 0},
    }

    # 5. Run Inference and Score
    print(f"Evaluating {len(lines)} samples...")
    for line in tqdm(lines):
        equation, true_reversed_ans = line.split("=")
        equation += "="
        true_ans = true_reversed_ans[::-1]

        # Determine the operation
        op = "+" if "+" in equation else ("-" if "-" in equation else "*")

        # Predict
        predicted_ans = generate_answer(model, tokenizer, equation)

        results[op]["total"] += 1
        if predicted_ans == true_ans:
            results[op]["correct"] += 1

    # Calculate Final Metrics
    metrics = {"accuracy": {}}
    total_correct = 0
    total_samples = 0

    for op, data in results.items():
        if data["total"] > 0:
            acc = data["correct"] / data["total"]
            metrics["accuracy"][op] = round(acc, 4)
            total_correct += data["correct"]
            total_samples += data["total"]

    metrics["accuracy"]["overall"] = round(total_correct / total_samples, 4)

    # Save to JSON
    metrics_path = project_root / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n✅ Evaluation Complete!")
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    main()
