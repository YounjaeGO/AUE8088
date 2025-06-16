import random
from pathlib import Path

def split_train_val(
    source_file="datasets/kaist-rgbt/train-all-04.txt",
    train_out="datasets/kaist-rgbt/train.txt",
    val_out="datasets/kaist-rgbt/val.txt",
    val_ratio=0.2,
    seed=42,
):
    with open(source_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    random.seed(seed)
    random.shuffle(lines)

    n_val = int(len(lines) * val_ratio)
    val_lines = lines[:n_val]
    train_lines = lines[n_val:]

    with open(train_out, "w") as f:
        f.write("\n".join(train_lines) + "\n")
    with open(val_out, "w") as f:
        f.write("\n".join(val_lines) + "\n")


if __name__ == "__main__":
    split_train_val()