#!/usr/bin/env python3
import os
import json
import random
import argparse

def split_dataset(input_file: str, out_dir: str, train_ratio: float = 0.8, val_ratio: float = 0.1):
    print(f"Loading trajectories from {input_file}...")
    
    # Group by episode to avoid splitting an episode across train/val/test
    episodes = {}
    with open(input_file, "r") as f:
        for line in f:
            step = json.loads(line)
            ep_id = step["episode_id"]
            if ep_id not in episodes:
                episodes[ep_id] = []
            episodes[ep_id].append(step)
            
    ep_keys = list(episodes.keys())
    random.shuffle(ep_keys)
    
    total = len(ep_keys)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    
    train_keys = ep_keys[:n_train]
    val_keys = ep_keys[n_train:n_train + n_val]
    test_keys = ep_keys[n_train + n_val:]
    
    os.makedirs(out_dir, exist_ok=True)
    
    splits = {
        "train.jsonl": train_keys,
        "val.jsonl": val_keys,
        "test.jsonl": test_keys
    }
    
    for split_name, keys in splits.items():
        out_path = os.path.join(out_dir, split_name)
        count = 0
        with open(out_path, "w") as f:
            for k in keys:
                for step in episodes[k]:
                    f.write(json.dumps(step) + "\n")
                    count += 1
        print(f"Wrote {count} steps across {len(keys)} episodes to {split_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split collected trajectories into train/val/test by episode.")
    parser.add_argument("--input", type=str, default="data/trajectories.jsonl", help="Input trajectories JSONL")
    parser.add_argument("--out-dir", type=str, default="data", help="Output directory")
    args = parser.parse_args()
    split_dataset(args.input, args.out_dir)
