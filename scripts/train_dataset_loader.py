import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any
from transformers import PreTrainedTokenizer

class TextWorldTrajectoryDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 2048, use_metadata: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_metadata = use_metadata
        self.samples = self._load_data(jsonl_path)
        
    def _load_data(self, jsonl_path: str) -> List[Dict[str, Any]]:
        samples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                step = json.loads(line)
                samples.append(step)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        step = self.samples[idx]
        
        # 1. Prepare Semantic State (User Message)
        # Note: the prompt builder already injected [METADATA MAPPING] optionally if use_metadata=True
        # If use_metadata=False, we strip it out if it exists
        semantic_state = step["semantic_world_state"]
        if not self.use_metadata and "[METADATA MAPPING]" in semantic_state:
            parts = semantic_state.split("[METADATA MAPPING]")
            # The metadata mapping goes until the next section "[MEMORY"
            next_section_idx = parts[1].find("[MEMORY")
            if next_section_idx != -1:
                semantic_state = parts[0] + parts[1][next_section_idx:]
            else:
                semantic_state = parts[0]
                
        optimal_action = step["optimal_action"]
        
        # 2. Convert to ChatML Messages
        messages = [
            {"role": "system", "content": "You are an AI agent reasoning over a semantic world model to play a text adventure game."},
            {"role": "user", "content": semantic_state},
            {"role": "assistant", "content": optimal_action}
        ]
        
        # 3. Tokenize with chat template
        # The prompt is everything up to the assistant's turn
        prompt_text = self.tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        # The full text includes the assistant's reply and EOS
        full_text = prompt_text + optimal_action + self.tokenizer.eos_token
        
        # Tokenize full text
        encodings = self.tokenizer(
            full_text, 
            truncation=True, 
            max_length=self.max_length, 
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encodings["input_ids"][0]
        attention_mask = encodings["attention_mask"][0]
        
        # Create labels (masking out the prompt)
        labels = input_ids.clone()
        
        # Find where the prompt ends
        prompt_encodings = self.tokenizer(prompt_text, truncation=True, max_length=self.max_length, return_tensors="pt")
        prompt_len = len(prompt_encodings["input_ids"][0])
        
        # Mask everything before the assistant response (set to -100 for CrossEntropyLoss)
        labels[:prompt_len] = -100
        
        # Also mask padding tokens
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def get_dataloader(jsonl_path: str, tokenizer: PreTrainedTokenizer, batch_size: int = 2, shuffle: bool = True, use_metadata: bool = True):
    dataset = TextWorldTrajectoryDataset(jsonl_path, tokenizer, use_metadata=use_metadata)
    if len(dataset) == 0:
        return None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("Loading dataset...")
    dataset = TextWorldTrajectoryDataset("data/train.jsonl", tokenizer, max_length=512, use_metadata=True)
    if len(dataset) > 0:
        sample = dataset[0]
        
        input_ids = sample["input_ids"]
        labels = sample["labels"]
        
        # Decode only the unmasked parts to ensure it captures exactly the assistant's action
        valid_labels = labels[labels != -100]
        decoded_action = tokenizer.decode(valid_labels)
        
        print("\n--- Test ---")
        print(f"Original Full Tokens: {len(input_ids)}")
        print(f"Unmasked Tokens (Action): {len(valid_labels)}")
        print(f"Decoded Action: '{decoded_action}'")
        print("Dataset loader masking is correct if it prints the optimal action.")
    else:
        print("Dataset empty.")
