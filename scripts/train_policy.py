import os
import torch
import argparse
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from scripts.train_dataset_loader import TextWorldTrajectoryDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Train Semantic Policy with LoRA")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--train_data", type=str, default="data/train.jsonl")
    parser.add_argument("--val_data", type=str, default="data/val.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints/semantic_policy")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--use_metadata", action="store_true", help="Include [METADATA MAPPING] in prompts")
    return parser.parse_args()

import time
from transformers import TrainerCallback
class InferenceMetricsCallback(TrainerCallback):
    def __init__(self, val_dataset, tokenizer, num_samples=5):
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        self.num_samples = min(num_samples, len(val_dataset) if val_dataset else 0)
        
    def on_evaluate(self, args, state, control, model, **kwargs):
        if self.num_samples == 0:
            return
            
        print("\n--- Running Inference Metrics on Held-out Set ---")
        model.eval()
        exact_matches = 0
        total_latency = 0
        
        with torch.no_grad():
            for i in range(self.num_samples):
                sample = self.val_dataset[i]
                input_ids = sample["input_ids"].unsqueeze(0).to(model.device)
                labels = sample["labels"].unsqueeze(0)
                
                # Find prompt length by finding where labels != -100
                prompt_len = (labels[0] == -100).sum().item()
                prompt_input_ids = input_ids[:, :prompt_len]
                
                start_time = time.time()
                outputs = model.generate(
                    prompt_input_ids,
                    max_new_tokens=32,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    temperature=0.0
                )
                latency = time.time() - start_time
                total_latency += latency
                
                generated_ids = outputs[0][prompt_len:]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                
                target_ids = labels[0][labels[0] != -100]
                target_text = self.tokenizer.decode(target_ids, skip_special_tokens=True).strip()
                
                if generated_text == target_text:
                    exact_matches += 1
                
                if i < 2:  # Print a couple of samples
                    print(f"Sample {i+1}:")
                    print(f"  Target: '{target_text}'")
                    print(f"  Generated: '{generated_text}'")
                    
        print(f"Exact Match: {exact_matches}/{self.num_samples} ({(exact_matches/self.num_samples)*100:.1f}%)")
        print(f"Mean Response Latency: {total_latency/self.num_samples:.2f}s")
        print("-------------------------------------------------\n")

def main():
    args = parse_args()
    
    # 1. Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 2. Load datasets
    train_dataset = TextWorldTrajectoryDataset(args.train_data, tokenizer, use_metadata=args.use_metadata)
    val_dataset = TextWorldTrajectoryDataset(args.val_data, tokenizer, use_metadata=args.use_metadata)
    
    print(f"Loaded {len(train_dataset)} training trajectories")
    if len(val_dataset) > 0:
        print(f"Loaded {len(val_dataset)} validation trajectories")
    else:
        print("Warning: Validation dataset is empty.")
    
    # 3. Load base model
    print(f"Loading base model {args.model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.config.use_cache = False
    
    # 4. Setup LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 5. Training Arguments
    # Note: Check max_steps instead of epochs for smoke testing if needed
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        evaluation_strategy="steps" if len(val_dataset) > 0 else "no",
        eval_steps=10 if len(val_dataset) > 0 else None,
        save_strategy="steps" if len(val_dataset) > 0 else "no",
        save_steps=10 if len(val_dataset) > 0 else None,
        save_total_limit=3,
        logging_steps=5,
        fp16=False, # Disable fp16 for local Mac compatibility during tiny tests
        report_to="none"
    )
    
    # We use a standard data collator that simply batches the pre-tokenized dicts
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    
    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if len(val_dataset) > 0 else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[InferenceMetricsCallback(val_dataset, tokenizer)] if len(val_dataset) > 0 else None,
    )
    
    # 7. Train
    print("Starting training...")
    trainer.train()
    
    # 8. Save final adapter
    print(f"Saving final model to {args.output_dir}/final...")
    model.save_pretrained(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))
    print("Training complete!")

if __name__ == "__main__":
    main()
