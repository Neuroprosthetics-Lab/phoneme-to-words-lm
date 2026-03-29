#!/usr/bin/env python3
"""Fine-tune a causal LLM with LoRA on domain-specific text for BCI rescoring."""

import argparse
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig


def load_sentences(file_path: str) -> List[str]:
    """Load sentences from a text file, one per line. Skips blank lines and comments."""
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            sentences.append(line)
    return sentences


def preprocess_sentences(sentences: List[str]) -> List[str]:
    """Lowercase, remove punctuation, and strip whitespace to match decoder output format.

    Mirrors the normalization applied by the decoder in kenlm_flashlight_text_lm.py:
    remove_punctuation + replace_words (from phoneme_to_words_lm.utils).
    """
    from phoneme_to_words_lm.utils import remove_punctuation, replace_words

    processed = []
    for s in sentences:
        s = replace_words(remove_punctuation(s))
        s = ' '.join(s.split())
        if s:
            processed.append(s)
    return processed


def prepare_dataset(
    source_files: dict,
    upsample_factors: dict,
    val_fraction: float = 0.05,
    seed: int = 42,
) -> Tuple[List[str], List[str], dict, dict]:
    """Load, preprocess, deduplicate, upsample, and split sentences from multiple sources.

    Args:
        source_files: Mapping of source name -> file path (e.g. {"personal": "path.txt"}).
        upsample_factors: Mapping of source name -> integer upsample factor.
        val_fraction: Fraction of each source to hold out for validation.
        seed: Random seed for reproducible splitting.

    Returns:
        (train_sentences, val_sentences, stats, val_by_source) where stats has
        per-source counts and val_by_source maps source name -> val sentence list.
    """
    if not source_files:
        raise ValueError("source_files must contain at least one source")

    import random
    rng = random.Random(seed)

    all_train = []
    all_val = []
    val_by_source = {}
    stats = {}

    for source_name, file_path in source_files.items():
        raw = load_sentences(file_path)
        processed = preprocess_sentences(raw)

        # Deduplicate (before upsampling)
        unique = list(dict.fromkeys(processed))

        # Stratified split
        rng.shuffle(unique)
        n_val = max(1, int(len(unique) * val_fraction))
        val_part = unique[:n_val]
        train_part = unique[n_val:]

        # Upsample train split only
        factor = upsample_factors.get(source_name, 1)
        train_part = train_part * factor

        all_train.extend(train_part)
        all_val.extend(val_part)
        val_by_source[source_name] = val_part

        stats[source_name] = {
            'raw': len(raw),
            'unique': len(unique),
            'train': len(train_part),
            'val': len(val_part),
            'upsample_factor': factor,
        }

    # Shuffle the combined train set
    rng.shuffle(all_train)

    return all_train, all_val, stats, val_by_source


@torch.no_grad()
def compute_perplexity(
    model,
    tokenizer,
    sentences: List[str],
    batch_size: int = 16,
    max_length: int = 512,
    device=None,
    desc: str = "Computing Perplexity",
) -> float:
    """Compute perplexity of the model on a list of sentences."""
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    total_loss = 0.0
    total_tokens = 0

    for i in tqdm(range(0, len(sentences), batch_size), desc=desc):
        batch = sentences[i:i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        labels = inputs.input_ids.clone()
        if tokenizer.pad_token_id is not None:
            labels[labels == tokenizer.pad_token_id] = -100

        outputs = model(**inputs, labels=labels, use_cache=False)

        num_valid_tokens = (labels != -100).sum().item()
        if num_valid_tokens > 0:
            total_loss += outputs.loss.item() * num_valid_tokens
            total_tokens += num_valid_tokens

        del inputs, outputs, labels
        torch.cuda.empty_cache()

    if total_tokens == 0:
        return float('inf')

    return math.exp(total_loss / total_tokens)


class PerplexityCallback(TrainerCallback):
    """Evaluate perplexity on validation set and save best model during training."""

    def __init__(
        self,
        tokenizer,
        val_sentences: List[str],
        val_sentences_by_source: dict,
        output_dir: str,
        batch_size: int = 16,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.val_sentences = val_sentences
        self.val_sentences_by_source = val_sentences_by_source
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.max_length = max_length
        self.best_perplexity = float('inf')

    def on_evaluate(self, args, state, control, model, **kwargs):
        if not self.val_sentences:
            return

        # Overall perplexity
        ppl = compute_perplexity(
            model, self.tokenizer, self.val_sentences,
            batch_size=self.batch_size,
            max_length=self.max_length,
            desc=f"Val PPL (step {state.global_step})",
        )
        print(f"\n[Step {state.global_step}] Perplexity: {ppl:.2f} (best: {self.best_perplexity:.2f})")

        # Per-source breakdown
        for source_name, source_sentences in self.val_sentences_by_source.items():
            if source_sentences:
                src_ppl = compute_perplexity(
                    model, self.tokenizer, source_sentences,
                    batch_size=self.batch_size,
                    max_length=self.max_length,
                    desc=f"  {source_name} PPL",
                )
                print(f"  {source_name}: {src_ppl:.2f}")

        if ppl < self.best_perplexity:
            self.best_perplexity = ppl
            print(f"[Step {state.global_step}] New best! Saving to {self.output_dir}")
            model.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)

        model.train()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a causal LLM with LoRA for BCI rescoring")
    parser.add_argument("--source-files", type=str, nargs="+", required=True,
                        help="Source files as name:path pairs, e.g. personal:/data/personal.txt switchboard:/data/swb.txt")
    parser.add_argument("--upsample-factors", type=str, nargs="*", default=[],
                        help="Upsample factors as name:factor pairs, e.g. switchboard:4")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save the LoRA adapter")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3.5-4B",
                        help="HuggingFace model name or path")
    parser.add_argument("--cache-dir", type=str, default="~/brand/huggingface",
                        help="HuggingFace cache directory")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--num-epochs", type=float, default=3)
    parser.add_argument("--eval-every", type=float, default=0.25,
                        help="Evaluate every N epochs")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--warmup-fraction", type=float, default=0.03)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None,
                        help="CUDA device(s), e.g. '0' or '0,1'")
    args = parser.parse_args()

    # Parse source files: "name:path" pairs
    source_files = {}
    for item in args.source_files:
        name, path = item.split(":", 1)
        source_files[name] = path

    # Parse upsample factors: "name:factor" pairs
    upsample_factors = {}
    for item in args.upsample_factors:
        name, factor = item.split(":", 1)
        upsample_factors[name] = int(factor)

    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # ---- Data ----
    print("Preparing dataset...")
    train_sentences, val_sentences, stats, val_sentences_by_source = prepare_dataset(
        source_files=source_files,
        upsample_factors=upsample_factors,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(f"Train: {len(train_sentences)}, Val: {len(val_sentences)}")
    for source_name, s in stats.items():
        print(f"  {source_name}: {s['raw']} raw -> {s['unique']} unique -> "
              f"{s['train']} train (x{s['upsample_factor']}) + {s['val']} val")

    # ---- Model & Tokenizer ----
    print(f"Loading model: {args.model_name}")
    cache_dir = os.path.expanduser(args.cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=cache_dir,
        dtype=torch.bfloat16,
        device_map="auto",
        use_cache=False,
    )

    # ---- Baseline Perplexity ----
    if val_sentences:
        print("Computing baseline perplexity...")
        ppl_before = compute_perplexity(
            model, tokenizer, val_sentences,
            max_length=args.max_seq_length,
            desc="Baseline PPL",
        )
        print(f"Baseline perplexity: {ppl_before:.2f}")
    else:
        ppl_before = None

    # ---- LoRA ----
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )

    # ---- Dataset ----
    def formatting_func(example):
        return {"text": example["text"] + tokenizer.eos_token}

    train_ds = Dataset.from_dict({"text": train_sentences})
    train_ds = train_ds.map(formatting_func)
    train_ds = train_ds.shuffle(seed=args.seed)

    val_ds = None
    if val_sentences:
        val_ds = Dataset.from_dict({"text": val_sentences})
        val_ds = val_ds.map(formatting_func)

    # ---- Training Config ----
    steps_per_epoch = len(train_ds) // (args.batch_size * args.gradient_accumulation_steps)
    eval_steps = max(1, int(steps_per_epoch * args.eval_every))
    total_steps = int(steps_per_epoch * args.num_epochs)
    warmup_steps = int(total_steps * args.warmup_fraction)

    print(f"Steps/epoch: {steps_per_epoch}, Eval every {eval_steps} steps, "
          f"Total: {total_steps}, Warmup: {warmup_steps}")

    output_dir = os.path.expanduser(args.output_dir)
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        logging_steps=10,
        bf16=True,
        save_strategy="no",
        eval_strategy="steps",
        eval_steps=eval_steps,
        report_to="none",
        seed=args.seed,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
    )

    ppl_callback = PerplexityCallback(
        tokenizer=tokenizer,
        val_sentences=val_sentences,
        val_sentences_by_source=val_sentences_by_source,
        output_dir=output_dir,
        max_length=args.max_seq_length,
    )

    # ---- Train ----
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
        args=training_args,
        callbacks=[ppl_callback],
    )

    print("Starting training...")
    trainer.train()

    # ---- Final Summary ----
    ppl_after = None
    if val_sentences:
        trainer.model.eval()
        ppl_after = compute_perplexity(
            trainer.model, tokenizer, val_sentences,
            max_length=args.max_seq_length,
            desc="Final PPL",
        )
        print(f"\nPerplexity: {ppl_before:.2f} -> {ppl_after:.2f}")
        print(f"Best during training: {ppl_callback.best_perplexity:.2f}")

    print(f"Best model saved to {output_dir}")

    # ---- Log Results ----
    results_file = Path(output_dir) / "finetuning_results.json"
    result_entry = {
        "timestamp": datetime.now().isoformat(),
        "model_name": args.model_name,
        "output_dir": output_dir,
        "source_files": source_files,
        "upsample_factors": upsample_factors,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "data_stats": stats,
        "baseline_ppl": round(ppl_before, 2) if ppl_before else None,
        "final_ppl": round(ppl_after, 2) if ppl_after else None,
        "best_ppl": round(ppl_callback.best_perplexity, 2) if ppl_callback.best_perplexity < float('inf') else None,
    }
    with open(results_file, "w") as f:
        json.dump(result_entry, f, indent=2)
    print(f"Results logged to {results_file}")


if __name__ == "__main__":
    main()
