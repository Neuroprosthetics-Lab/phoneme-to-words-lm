#!/usr/bin/env python3
"""Fine-tune a causal LLM with LoRA on domain-specific text for BCI rescoring."""

import argparse
import json
import math
import os
import hashlib
import random
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

from phoneme_to_words_lm.utils import HF_CACHE_DIR, TEXT_NORMALIZATION_VERSION
from phoneme_to_words_lm.sweep_contract import file_identity, positive_integer
from phoneme_to_words_lm.llm_scoring import (
    DEFAULT_PREFIX, SCORER_VERSION, prepare_scoring_inputs, training_example,
    ScoringDataCollator, sentence_perplexity, score_sentences,
)


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

    Mirrors the normalization applied by the decoder in decoder.py:
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
    """Split global normalized identities before per-source training upsampling.

    Args:
        source_files: Mapping of source name -> file path (e.g. {"personal": "path.txt"}).
        upsample_factors: Mapping of source name -> integer upsample factor.
        val_fraction: Fraction of global normalized sentence identities held out.
        seed: Random seed for reproducible splitting.

    Returns:
        (train_sentences, val_sentences, stats, val_by_source) where stats has
        per-source counts and val_by_source maps source name -> val sentence list.
    """
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError('source_files must contain at least one source')
    if isinstance(val_fraction, bool) or not math.isfinite(val_fraction) or not 0 <= val_fraction < 1:
        raise ValueError('val_fraction must be in [0,1)')
    if not isinstance(upsample_factors, dict) or set(upsample_factors)-set(source_files):
        raise ValueError('Upsampling factors must refer to known sources')
    for name, value in upsample_factors.items():
        positive_integer(f'upsample factor for {name}', value)
    rng = random.Random(seed)
    sources, raw_counts = {}, {}
    for name, path in source_files.items():
        raw = load_sentences(str(Path(path).expanduser()))
        sources[name] = sorted(set(preprocess_sentences(raw)))
        raw_counts[name] = len(raw)
    identities = sorted({text for values in sources.values() for text in values})
    if not identities:
        raise ValueError('No usable normalized training sentences')
    if val_fraction and len(identities) < 2:
        raise ValueError('Need at least two distinct sentences for train/validation; use val_fraction=0 for training only')
    rng.shuffle(identities)
    n_val = 0
    if val_fraction:
        n_val = min(len(identities) - 1, max(1, int(len(identities) * val_fraction)))
    validation = set(identities[:n_val])
    train, by_source, stats = [], {}, {}
    for name in sorted(sources):
        unique = sources[name]
        val_part = [text for text in unique if text in validation]
        train_part = [text for text in unique if text not in validation]
        factor = upsample_factors.get(name, 1)
        train.extend(train_part*factor)
        by_source[name] = val_part
        stats[name] = dict(raw=raw_counts[name], unique=len(unique),
                           train=len(train_part)*factor, val=len(val_part), upsample_factor=factor)
    rng.shuffle(train)
    return train, sorted(validation), stats, by_source


@torch.no_grad()
def compute_perplexity(model, tokenizer, sentences, batch_size=16, max_length=512,
                       device=None, desc="Computing Perplexity", *,
                       prefix=DEFAULT_PREFIX, score_eos=False):
    """Use the same exact target masks/counts as decoder sentence_v1 scoring.

    device/desc remain accepted for caller compatibility. The model's device
    determines inference placement; candidates are never silently truncated.
    """
    if device is not None and torch.device(device) != torch.device(model.device):
        raise ValueError('device must match the loaded model device')
    return sentence_perplexity(model, tokenizer, sentences, batch_size=batch_size,
                               max_length=max_length, prefix=prefix, score_eos=score_eos)


def validation_report(model, tokenizer, sentences, by_source=None, *, batch_size=16,
                      max_length=512, prefix=DEFAULT_PREFIX, score_eos=False):
    """Score each unique sentence once; aggregate exact counts overall/per source."""
    unique = list(dict.fromkeys(sentences))
    was_training = model.training
    model.eval()
    try:
        raw, counts = score_sentences(model, tokenizer, unique, batch_size=batch_size,
                                      max_length=max_length, prefix=prefix, score_eos=score_eos)
    finally:
        model.train(was_training)
    indices = {text: i for i, text in enumerate(unique)}

    def aggregate(texts):
        ids = [indices[text] for text in dict.fromkeys(texts)]
        count = sum(counts[i] for i in ids)
        nll = -math.fsum(raw[i] for i in ids)
        try:
            ppl = math.exp(nll/count) if count else None
        except OverflowError:
            ppl = None
        return dict(nll=nll, target_tokens=count, perplexity=ppl)
    return {
        'overall': aggregate(unique),
        'by_source': {
            name: aggregate(values) for name, values in (by_source or {}).items()
        },
    }


def save_adapter(model, tokenizer, directory, *, prefix, score_eos):
    """Complete a save before publishing any of its files."""
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.adapter-', dir=directory.parent) as temporary:
        model.save_pretrained(temporary)
        tokenizer.save_pretrained(temporary)
        files = list(Path(temporary).glob('adapter_model.*'))
        if not Path(temporary, 'adapter_config.json').is_file() or not any(p.stat().st_size for p in files):
            raise RuntimeError('Training did not produce a usable PEFT adapter')
        Path(temporary, 'llm_scoring.json').write_text(json.dumps(
            dict(version=SCORER_VERSION, prefix=prefix, score_eos=score_eos), indent=2))
        directory.mkdir(parents=True, exist_ok=True)
        for source in Path(temporary).iterdir():
            if source.is_file():
                source.replace(directory/source.name)


class PerplexityCallback(TrainerCallback):
    """Evaluate on optimizer-step boundaries without a duplicate Trainer pass."""
    def __init__(self, tokenizer, val_sentences, val_sentences_by_source, output_dir,
                 batch_size=16, max_length=512, prefix=DEFAULT_PREFIX, score_eos=False,
                 eval_steps=1):
        self.tokenizer = tokenizer
        self.sentences = val_sentences
        self.by_source = val_sentences_by_source
        self.output_dir = output_dir
        self.options = dict(batch_size=batch_size, max_length=max_length, prefix=prefix, score_eos=score_eos)
        self.eval_steps = eval_steps
        self.best_perplexity = math.inf
        self.best_report = None
        self.last_report = None
        self.last_step = None

    def on_step_end(self, args, state, control, model, **kwargs):
        if self.sentences and state.global_step % self.eval_steps == 0:
            self.on_evaluate(args, state, control, model, **kwargs)

    def on_evaluate(self, args, state, control, model, **kwargs):
        self.evaluate(model, state.global_step)

    def evaluate(self, model, step):
        if not self.sentences:
            return
        if self.last_step == step:
            return self.last_report
        report = validation_report(model, self.tokenizer, self.sentences, self.by_source, **self.options)
        self.last_report, self.last_step = report, step
        ppl = report['overall']['perplexity']
        print(f"[Step {step}] Validation: {report}", flush=True)
        if ppl is not None and math.isfinite(ppl) and ppl < self.best_perplexity:
            save_adapter(model, self.tokenizer, self.output_dir,
                         prefix=self.options['prefix'], score_eos=self.options['score_eos'])
            self.best_perplexity, self.best_report = ppl, report
        return report


def optimizer_schedule(n_examples, batch_size, accumulation, epochs, eval_every, max_steps=-1):
    for name, value in (('examples', n_examples), ('batch_size', batch_size), ('accumulation', accumulation)):
        positive_integer(name, value)
    for name, value in (('epochs', epochs), ('eval_every', eval_every)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    steps = math.ceil(math.ceil(n_examples/batch_size)/accumulation)
    return dict(steps_per_epoch=steps, total_steps=max_steps if max_steps > 0 else math.ceil(epochs*steps),
                eval_steps=max(1, math.ceil(steps*eval_every)))


def verify_adapter_reload(model, tokenizer, directory, sentences, expected, options):
    """Load serialized LoRA weights into the unchanged base; avoid a second 4B copy."""
    name = 'verify_saved'
    previous = model.active_adapter
    model.load_adapter(str(directory), adapter_name=name, is_trainable=False)
    try:
        model.set_adapter(name)
        actual = validation_report(model, tokenizer, sentences, **options)['overall']
        if actual['target_tokens'] != expected['target_tokens'] or not math.isclose(
            actual['nll'], expected['nll'], rel_tol=1e-5, abs_tol=.01):
            raise RuntimeError(f'Reloaded adapter differs from saved checkpoint: {directory}')
        return dict(target_tokens=actual['target_tokens'], nll_absolute_error=abs(actual['nll']-expected['nll']))
    finally:
        model.set_adapter(previous)
        model.delete_adapter(name)


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
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="HuggingFace cache directory (default: ~/brand/huggingface)")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--llm-prefix", default=DEFAULT_PREFIX,
                        help="Textual sentence conditioning prefix; default is a newline")
    parser.add_argument("--score-eos", action="store_true",
                        help="Train/evaluate EOS as a target; pass llm_score_eos=True when decoding")
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
    parser.add_argument('--max-steps', type=int, default=-1, help='Optional optimizer-step limit')
    parser.add_argument('--dtype', choices=['bfloat16', 'float16', 'float32'], default='bfloat16')
    parser.add_argument('--eval-steps', type=int, default=None, help='Override epoch-based evaluation interval')
    args = parser.parse_args()

    if args.device is not None:
        if ',' in args.device or int(os.environ.get('WORLD_SIZE', '1')) != 1:
            raise ValueError('This training utility currently supports one process/device per run')
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Use one process/device per run for validation/checkpoint selection')
    from transformers import set_seed
    set_seed(args.seed)

    # Parse and validate data/training settings before loading the model.
    def parse_pairs(items, conversion=str):
        result = {}
        for item in items:
            name, value = item.split(':', 1)
            if not name or name in result:
                raise ValueError(f'Duplicate or empty source name: {name!r}')
            result[name] = conversion(value)
        return result

    source_files = parse_pairs(args.source_files)
    upsample_factors = parse_pairs(args.upsample_factors, int)
    train_sentences, val_sentences, stats, by_source = prepare_dataset(
        source_files, upsample_factors, val_fraction=args.val_fraction, seed=args.seed)
    schedule = optimizer_schedule(len(train_sentences), args.batch_size,
        args.gradient_accumulation_steps, args.num_epochs, args.eval_every, args.max_steps)
    for name in ('max_seq_length', 'lora_rank', 'lora_alpha'):
        positive_integer(name, getattr(args, name))
    if args.max_steps != -1:
        positive_integer('max_steps', args.max_steps)
    if args.eval_steps is not None:
        positive_integer('eval_steps', args.eval_steps)
        schedule['eval_steps'] = args.eval_steps
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('learning_rate must be finite and positive')
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0 or not 0 <= args.warmup_fraction <= 1:
        raise ValueError('Invalid weight_decay or warmup_fraction')
    output = Path(args.output_dir).expanduser().resolve()
    if (output/'adapter_config.json').exists() or (output/'finetuning_results.json').exists():
        raise ValueError('Use a new output directory for each training run')
    try:
        from datasets import Dataset
        from peft import LoraConfig, TaskType
        from trl import SFTTrainer, SFTConfig
    except ImportError as exc:
        raise ImportError('Finetuning requires datasets, peft and trl; see the tested training environment in the results report') from exc
    cache_dir = os.path.expanduser(args.cache_dir or HF_CACHE_DIR)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    # Prepare/validate every target before allocating model weights.
    def make_dataset(sentences):
        requests = prepare_scoring_inputs(tokenizer, sentences, prefix=args.llm_prefix,
                                          score_eos=args.score_eos, max_length=args.max_seq_length)
        return Dataset.from_list([training_example(r) for r in requests])

    train_ds = make_dataset(train_sentences).shuffle(seed=args.seed)
    if val_sentences:
        make_dataset(val_sentences)
    use_cuda = torch.cuda.is_available()
    if not use_cuda and args.dtype != 'float32':
        raise ValueError('CPU training requires --dtype float32')
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=cache_dir,
        dtype=getattr(torch, args.dtype),
        attn_implementation='sdpa',
        device_map={'': 'cuda:0' if use_cuda else 'cpu'},
    )
    model.config.use_cache = False
    options = dict(batch_size=args.batch_size, max_length=args.max_seq_length,
                   prefix=args.llm_prefix, score_eos=args.score_eos)
    baseline = validation_report(model, tokenizer, val_sentences, by_source, **options) if val_sentences else None
    targets = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    matched = [name for name, _ in model.named_modules() if name.rsplit('.', 1)[-1] in targets]
    if not matched:
        raise ValueError('No configured LoRA target modules found in this model')
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.,
        bias='none',
        task_type=TaskType.CAUSAL_LM,
        target_modules=targets,
    )
    output.mkdir(parents=True, exist_ok=True)
    def identity(text):
        return hashlib.sha256(text.encode()).hexdigest()

    split = dict(train_unique=sorted(identity(t) for t in set(train_sentences)),
                 validation=sorted(identity(t) for t in val_sentences),
                 validation_by_source={name: [identity(t) for t in values] for name, values in by_source.items()})
    (output/'split_manifest.json').write_text(json.dumps(split, indent=2))
    callback = PerplexityCallback(tokenizer, val_sentences, by_source, str(output/'best'),
                                  eval_steps=schedule['eval_steps'], **options)
    # The callback handles validation and best-checkpoint selection in one pass.
    training_args = SFTConfig(
        output_dir=str(output/'trainer'),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=int(schedule['total_steps']*args.warmup_fraction),
        lr_scheduler_type='cosine',
        optim='adamw_torch',
        logging_steps=1,
        bf16=args.dtype == 'bfloat16',
        fp16=args.dtype == 'float16',
        use_cpu=not use_cuda,
        save_strategy='no',
        eval_strategy='no',
        report_to='none',
        seed=args.seed,
        dataset_kwargs={'skip_prepare_dataset': True},
        max_length=args.max_seq_length,
        packing=False,
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        data_collator=ScoringDataCollator(tokenizer.pad_token_id),
        train_dataset=train_ds,
        peft_config=peft_config,
        args=training_args,
        callbacks=[callback],
    )
    train_output = trainer.train()
    save_adapter(trainer.model, tokenizer, output/'final', prefix=args.llm_prefix, score_eos=args.score_eos)
    final = callback.evaluate(trainer.model, trainer.state.global_step)
    # No evaluation or no finite best still produces a usable selected adapter.
    selected = 'best' if callback.best_report is not None else 'final'
    checks = {}
    check_sentences = val_sentences or list(dict.fromkeys(train_sentences))[:4]
    if final is not None:
        expected_final = final['overall']
    else:
        expected_final = validation_report(
            trainer.model, tokenizer, check_sentences, **options)['overall']
    checks['final'] = verify_adapter_reload(trainer.model, tokenizer, output/'final', check_sentences, expected_final, options)
    if callback.best_report is not None:
        checks['best'] = verify_adapter_reload(trainer.model, tokenizer, output/'best',
                                               val_sentences, callback.best_report['overall'], options)
    # Keep the existing decoder-facing output-dir API: root contains the selected adapter.
    for source in (output/selected).iterdir():
        if source.is_file():
            shutil.copy2(source, output/source.name)
    import importlib.metadata
    results = dict(timestamp=datetime.now().isoformat(), model_name=args.model_name,
        text_normalization=TEXT_NORMALIZATION_VERSION,
        training_config=vars(args), source_identities={name: file_identity(Path(path).expanduser()) for name, path in source_files.items()},
        source_files=source_files, upsample_factors=upsample_factors, stats=stats,
        train_examples=len(train_sentences), validation_unique=len(val_sentences), schedule=schedule,
        actual_optimizer_steps=trainer.state.global_step, selected_adapter=selected,
        baseline_validation=baseline, final_validation=final, best_validation=callback.best_report,
        adapter_reload_checks=checks, lora_target_modules=matched, training_metrics=train_output.metrics,
        versions={name: importlib.metadata.version(name) for name in ('torch','transformers','peft','trl','datasets')})
    (output/'finetuning_results.json').write_text(json.dumps(results, indent=2, allow_nan=False))
    print(f'Saved final adapter and selected {selected} adapter in {output}', flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
