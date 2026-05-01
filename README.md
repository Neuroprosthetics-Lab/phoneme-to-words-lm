# phoneme_to_words_lm

CTC beam search decoding using KenLM n-gram language models via Flashlight-Text's `LexiconDecoder`, with optional LLM rescoring of n-best hypotheses. Converts phoneme-level CTC output into word-level sentences.

## Why this over WFST?

This package replaces the standard n-gram WFST decoding pipeline (Kaldi + OpenFST + custom C++) with an implementation built on PyTorch and Flashlight-Text. It uses the same underlying n-gram language model, achieves similar or better accuracy, and is dramatically more efficient:

| | WFST pipeline | This package |
|---|---|---|
| **Accuracy** | baseline | Similar or better |
| **Disk** (unpruned 5-gram, 125k-word lexicon) | 350+ GB | ~50 GB |
| **RAM at runtime** | 350+ GB (full model in memory) | <10 GB (trie loaded on demand) |
| **Load time** | ~15 min | <10 sec |
| **Decode speed** | baseline | 5-10x faster |
| **LLM rescoring** | OPT-6.7B (~16 GB VRAM) | Qwen3.5 model family (2-10 GB VRAM, faster, stronger) |
| **Hardware requirement** | Workstation / server | Consumer GPU |
| **Codebase** | Python + C++ (Kaldi, OpenFST) | Python + PyTorch + Flashlight-Text |

Additional capabilities: LLM rescoring with GPU batch processing over n-best lists, LoRA finetuning of the rescoring LLM, online (streaming) decoding, Optuna hyperparameter sweeps, hotword biasing, and contextual rescoring. Fast enough to get validation WER feedback during acoustic-model training instead of relying on PER.

## Installation

### Prerequisites

- **Miniconda** installed at `~/miniconda3` ([install guide](https://docs.anaconda.com/miniconda/install/))
- **CMake** (Linux: `sudo apt install cmake`)

### Environment setup

Run the setup script to create a conda environment with all dependencies:

```bash
./env_setup.sh           # full install (default)
./env_setup.sh --no-gpu  # skip GPU acceleration packages
conda activate phoneme_lm
```

This creates a `phoneme_lm` conda environment with PyTorch, flashlight-text (bundled with the required `kTrieMaxLabel = 60` patch), KenLM, and all other dependencies. The package itself is installed in editable mode (`pip install -e .`).

### Manual installation

If you already have PyTorch, KenLM, and the GPU packages, just `pip install -e .`. You also need the bundled `flashlight_text/` (it has a `kTrieMaxLabel = 60` patch — phoneme homophones can exceed the upstream default of 6):

```bash
pip install ./flashlight_text --no-build-isolation
```

## HuggingFace Cache Configuration

All HuggingFace downloads (LM files and LLM models) are stored in a single cache directory. The default is `~/brand/huggingface`. To change it, edit the `HF_CACHE_DIR` variable in `phoneme_to_words_lm/utils.py`.

## KenLM File Setup

Pre-built LM files are available on HuggingFace Hub. You can download them with the included script, or build your own from scratch.

### Download pre-built files

The 5-gram KenLM binary, lexicon, and token list are hosted at [nckcard/phoneme-lm-5gram](https://huggingface.co/nckcard/phoneme-lm-5gram).

```bash
# Download the unpruned 5-gram model (~50 GB) + lexicon + tokens
python -m phoneme_to_words_lm.download_5gram --output-dir /path/to/lm_files

# Download the pruned model instead (smaller)
python -m phoneme_to_words_lm.download_5gram --pruned --output-dir /path/to/lm_files

# Override cache directory for this download
python -m phoneme_to_words_lm.download_5gram --output-dir /path/to/lm_files --cache-dir ~/other/cache
```

The script downloads files into the HuggingFace cache (for deduplication and resumable downloads) and creates symlinks in `--output-dir`.

You can also download files programmatically:

```python
from phoneme_to_words_lm.download_5gram import download_5gram_files

paths = download_5gram_files(output_dir='/path/to/lm_files', pruned=False)
# paths = {'lexicon.txt': '...', 'tokens.txt': '...', '5gram_unpruned.bin': '...'}
```

### Build your own files

Use the n-gram training pipeline under `ngram/` — it produces `lm.bin`, `lexicon.txt`, and `tokens.txt` ready for the decoder. See [N-gram Training](#n-gram-training) below and `ngram/README.md` for the full walkthrough.

If you instead have a pre-built ARPA model and a WFST-format lexicon, you'll need to (1) compile the ARPA with KenLM's `build_binary`, (2) reformat the lexicon to `WORD P1 P2 P3 SIL` (one entry per word, SIL appended), and (3) write a `tokens.txt` with one token per line in the order `BLANK, SIL, AA..ZH` (the standard ARPAbet 39-phoneme set). All three files go in one folder.

## Quick Start

Use the phoneme_lm conda env: `conda activate phoneme_lm`

```python
import torch
from phoneme_to_words_lm import KenLMFlashlightTextLM

decoder = KenLMFlashlightTextLM(
    lexicon_path='/path/to/lexicon.txt',
    tokens_path='/path/to/tokens.txt',
    kenlm_model_path='/path/to/lm.bin',
    do_llm_rescoring=False,  # set True for LLM rescoring
)

# logits: (batch, time, num_tokens) raw pre-softmax tensor, CPU float32
# lengths: (batch,) lengths of each sequence in the batch, CPU int tensor
#     lengths is not necessary for batches of size 1, but required for larger batches to ignore padding frames
results = decoder.offline_decode(logits, lengths)
print(results[0]['word_seqs'][0])  # best hypothesis
```

## Logit Preprocessing

The decoder applies temperature scaling, log-softmax, and the blank penalty internally (via the `temperature` and `blank_penalty` constructor args). You only need to make sure logit tokens are ordered to match `tokens.txt`: `[BLANK, SIL, AA..ZH]`. If your model emits `[BLANK, AA..ZH, SIL]`, swap SIL to position 1 first:

```python
logits = torch.concat((logits[:, :, 0:1], logits[:, :, -1:], logits[:, :, 1:-1]), dim=-1)
```

## Hyperparameters

### Beam search parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `beam_size` | Number of candidate sentences kept at each step. Higher = slower but better. | 100-1000 |
| `beam_threshold` | Pruning threshold relative to best beam score. Lower = more aggressive pruning. | 25-50 |
| `token_beam_size` | Number of candidate tokens considered at each step. | 15-41 |
| `n_best` | Number of top hypotheses kept for rescoring. | 100 |
| `lm_weight` | Weight on the KenLM language model score. | 2.0-3.0 |
| `word_score` | Per-word bonus/penalty. Positive = longer sentences, negative = shorter. | 0.0 |
| `unk_score` | Score for unknown words (-inf = forbid). | -inf |
| `sil_score` | Score for silence tokens. | 0.0 |
| `log_add` | Use log-add (true) or max (false) for merging hypothesis scores. | true |

Internal beam scoring equation:
```
combined_score = acoustic_score + (lm_weight * ngram_score) + (word_score * num_words)
```

### Preprocessing parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `temperature` | Logit temperature scaling (applied before softmax). | 1.0 |
| `blank_penalty` | Blank token penalty -- `log(value)` is subtracted from blank logit. 1.0 = disabled. | 9.0 |
| `blank_skip_threshold` | Skip frames where blank prob exceeds this (1.0 = disabled). | 1.0 |

### LLM rescoring parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `llm_alpha` | Weight on LLM score in final rescoring. | 0.50-0.65 |
| `llm_length_penalty` | Per-token length penalty applied to LLM score. | 0.0 |
| `llm_batch_size` | Batch size for LLM inference over n-best hypotheses. | 32 |

Final scoring equation with LLM rescoring:
```
final_score = acoustic_score + (lm_weight * ngram_score) + (llm_alpha * llm_score) + (word_score * num_words)
```

## LLM Rescoring

Rescore the n-best beam search hypotheses with a causal LLM. Each candidate is preprocessed (punctuation removal, word replacement), batch-scored by the LLM (sum of per-token log-probabilities, with optional `llm_length_penalty`), and combined with the beam score: `beam_score + llm_alpha * llm_score`.

The LLM is loaded via HuggingFace `transformers`. Default is **Qwen3.5-4B** (~8-10 GB in bfloat16); swap in any causal LM.

```python
decoder = KenLMFlashlightTextLM(
    lexicon_path='/path/to/lexicon.txt',
    tokens_path='/path/to/tokens.txt',
    kenlm_model_path='/path/to/lm.bin',
    do_llm_rescoring=True,
    llm_model_name='Qwen/Qwen3.5-4B', # llm_cache_dir defaults to ~/brand/huggingface
    llm_device='cuda:0',
    llm_dtype='bfloat16',
    llm_alpha=0.55,
    llm_lora_path=None,  # or path to LoRA adapter
)
```

## Contextual Rescoring

Pass context strings (previous sentences, domain hints, keywords, etc.) to condition the LLM during rescoring. Context is prepended to each n-best hypothesis (only the hypothesis tokens are scored) and applied only at the LLM stage, not the n-gram beam search.

### Offline decoding with context

`offline_decode()` accepts a `contexts` list with one string per batch item:

```python
results = decoder.offline_decode(
    batched_logits,    # shape (B, T, num_tokens)
    batched_lengths,
    contexts=[
        "the patient reported chest pain",       # context for batch item 0
        "we need two cups of flour and an egg",  # context for batch item 1
    ],
)
```

### Online (streaming) decoding with context

`online_decode_end()` accepts a single `context` string:

```python
ctx = ""
for utterance_logits in utterance_stream:
    decoder.online_decode_begin()
    for frame_logits in utterance_logits:
        decoder.online_decode_step(frame_logits)
    # Pass previous decoded sentences as context
    final = decoder.online_decode_end(context=ctx)
    ctx = ctx + " " + final['word_seqs'][0]  # accumulate for next utterance
```

### Example context formats

The LLM sees raw text, so any sensible English prefix works — previous sentences, a `domain: cooking\n` hint, a `keywords: ...\n` line, multi-turn dialogue, or a combination:

```python
context = "previous: i need two cups of flour\ncurrent: "
context = "domain: cooking\nkeywords: flour oven baking\n"
```

The base LLM uses context without retraining; LoRA-finetuning on context-prefixed pairs improves utilization further. Keep contexts to a few recent sentences — long contexts grow memory and latency.

## Hotword Biasing

Boost the n-gram probability of specific words at decode time without retraining the LM — useful for names, jargon, and domain vocabulary the LM has seen weakly or not at all. Each hotword gets a log10 bonus added to the KenLM score at word completion, plus a trie lookahead seed so partial-word paths survive beam pruning. The KenLM itself is never modified.

### Constraints

- Hotwords must already exist in `lexicon.txt` (no g2p fallback — the decoder raises `KeyError` listing missing words). Add them to the lexicon and rebuild.
- Bonuses are log10. `+1.0` ≈ 10× boost; useful range is `1.0`–`5.0`. The trie-side lookahead bias is capped at 5.0.

### Initial hotwords

Pass either an inline mapping or a YAML file path:

```python
decoder = KenLMFlashlightTextLM(
    lexicon_path='/path/to/lexicon.txt',
    tokens_path='/path/to/tokens.txt',
    kenlm_model_path='/path/to/lm.bin',
    hotwords={'alpha': 4.0, 'bravo': 4.0, 'nato': 2.5},
)
# or
decoder = KenLMFlashlightTextLM(
    ...,
    hotwords_path='examples/example_hotwords.yaml',
)
```

The YAML file is a flat top-level mapping of `word: bonus`. See `examples/example_hotwords.yaml` for a template.

### Runtime updates

```python
decoder.set_hotwords({'alpha': 4.0, 'bravo': 4.0})  # replace active set
decoder.get_hotwords()                              # currently-applied mapping
decoder.clear_hotwords()                            # remove all
```

`set_hotwords` is *deferred* — changes apply at the next utterance boundary. Changing the *set* of hotwords rebuilds the trie/decoder (a few seconds for a 312k-entry lexicon); changing only the *bonuses* is essentially free. The KenLM is reused either way.

## Online Decoding

`KenLMFlashlightTextLM` supports real-time streaming via flashlight's incremental beam search. Use `online_decode_begin()` to start an utterance, `online_decode_step(logits)` to feed frame(s) and read the current best sentence (input shape `(T, num_tokens)` or `(1, T, num_tokens)`), and `online_decode_end()` to finalize and get the full n-best list with optional LLM rescoring.

```python
decoder.online_decode_begin()
for frame_logits in logit_stream:
    result = decoder.online_decode_step(frame_logits)
    print(f"[{result['total_frames']} frames] {result['word_seq']}")
final = decoder.online_decode_end()
print(f"Final: {final['word_seqs'][0]}")
```

## LLM Finetuning

Fine-tune a causal LLM with LoRA on domain-specific text for improved rescoring:

```bash
conda activate phoneme_lm
python -m phoneme_to_words_lm.finetune_llm \
    --source-files personal:/data/personal.txt switchboard:/data/swb.txt \
    --upsample-factors switchboard:4 \
    --output-dir /path/to/lora_adapter \
    --model-name Qwen/Qwen3.5-4B \
    --num-epochs 3 \
    --lora-rank 16 \
    --lora-alpha 32
```

The script loads sentences from one or more source files, normalizes them (lowercase, punctuation removal, word replacement), deduplicates and upsamples per-source, trains LoRA adapters on the base LLM with held-out perplexity evaluation, and saves the best checkpoint.

Use the resulting adapter via `llm_lora_path`:

```python
decoder = KenLMFlashlightTextLM(
    ...,
    llm_lora_path='/path/to/lora_adapter',
)
```

## N-gram Training

To build your own KenLM `.bin` from scratch, use the pipeline under `ngram/`. It supports word-level and character-level (spelling) LMs, multiple corpora with weighted interpolation, and per-corpus n-gram orders. The pipeline normalizes raw text, trains per-corpus models with `lmplz`, optionally interpolates (SRILM `ngram` or KenLM `interpolate`), and emits `lm.bin` + `lexicon.txt` + `tokens.txt` ready for the decoder. Defaults to the bundled CMU dict.

```bash
conda activate phoneme_lm
cd ngram
python train_ngram_lm.py --config example_config.yaml
```

See `ngram/README.md` for the full config reference, prerequisites (including how to build KenLM's `interpolate` binary, which `vcpkg install kenlm` does not provide), and worked examples.

## Hyperparameter Sweep

Optuna sweep for the LM decoder, optimizing a weighted single-objective combining WER and decode time:

```
score = wer_weight * wer + time_weight * avg_decode_time   (lower is better)
```

```bash
# Single GPU sweep
python sweep/lm_sweep.py --sweep_config sweep/example_lm_sweep.yaml --devices cuda:0

# Concurrency on a single GPU (e.g., 4 concurrent trials on cuda:0)
python sweep/lm_sweep.py --sweep_config sweep/example_lm_sweep.yaml --devices cuda:0:4

# Multi-GPU with concurrency
python sweep/lm_sweep.py --sweep_config sweep/example_lm_sweep.yaml --devices cuda:0:2,cuda:1

# Resume a previous sweep
python sweep/lm_sweep.py --sweep_config sweep/example_lm_sweep.yaml --resume
```

See `sweep/example_lm_sweep.yaml` for TPE search and `sweep/example_lm_sweep_grid.yaml` for exhaustive grid search. The sweep requires a pkl file of precomputed phoneme logits (logits + transcriptions). The `objective_weights` block in the sweep config controls the WER/time balance.

For interactive exploration of results:

```bash
pip install optuna-dashboard
optuna-dashboard sqlite:///path/to/sweep_output/sweep_study.db
```

## Fake Logits Demo

Test the decoder pipeline without a trained neural model by creating synthetic logits:

```bash
conda activate phoneme_lm
python examples/fake_logits_demo.py \
    --lexicon_path /path/to/lexicon.txt \
    --tokens_path /path/to/tokens.txt \
    --kenlm_model_path /path/to/lm.bin \
    --text "hello how are you" \
    --noise_std 3.0
```

This phonemizes the input text, creates logit frames with high values at the correct phoneme indices plus Gaussian noise, and runs both offline and online decoding. Useful for verifying the decoder is working correctly.
