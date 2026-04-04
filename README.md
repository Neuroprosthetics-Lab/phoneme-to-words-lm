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

Additional capabilities:
- **LLM rescoring** with efficient GPU batch processing over large n-best lists (e.g., 100 hypotheses)
- **LoRA finetuning** of the rescoring LLM on domain-specific text for improved accuracy
- **Online (streaming) decoding** for real-time applications
- **Optuna hyperparameter sweep** with multi-objective optimization (WER + decode speed)
- Fast enough to be used to get validation WER feedback during training/finetuning of the acoustic model, enabling new possibilities for end-to-end optimization and avoiding overfitting on PER.

## Table of Contents

- [Installation](#installation)
- [HuggingFace Cache Configuration](#huggingface-cache-configuration)
- [KenLM File Setup](#kenlm-file-setup)
- [Quick Start](#quick-start)
- [Logit Preprocessing](#logit-preprocessing)
- [Hyperparameters](#hyperparameters)
- [LLM Rescoring](#llm-rescoring)
- [Online Decoding](#online-decoding)
- [LLM Finetuning](#llm-finetuning)
- [Hyperparameter Sweep](#hyperparameter-sweep)
- [Fake Logits Demo](#fake-logits-demo)

## Installation

### Prerequisites

- **Miniconda** installed at `~/miniconda3` ([install guide](https://docs.anaconda.com/miniconda/install/))
- **CMake** (Linux: `sudo apt install cmake`)

### Environment setup

Run the setup script to create a conda environment with all dependencies:

```bash
./env_setup.sh           # full install (default)
./env_setup.sh --no-gpu  # skip GPU acceleration packages
```

This creates a `phoneme_lm` conda environment with PyTorch, flashlight-text (bundled with the required `kTrieMaxLabel = 60` patch), KenLM, and all other dependencies. The package itself is installed in editable mode (`pip install -e .`).

### Manual installation

If you already have an environment with PyTorch, kenLM, flashlight-text, and the optional GPU acceleration packages, you can skip the setup script and just install the package:

```bash
pip install -e .
```

Note: flashlight-text requires a source modification because phoneme sequence homophones can exceed the default max trie label length of 6. The bundled `flashlight_text/` directory has this patch pre-applied. Install it with:

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

Note: you could conceivably use any tokenization scheme you want, but here we focus on phoneme-level decoding with a standard ARPAbet phoneme set and a 5gram model trained on openwebtext2. The instructions below assume you are starting with a WFST lexicon and an ARPA n-gram language model.

### 1. Build KenLM from source

You need the `build_binary` CLI tool to compile `.arpa` files into the efficient `.bin` format. The easiest way is via vcpkg:

```bash
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh
./vcpkg integrate install
./vcpkg install kenlm
```

After installation, `build_binary` will be at:
```
vcpkg/installed/x64-linux/tools/kenlm/build_binary
```

### 2. Compile the language model

Start with an `.arpa` file and compile it to `.bin`:

```bash
build_binary PATH/TO/INPUT.arpa PATH/TO/OUTPUT.bin
```

### 3. Convert the lexicon

Convert a WFST lexicon to Flashlight-Text format (`WORD P1 P2 P3 SIL`). Note that SIL is appended to the end of every pronunciation:

```python
import re
VARIANT_RE = re.compile(r"\(\d+\)$")

source_file = '/path/to/wfst/lexicon.txt'
output = '/path/to/kenlm/lexicon.txt'

with open(source_file, 'r') as f:
    lines = f.readlines()
lines.sort()

reformatted_lines = []
for line in lines:
    parts = line.strip().split()
    word = parts[0]
    # remove variant suffixes like '(1)', '(2)', etc.
    word = VARIANT_RE.sub('', word)
    phones = ' '.join(parts[1:])
    phones += ' SIL'  # Append 'SIL' to the end of the phones
    reformatted_lines.append(f"{word} {phones}\n")

with open(output, 'w') as f:
    f.writelines(reformatted_lines)
```

### 4. Create `tokens.txt`

List all tokens, one per line. **The order matters** — it must be `BLANK, SIL, AA..ZH`:

```
BLANK
SIL
AA
AE
AH
AO
AW
AY
B
CH
D
DH
EH
ER
EY
F
G
HH
IH
IY
JH
K
L
M
N
NG
OW
OY
P
R
S
SH
T
TH
UH
UW
V
W
Y
Z
ZH
```

### 5. Organize files

Your `lm.bin`, `lexicon.txt`, and `tokens.txt` should all go into one folder.

## Quick Start

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

The decoder internally handles temperature scaling, log-softmax conversion, and blank penalty application (via the `temperature`, `blank_penalty` constructor parameters). You only need to ensure your logit tokens are in the correct order before passing them to the decoder.

### Token order

The decoder expects logits with tokens ordered to match `tokens.txt`: `[BLANK, SIL, AA..ZH]`.

If your model outputs logits in the order `[BLANK, AA..ZH, SIL]` (BLANK at index 0, SIL at the end), rearrange before decoding:

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

Rescore the n-best beam search hypotheses with a causal LLM for improved accuracy.

### How it works

1. The beam search decoder returns `n_best` candidate sentences
2. Each candidate is preprocessed with `remove_punctuation()` and `replace_words()`
3. Candidates are batched and scored by the LLM (sum of per-token log-probabilities)
4. An optional `llm_length_penalty` can be subtracted per-token
5. Final scores combine beam scores with weighted LLM scores: `beam_score + llm_alpha * llm_score`

### Model setup

The LLM is loaded via HuggingFace's `transformers` library. Currently using **Qwen3.5-4B** (~8-10 GB GPU memory in bfloat16). You can swap in other sizes or model families.

### Example

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

## Online Decoding

`KenLMFlashlightTextLM` supports real-time streaming decoding via flashlight's incremental beam search API.

### API

1. **`online_decode_begin()`** -- Initialize decoder state for a new utterance.
2. **`online_decode_step(logits)`** -- Feed new frame(s) and get the current best sentence. Input: raw logits, shape `(T, num_tokens)` or `(1, T, num_tokens)`.
3. **`online_decode_end()`** -- Finalize decoding, get full n-best list with optional LLM rescoring.

### Example

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
python -m phoneme_to_words_lm.finetune_llm \
    --source-files personal:/data/personal.txt switchboard:/data/swb.txt \
    --upsample-factors switchboard:4 \
    --output-dir /path/to/lora_adapter \
    --model-name Qwen/Qwen3.5-4B \
    --num-epochs 3 \
    --lora-rank 16 \
    --lora-alpha 32
```

The script:
- Loads sentences from one or more source files (e.g., Switchboard corpus)
- Processes sentences - lowercasing, punctuation removal, word replacement (e.g., "ok" -> "okay")
- Deduplicates and upsamples per-source
- Trains LoRA adapters on the base LLM
- Evaluates perplexity on a validation split during training
- Saves the best adapter checkpoint

Use the resulting adapter with the decoder via `llm_lora_path`:

```python
decoder = KenLMFlashlightTextLM(
    ...,
    llm_lora_path='/path/to/lora_adapter',
)
```

## Hyperparameter Sweep

Multi-objective Optuna sweep for the LM decoder, optimizing both WER and decode time:

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

See `sweep/example_lm_sweep.yaml` for TPE search and `sweep/example_lm_sweep_grid.yaml` for exhaustive grid search. The sweep requires a pkl file of precomputed phoneme logits (logits + transcriptions).

For interactive exploration of results:

```bash
pip install optuna-dashboard
optuna-dashboard sqlite:///path/to/sweep_output/sweep_study.db
```

## Fake Logits Demo

Test the decoder pipeline without a trained neural model by creating synthetic logits:

```bash
python examples/fake_logits_demo.py \
    --lexicon_path /path/to/lexicon.txt \
    --tokens_path /path/to/tokens.txt \
    --kenlm_model_path /path/to/lm.bin \
    --text "hello how are you" \
    --noise_std 3.0
```

This phonemizes the input text, creates logit frames with high values at the correct phoneme indices plus Gaussian noise, and runs both offline and online decoding. Useful for verifying the decoder is working correctly.
