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

If you already have PyTorch, KenLM, and the GPU packages, just `pip install -e .`. You also need the bundled `flashlight_text/`: it includes native hotword scoring and a `kTrieMaxLabel = 60` patch (phoneme homophones can exceed the upstream default of 6):

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

Use the n-gram training pipeline under `ngram/` — it produces `lm_unpruned.bin`, `lexicon.txt`, and `tokens.txt` ready for the decoder. See [N-gram Training](#n-gram-training) below and `ngram/README.md` for the full walkthrough.

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

The native decoder combines acoustic and weighted LM scores with word, silence,
unknown-word penalties, and optional log-add state merging. The returned
`beam_scores` preserve this total. `acoustic_scores` and `ngram_scores` are
representative-path diagnostics and cannot reconstruct a log-add total.

### Preprocessing parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `temperature` | Logit temperature scaling (applied before softmax). | 1.0 |
| `blank_penalty` | Blank token penalty -- `log(value)` is subtracted from blank logit. 1.0 = disabled. | 9.0 |
| `blank_skip_threshold` | Keep the first original frame per high-confidence blank run; 1.0 disables compression. | .98 to enable; default 1.0 |

Blank confidence is measured after temperature scaling and before the blank
penalty. Compression preserves one original frame per run, including leading
and trailing runs, so repeated phonemes retain a separating blank opportunity.
Every retained frame keeps its full token distribution. This approximates CTC
on a shorter sequence: omitted frames contribute neither acoustic scores nor
alternative alignments. Scores are native totals for the compressed sequence.
Streaming carries run state across chunk boundaries; empty chunks do not split
a run. `total_frames` counts received frames, while `search_frames` counts retained
frames. Results record the policy and settings in `preprocessing_config`.

### LLM rescoring parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `llm_alpha` | Weight on LLM score; retune on validation data after changing scoring policy. | default 0.55 |
| `llm_length_penalty` | Per-token length penalty applied to LLM score. | 0.0 |
| `llm_batch_size` | Maximum candidates per model call; batches are bucketed by length. | 100 |
| `llm_prefix` | Textual conditioning prefix for every sentence. | newline (`"\n"`) |
| `llm_score_eos` | Include EOS probability and count it in length penalty. | False |
| `llm_max_length` | Maximum full input tokens; overlength inputs raise an error. | 512 |
| `llm_max_batch_tokens` | Maximum padded batch size × sequence length. | 2048 |
| `llm_logprob_chunk_size` | Number of vocabulary rows processed in FP32 at once. | 16 |

Final scoring equation with LLM rescoring:
```
llm_score = raw_llm_score - llm_length_penalty * llm_token_count
final_score = beam_score + llm_alpha * llm_score
```

## LLM Rescoring

Rescore the n-best beam search hypotheses with a causal LLM. Each candidate is preprocessed (punctuation removal, word replacement), batch-scored by the LLM (sum of per-token log-probabilities, with optional `llm_length_penalty`), and combined with the beam score: `beam_score + llm_alpha * llm_score`.

The `sentence_v1` policy jointly tokenizes prefix, context and candidate, so the
first candidate token is scored even when the tokenizer has no BOS. With context,
the exact text is `prefix + context + newline + candidate` (the extra newline is
omitted if context already ends with one). Tokens crossing the character boundary
belong to the candidate. Padding and context targets are excluded from both sums
and counts. A fast tokenizer with offset mappings is required. EOS scoring is
explicit and defaults off; choose the policy using validation data.

The output includes `beam_scores`, `raw_llm_scores`, `llm_token_counts`, adjusted
`llm_scores`, raw spellings (`raw_word_seqs`) and `word_ids`. If the best native
path emits no words (including exact ties), the decoder abstains: `status='empty'`,
`word_seqs=['']`, and the native beam/acoustic/LM scores are preserved. This
means no completed words, not a calibrated silence detector. Empty results skip
LLM scoring (`None` scores, zero token counts). Missing input or valid paths use
`status='no_input'` or `'no_path'` with `-inf` scores; ordinary results use `'ok'`.
Lower-scoring empty paths do not enter LLM competition. Exact beam-score ties are
resolved deterministically by text; exact final-score ties use native beam score,
then text. Reranking legacy results without native beam
scores raises an error: decode them again rather than reconstructing totals.

Text normalization uses policy `english_v3`: Unicode apostrophes and decomposable
Latin accents are cleaned before filtering. Separator punctuation creates spaces
(`left/right` → `left right`, `stop--wait` → `stop wait`); internal hyphens remain.
Paired quotation marks are removed, contraction suffixes such as `we 're` are
joined, and leading elisions such as `potion 'cause` stay separate. Ambiguous words
such as `lets`, `cant`, `wont`, `masters` and `masters'` keep their spelling.
British/American spelling normalization remains supported. Experiment and
training metadata record the policy; recompute references, splits, candidate
scores and tuning in a new output directory when migrating from older results.
Raw word IDs/spellings remain available. N-gram corpus preprocessing also
expands numbers and splits sentences; it does not rewrite prebuilt resources.

Inference disables unused model caches and bounds FP32 probability workspace.
GPU OOMs retry smaller batches; a singleton that cannot fit raises an error.
Candidates and contexts are never silently truncated. The token budget bounds
activations/logits, not model-weight memory, so choose a model that fits the device.

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

The base LLM can use context without retraining. The finetuning CLI below accepts plain sentences; context/candidate training pairs require a separate data-format extension. Longer contexts increase memory and latency.

## Hotword Biasing

Boost the n-gram probability of specific words at decode time without retraining the LM — useful for names, jargon, and domain vocabulary the LM has seen weakly or not at all. Each hotword gets a log10 bonus added to the KenLM score at word completion, plus a trie lookahead seed so partial-word paths survive beam pruning. The KenLM itself is never modified.

### Constraints

- Hotwords must already exist in `lexicon.txt`. Lookup ignores case while preserving original KenLM word IDs; ambiguous case collisions and duplicate normalized keys raise errors. Missing words raise `KeyError` (no g2p fallback).
- Bonuses are finite log10 values. At word completion, `+1.0` adds one log10 unit before `lm_weight` is applied; negative and zero bonuses are allowed. Every listed word also receives a fixed **+5.0** trie lookahead seed, even with a zero or negative bonus. Removing the word from the mapping removes that seed.

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

With the bundled extension installed, scoring stays in C++: an empty mapping
uses KenLM directly, and a nonempty mapping uses `HotwordLM`. Check
`decoder.hotword_backend` for `kenlm`, `native`, or `python`. Older extensions
use a slower Python fallback for nonempty mappings and emit a rebuild warning.
Reinstall the bundled extension and restart the Python process or notebook
kernel to enable the native backend. Keep the decoder instance alive throughout
an utterance; one instance supports one active operation at a time.

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

A decoder instance supports one active utterance at a time. Offline decoding,
resource reload, or a second begin during a stream raises an error. Use
`online_decode_abort()` to explicitly discard a stream. Changes to search options
require `init_ngram_decoder()`; temperature, blank penalty and n-best may change
between utterances. Chunks are submitted to native code together, and `total_frames`
includes blank frames.

`ngram_time` is per-item offline preprocessing/search/extraction time; online it
includes all step work and finalization. Offline `ngram_batch_time`,
`llm_batch_time` (also exposed as legacy `llm_rescore_time`) and `batch_total_time`
are batch-scoped. Do not sum batch-scoped timings across items. Streaming retains
history for the finite utterance; indefinite history pruning is not implemented.

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

The script normalizes and deduplicates sentences, then splits **global sentence
identities** before per-source training upsampling. A sentence shared by sources
cannot enter both training and validation. Validation identities are scored once;
overall and per-source perplexity use summed negative log likelihood divided by
exact target-token counts. Source PPL values are not averaged together.

Each run saves `final/` and, when scheduled validation succeeds, `best/`. The
output directory itself contains the selected adapter: best if available, final
otherwise. Saved adapters are reloaded and their validation scores checked before
success is reported. `finetuning_results.json` distinguishes baseline, final and
best results; `split_manifest.json` records hashed sentence identities. Use a
fresh output directory for each run.

Use `--val-fraction 0` for training without validation; otherwise at least two
unique normalized sentences are required. `--max-steps` and `--eval-steps` permit
short runs, including runs ending before their first evaluation. Only single
process, single device training is supported (`--device 0`, or `--device cpu
--dtype float32`). PEFT, TRL and datasets are optional training dependencies; see
the tested isolated environment in the [results report](IMPLEMENTATION_TESTING_AND_RESULTS.md).
The general environment setup script was not revalidated by batch E.

Finetuning and perplexity use the same `sentence_v1` target masks as rescoring.
Use `--llm-prefix` and optional `--score-eos` to choose the policy; the saved
`llm_scoring.json` records it. Set matching decoder options when loading the
adapter. Prepared inputs exceeding `--max-seq-length` raise an error rather than
silently truncating sentence targets. Perplexity counts exact targets and keeps
real EOS labels even when PAD equals EOS.

Use the resulting adapter via `llm_lora_path`:

```python
decoder = KenLMFlashlightTextLM(
    ...,
    llm_lora_path='/path/to/lora_adapter',
)
```

## N-gram Training

To build your own KenLM `.bin` from scratch, use the pipeline under `ngram/`. It supports word-level and character-level (spelling) LMs, multiple corpora with weighted interpolation, and per-corpus n-gram orders. The pipeline normalizes raw text, trains per-corpus models with `lmplz`, optionally interpolates (SRILM `ngram` or KenLM `interpolate`), and emits `lm_unpruned.bin` + `lexicon.txt` + `tokens.txt` ready for the decoder. Defaults to the bundled CMU dict.

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

See `sweep/example_lm_sweep.yaml` for TPE search and
`sweep/example_lm_sweep_grid.yaml` for exhaustive grid search. WER is an edit-count
fraction (0.01 means 1%); time is mean seconds per sentence. The objective weights
multiply these raw quantities; there is no min-max normalization. Concurrent
workers contend for CPU/GPU resources, so benchmark finalists in isolation.

The pickle contains a list of records with `(T, C)` logits, `transcription` as a
string or zero-terminated integer character codes, and `adjusted_len` or
`adjusted_lens`. If both lengths are present they must agree. Optional string
`context` is forwarded to LLM rescoring. Empty normalized references and
letter-by-letter spelling references are filtered before `eval_every_nth`
subsampling; `max_sentences` optionally caps the selected set.

All supported decoder settings can be supplied as fixed top-level fields or in
`parameters`; sampled/fixed parameter entries override top-level values. This
includes adapters, hotwords, score penalties, log-add and LLM scoring policy.
Resource paths remain fixed, `--devices` assigns workers, and `alpha_range` selects
LLM alpha post hoc (`llm_alpha` must be omitted or zero). Each trial saves its full
effective configuration, source indices, candidates and scores. Edit counts are
computed once per candidate and reused for alpha/length-penalty selection. Changing
search settings still requires decoding again; persistent score caching is future work.

Metrics include n-gram/final/oracle WER, empty rate, median candidate count,
reference OOV rate against the **lexicon**, mean/p50/p95 end-to-end decode time and
throughput. Load time and first-decode time are separate. By default the first
sentence is excluded from latency statistics, while every sentence contributes to
WER (`timing_warmup_sentences: 1`; at least one timed sentence is retained).
`worker_timeout_seconds` defaults to null; subprocess failures, timeouts and
malformed results become Optuna **FAIL** trials.

Before starting workers, the launcher hashes the dataset, resources, model and
adapter files, tokenizer, native binaries, source code, runtime versions and
configuration. Models must already be available locally or in the HF cache;
workers use the resolved snapshot. `--resume` requires a matching identity.
Changed data/code/scoring settings or legacy studies require a new study/output
directory. Hashing large models has an explicit startup I/O cost; it happens once
per launcher invocation. Resume adds `n_trials` for non-grid samplers; a grid
resumes its remaining combinations. The seed is recorded, but concurrent
scheduling and resumed sampler state need not reproduce an uninterrupted proposal
sequence.

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

## Regression checks and migration

For repeatable accuracy/latency selection with separate validation and test data,
see [the evaluation guide](EVALUATION_GUIDE.md). One command screens configurations,
evaluates finalists, freezes the selected settings, and produces test results,
CSV/JSON/Markdown reports and an accuracy/latency plot:

```bash
python -B benchmarks/run_evaluation.py --config benchmarks/example_evaluation.yaml \
    --validation-logits /path/to/val.pkl --test-logits /path/to/test.pkl \
    --output benchmark_runs/my_comparison
```

Set the resource/model paths in the example YAML first. The default objective is
`1000 * WER_fraction + mean_seconds`; both metrics remain visible separately.
Grouped one-file splits, validation-only runs, resumable measurements and exported
decoder configurations are supported. Test results never select the configuration.

Run `python -m unittest discover -s tests -v` in the decoder environment.
`benchmarks/evaluate_logits.py` evaluates supplied local logits and model snapshots
without Redis. See [implementation testing and results](IMPLEMENTATION_TESTING_AND_RESULTS.md)
for commands, resource identities, before/after metrics and npl-davis migration notes.

Existing callers can use `blank_skip_threshold=.98` with the keep-first policy,
or `1.0` for full CTC. The old policy deleting every high-blank frame is no longer
used. Existing score consumers should use `beam_scores` and the final equation above. Recalibrate
LLM alpha/length penalties when adopting the corrected first-token/context policy.
