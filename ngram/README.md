# N-gram Language Model Training

Train KenLM n-gram language models for use with the `KenLMFlashlightTextLM` decoder. Supports word-level LMs (for general sentence decoding) and character-level spelling LMs (for letter-by-letter decoding), multiple corpora with weighted interpolation, and per-corpus n-gram orders.

## Prerequisites

### KenLM (lmplz + build_binary)

`lmplz` trains n-gram models from text and `build_binary` compiles `.arpa` files into the efficient `.bin` format used at runtime.

**Install via vcpkg:**
```bash
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg && ./bootstrap-vcpkg.sh
./vcpkg install kenlm
```

After installation, the binaries are at:
```
vcpkg/installed/x64-linux/tools/kenlm/lmplz
vcpkg/installed/x64-linux/tools/kenlm/build_binary
```

Add them to your PATH or provide absolute paths in the config.

### Interpolation tool (multi-corpus only)

If you have more than one corpus, you need one of the following two tools to interpolate them. Pick one and set `interpolator_backend` accordingly.

#### SRILM (`ngram`)

Download from http://www.speech.sri.com/projects/srilm/ (free for non-commercial use). After building, the `ngram` binary is in `bin/`.

SRILM loads every per-corpus ARPA into RAM simultaneously, so it does not scale to very large model sets. For configs with tens of GB of ARPAs, prefer KenLM `interpolate` below.

#### KenLM `interpolate`

KenLM ships a separate streaming interpolator that sort-merges on disk instead of loading full tries. **It is not built by `vcpkg install kenlm`** — you have to build it from source:

```bash
git clone https://github.com/kpu/kenlm.git
cd kenlm && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8 interpolate
```

Dependencies (Ubuntu): `cmake`, `g++`, `libboost-all-dev`, `libeigen3-dev`, `libbz2-dev`, `liblzma-dev`, `zlib1g-dev`.

The resulting binary lives at `kenlm/build/bin/interpolate`.

### Python dependencies

```bash
pip install num2words omegaconf nltk
```

The `phoneme_to_words_lm` package must be importable (installed in editable mode from the repository root: `pip install -e .`).

On first run, the NLTK `punkt_tab` tokenizer data will be downloaded automatically if not already present.

## Quick Start

**Single corpus, word-level 5-gram:**
```yaml
# my_config.yaml
output_dir: ~/lm_output/my_5gram
lm_type: word
interpolator_backend: srilm   # required; choice is immaterial for a single-corpus run
memory: "80%"
corpora:
  - path: /path/to/corpus.txt
    weight: 1
    order: 5
    pruning: [0, 0, 1, 1, 2]
lmplz_path: lmplz
build_binary_path: build_binary
# cmu_dict_path: null     # defaults to the cmu_dict.pkl bundled with phoneme_to_words_lm
```

```bash
python train_ngram_lm.py --config my_config.yaml
```

## Config Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `output_dir` | string | yes | Output directory for all generated files |
| `lm_type` | string | yes | `"word"` or `"spelling"` |
| `interpolator_backend` | string | **yes** | `"srilm"` or `"kenlm"`. See [Multi-Corpus Interpolation](#multi-corpus-interpolation) for the tradeoff. No default. |
| `memory` | string | no | Memory limit for lmplz (and for the KenLM interpolate sort buffer). Default: `"80%"`. Can be `"80%"` or `"8G"`. |
| `normalize_workers` | int | no | Number of worker processes used during text normalization. Default: auto (`min(cpu_count, 16)`). Set to `1` for deterministic serial output (byte-identical line order, useful for regression diffs). Parallel normalization is ~15-20× faster on multi-GB corpora; the unordered chunk output does not affect KenLM n-gram counts. |
| `corpora` | list | yes | One or more corpus definitions (see below) |
| `lmplz_path` | string | no | Path to `lmplz` binary (default: `"lmplz"`) |
| `build_binary_path` | string | no | Path to `build_binary` binary (default: `"build_binary"`) |
| `srilm_ngram_path` | string | no | Path to SRILM `ngram` binary (default: `"ngram"`). Only required when `interpolator_backend: srilm` and there is more than one corpus. |
| `kenlm_interpolate_path` | string | no | Path to KenLM `interpolate` binary (default: `"interpolate"`). Only required when `interpolator_backend: kenlm` and there is more than one corpus. |
| `cmu_dict_path` | string | word only | Path to a CMU pronunciation dictionary (`.pkl` or text format). When omitted (or null), defaults to the `cmu_dict.pkl` bundled with the `phoneme_to_words_lm` package. |
| `english_words_path` | string | no | Path to a lowercase word list (one word per line). When set, OOV words must be in this list to be phonemized by g2p-en; otherwise they're dropped from the lexicon. When unset (null), every OOV word is dropped — the lexicon is restricted to the CMU-dict intersection of the corpus vocab, with no g2p-en fallback. Word mode only. |

### Corpus fields

Each entry in `corpora` has:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | see below | Human-readable label used for intermediate filenames (`normalized/<name>.txt`, `arpa/<name>.arpa` or `intermediate/<name>.*`). Defaults to the source file's basename (without extension). Must match `[A-Za-z0-9][A-Za-z0-9_.-]*` and be unique across corpora. **Required** when `path` or `normalized_path` is a list. |
| `path` | string \| list[string] | see below | Raw text corpus (one sentence per line). May be a list of files; each is normalized individually and the results are concatenated into one per-corpus model. |
| `normalized_path` | string \| list[string] | see below | Pre-normalized text file(s). May be a list, which are concatenated. |
| `arpa_path` | string | see below | Existing `.arpa` file (skip normalization + training). Always single-file. **Only valid when `interpolator_backend: srilm`.** |
| `intermediate_path` | string | see below | Existing lmplz `--intermediate` prefix. Refers to the set of files `<prefix>.1`, `<prefix>.2`, ..., `<prefix>.vocab`, `<prefix>.kenlm_intermediate` produced by a prior run. **Only valid when `interpolator_backend: kenlm`.** |
| `order` | int | yes | N-gram order for this corpus (1-10) |
| `weight` | number | multi-corpus | Interpolation weight (unnormalized, auto-scaled to sum to 1.0) |
| `pruning` | list[int] | no | Per-order count pruning thresholds. Length must match `order`. |

**Source priority**: `arpa_path` > `intermediate_path` > `normalized_path` > `path`. At least one must be provided. When `name` is omitted, it is derived from the highest-priority source's basename in the same order. `arpa_path` and `intermediate_path` are mutually exclusive with the chosen backend — specifying `arpa_path` in `kenlm` mode (or `intermediate_path` in `srilm` mode) is a validation error.

**Grouping multiple files into one ARPA**: providing a list for `path` or `normalized_path` concatenates the normalized text from all files into a single per-corpus ARPA. Use this to merge related sub-corpora (e.g., multiple conversational sources) into one model before interpolation. The group's `weight` applies to the merged ARPA; each file's contribution within the group is proportional to its token count, so duplicate input files to upweight them within the group.

## LM Types

### Word mode (`lm_type: word`)

Standard word-level n-gram. The training corpus is plain text (one sentence per line). After normalization, each line is a sequence of words. The lexicon maps words to CMU pronunciation dictionary phoneme sequences. Words not in CMU dict are optionally phonemized using g2p-en as a fallback — see `english_words_path` below.

OOV handling only affects the lexicon; the n-gram LM itself is always trained on the full normalized corpus, and sentences are never skipped. If `english_words_path` is configured, an OOV word must appear in that lowercase word list before it is sent to g2p-en — words that fail the check are dropped from the lexicon entirely. This filters out misspellings, concatenated tokens (e.g. `abouttechnologiesand`), and other garbage strings that would otherwise pollute the decoder vocabulary. If `english_words_path` is unset (null), every OOV word is dropped from the lexicon with no g2p-en fallback, restricting the decoder vocabulary to CMU-dict words only. The final log line reports per-bucket counts: `cmu`, `g2p`, `filtered` (failed the word-list check, or no word list set), and `skipped` (g2p produced no phonemes). OOV words that were kept (via g2p-en) are saved to `oov_g2p.txt`, and rejected words are saved to `rejected_oov.txt` with a per-word reason tag.

### Spelling mode (`lm_type: spelling`)

Character-level n-gram for letter-by-letter spelling. The training corpus is still plain text, but during normalization each word is expanded to space-separated characters, one word per line:

```
"hello world" -> "h e l l o"     (line 1)
                 "w o r l d"     (line 2)
```

Lines are deduplicated so common short words (e.g., "i" -> "i") don't dominate the training data. The lexicon maps 26 lowercase letters to their spoken-letter-name phoneme sequences (e.g., `a\tEY SIL`, `b\tB IY SIL`).

## Pruning

KenLM uses **count-based pruning** via `lmplz --prune`. Each value is a minimum count threshold: n-grams observed fewer times than the threshold are discarded.

| Value | Meaning |
|-------|---------|
| `0` | Keep all n-grams (no pruning) |
| `1` | Keep all (count is always >= 1, so same as 0) |
| `2` | Discard n-grams that appeared only once (hapax legomena) |
| `3` | Discard n-grams appearing fewer than 3 times |
| ... | etc. |

The pruning list has one value per n-gram order. The first value (unigrams) **must be 0** — unigrams cannot be pruned.

**Example**: `pruning: [0, 0, 1, 1, 2]` for a 5-gram model:
- Unigrams: keep all (required)
- Bigrams: keep all
- Trigrams: keep all (threshold 1 = keep everything with count >= 1)
- 4-grams: keep all (same)
- 5-grams: discard singletons (keep only count >= 2)

Higher-order n-grams are typically pruned more aggressively because they are sparser and more likely to be noise. Pruning can reduce model size by 10-50x for large models with little impact on perplexity.

This is different from probability-based pruning (as in some Kaldi recipes). KenLM's approach is simpler: it just counts how many times each n-gram was observed in the training data.

## Multi-Corpus Interpolation

When multiple corpora are specified, each is trained as a separate n-gram model, then merged using the configured `interpolator_backend`.

**Key points:**
- **Weights are unnormalized.** Specify `weight: 1, 2, 5` etc. and they are auto-scaled to sum to 1.0.
- **Different orders are OK.** A 5-gram general model can be interpolated with a 3-gram domain model. The merged model will have the maximum order.
- **Different pruning is OK.** Each corpus can have its own pruning thresholds.

Both backends perform probability-space linear interpolation: `P(w|h) = λ₁P₁(w|h) + λ₂P₂(w|h) + ...`. This is superior to simple corpus concatenation because it preserves the probability distributions of each model.

### Choosing a backend

| | `srilm` | `kenlm` |
|---|---|---|
| Per-corpus artifact | ARPA (`arpa/<name>.arpa`) | lmplz `--intermediate` files (`intermediate/<name>.*`) |
| How it interpolates | Loads every ARPA trie into RAM simultaneously | Streams the intermediate files through a disk-sorted merge |
| Peak RAM | ≳ sum of all per-corpus ARPAs, in-memory | ≈ `memory` (sort buffer) |
| Scales to 100+ GB of total sub-model | No (OOM) | Yes |
| External binary | `ngram` from SRILM | `interpolate` from KenLM (built separately; see prerequisites) |

**When to pick `kenlm`**: many corpora, very large corpora (e.g. OpenWebText, Wikipedia, OpenSubtitles), or any case where SRILM is running out of RAM. This is the recommended backend for serious multi-corpus mixtures.

**When to pick `srilm`**: small/medium mixtures that comfortably fit in memory, or when you want continuity with existing artifacts — SRILM will reuse per-corpus ARPAs from previous runs via `arpa_path`, while KenLM's intermediates are a separate set of files.

### Switching backends

The two backends use **different per-corpus artifact formats** (ARPA vs intermediate) and they are not interconvertible. Switching backends therefore requires re-running `lmplz` for every corpus. You can still short-circuit normalization by pointing `normalized_path` at the existing `normalized/<name>.txt` files from a previous run.

## Reusing Artifacts

All intermediate files are saved to the output directory. Exactly which per-corpus subdirectory appears depends on the backend:

```
output_dir/
├── normalized/<corpus_name>.txt           # normalized text files (one per corpus)
├── arpa/<corpus_name>.arpa                # per-corpus ARPA files  (srilm backend)
├── intermediate/<corpus_name>.{1..N,vocab,kenlm_intermediate}
│                                          # per-corpus intermediate files (kenlm backend)
├── tmp_interpolate/                       # KenLM interpolate's scratch sort dir (kenlm backend)
├── lm.arpa                    # final (interpolated) ARPA
├── lm_unpruned.bin            # compiled binary
├── lexicon.txt
├── oov_g2p.txt                # OOV words with g2p-en pronunciations (word mode only)
├── rejected_oov.txt           # OOV words dropped from the lexicon, with reason (word mode only)
├── log.txt                    # full run log
└── tokens.txt
```

You can reuse these in a later run:

**Reuse a normalized text file** (skip normalization, re-train):
```yaml
corpora:
  - normalized_path: /path/to/old_model/normalized/corpus_0.txt
    weight: 5
    order: 5
    pruning: [0, 0, 1, 1, 2]
```

**Reuse an ARPA file** (`srilm` backend; skip normalization + training, just re-interpolate):
```yaml
interpolator_backend: srilm
corpora:
  - arpa_path: /path/to/old_model/arpa/corpus_0.arpa
    weight: 5
    order: 5
  - path: /path/to/new_domain_corpus.txt
    weight: 1
    order: 3
```

**Reuse KenLM intermediate files** (`kenlm` backend; skip normalization + training). `intermediate_path` is a file prefix — lmplz wrote `<prefix>.1`, `<prefix>.2`, ..., `<prefix>.vocab`, and `<prefix>.kenlm_intermediate` alongside it:
```yaml
interpolator_backend: kenlm
corpora:
  - intermediate_path: /path/to/old_model/intermediate/corpus_0
    weight: 5
    order: 5
  - path: /path/to/new_domain_corpus.txt
    weight: 1
    order: 3
```

This is useful for re-weighting existing models or adding a new domain corpus to an existing general model without re-training from scratch.

## Text Normalization Pipeline

All raw text corpora go through these normalization steps (implemented in `text_normalize.py`):

**Pre-processing (per input line):**

1. **Decode HTML entities**: `&amp;` -> `&`, `&#39;` -> `'`, etc.
2. **Normalize Unicode punctuation**: smart/curly quotes -> straight, em/en dashes -> spaces, ellipsis -> period. Also: `&` -> `" and "`; spaced honorifics collapsed (`"Mr . Brown"` -> `"Mr. Brown"`) so the sentence tokenizer doesn't break on them; orphan apostrophes from spaced smart-quote contractions re-joined (`"I ' m"` -> `"I'm"`).
3. **Remove URLs and emails**: strips `http://...`, `www.`, and email addresses.
4. **Sentence split**: uses NLTK `sent_tokenize` to split multi-sentence lines. One input line may produce multiple output sentences, giving KenLM proper `<s>`/`</s>` sentence boundaries.

**Per-sentence normalization:**

5. **Expand ordinals**: `"1st"` -> `"first"`, `"23rd"` -> `"twenty third"`.
6. **Expand decades**: `"1990s"` -> `"nineteen nineties"`.
7. **Numbers to words**: digits adjacent to letters are split with a space first (`"100mg"` -> `"100 mg"`, `"covid19"` -> `"covid 19"`), so embedded numbers are also expanded. Standalone numbers are converted via `num2words`. Four-digit integers in 1000-2099 are read as years (`"1992"` -> `"nineteen ninety two"` rather than `"one thousand nine hundred ninety two"`). Hyphens and commas in `num2words` output are cleaned (e.g., `"42"` -> `"forty two"`, not `"forty-two"`).
8. **Unicode normalization**: NFKD decomposition strips diacritics so accented characters survive as base letters (`"cafe"` with accent -> `"cafe"`).
9. **Remove punctuation**: hyphens, slashes, periods, pipes, and brackets -> spaces (so `"well-known"` -> `"well known"`, `"Dr.Sara"` -> `"dr sara"`, `"1/2"` -> `"1 2"`, `"[[link|thumb]]"` -> `"link thumb"`), strips all non-alphabetic characters except apostrophes, lowercases, collapses whitespace. Runs of adjacent apostrophes are collapsed (`"''economic"` -> `"'economic"`) and apostrophes at word boundaries — which come from quote marks wrapping words, e.g. `"'Beatles'"` or `"'climbing"` — are stripped, while internal contraction apostrophes are preserved (`"don't"`, `"it's"`, `"o'clock"`).
10. **Replace words**: normalizes British spellings to American (e.g., `"colour"` -> `"color"`), expands contractions (e.g., `"dont"` -> `"don't"`), and replaces abbreviations (e.g., `"mr"` -> `"mister"`). Uses `replace_words()` from `phoneme_to_words_lm.utils`.
11. **Filter empty lines**: blank sentences after normalization are dropped.

## Output Files

| File | Description |
|------|-------------|
| `lm.arpa` | Final ARPA-format language model (merged if multi-corpus). Kept for inspection and reuse. |
| `lm_unpruned.bin` | KenLM binary in trie format, loaded by `KenLMFlashlightTextLM` at runtime. |
| `lexicon.txt` | Word-to-phoneme mappings. Tab-separated: `WORD\tP1 P2 ... SIL`. Words with multiple pronunciations have multiple lines. |
| `oov_g2p.txt` | OOV words not in CMU dict that were phonemized by g2p-en, with their phoneme sequences. Tab-separated: `WORD\tP1 P2 ...` (word mode only). |
| `rejected_oov.txt` | OOV words *not* included in the lexicon, with the reason (`filtered` = not in the english word list, `g2p_failed` = g2p produced no valid phonemes). Tab-separated: `WORD\tREASON` (word mode only). |
| `log.txt` | Full run log (same content as stderr output). Overwritten on each run. |
| `tokens.txt` | Token list: BLANK, SIL, then 39 phonemes in alphabetical order (41 lines total). Static across all LMs. |

## Examples

### Word-level 5-gram with two corpora (SRILM)

```yaml
output_dir: ~/lm_output/general_domain_5gram
lm_type: word
interpolator_backend: srilm
memory: "80%"
corpora:
  - path: /data/openwebtext_sentences.txt
    weight: 5
    order: 5
    pruning: [0, 0, 1, 1, 2]
  - path: /data/participant_transcripts.txt
    weight: 1
    order: 3
    pruning: null
lmplz_path: /opt/vcpkg/installed/x64-linux/tools/kenlm/lmplz
build_binary_path: /opt/vcpkg/installed/x64-linux/tools/kenlm/build_binary
srilm_ngram_path: /opt/srilm/bin/ngram
```

### Large multi-corpus mixture (KenLM)

Use this shape when per-corpus ARPAs would collectively exceed available RAM.

```yaml
output_dir: ~/lm_output/large_mixture
lm_type: word
interpolator_backend: kenlm
memory: "80%"
corpora:
  - path: /data/openwebtext_sentences.txt
    weight: 5
    order: 5
    pruning: [0, 0, 1, 1, 2]
  - path: /data/wikipedia_sentences.txt
    weight: 4
    order: 5
    pruning: [0, 0, 1, 1, 2]
  - path: /data/opensubs_sentences.txt
    weight: 3
    order: 5
    pruning: [0, 0, 1, 1, 2]
  - path: /data/participant_transcripts.txt
    weight: 1
    order: 5
    pruning: null
lmplz_path: /opt/vcpkg/installed/x64-linux/tools/kenlm/lmplz
build_binary_path: /opt/vcpkg/installed/x64-linux/tools/kenlm/build_binary
kenlm_interpolate_path: /path/to/kenlm/build/bin/interpolate
```

### Spelling-mode 5-gram

```yaml
output_dir: ~/lm_output/spelling_5gram
lm_type: spelling
interpolator_backend: srilm
memory: "50%"
corpora:
  - path: /data/english_word_list.txt
    weight: 1
    order: 5
    pruning: null
lmplz_path: lmplz
build_binary_path: build_binary
```

### Re-weight existing models

```yaml
output_dir: ~/lm_output/reweighted_model
lm_type: word
interpolator_backend: srilm
memory: "80%"
corpora:
  - arpa_path: /path/to/general_5gram/arpa/corpus_0.arpa
    weight: 3
    order: 5
  - arpa_path: /path/to/domain_3gram/arpa/corpus_0.arpa
    weight: 2
    order: 3
lmplz_path: lmplz
build_binary_path: build_binary
srilm_ngram_path: ngram
```

Note: when reusing ARPA files, `cmu_dict_path` is still used (for lexicon generation); it falls back to the bundled CMU dict if you don't set it. If you also want the lexicon to cover all the words from the original training corpora, provide the original `path` or `normalized_path` alongside the `arpa_path` so vocab can be scanned. Otherwise, the lexicon will only include words from corpora that have a scannable text source.
