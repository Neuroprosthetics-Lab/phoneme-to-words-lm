# Repeatable decoder validation and development testing

Run from the repository root in the decoder environment. Copy
`benchmarks/example_evaluation.yaml`, set the logits/resource paths and choose
the configurations to compare. Models must already exist locally or in the HF
cache. No downloads or training occur during evaluation.

Normalization policy `english_v2` preserves ambiguous words (`lets`, `cant`,
`wont`, `masters`, `masters'`) and handles Unicode apostrophes/Latin accents.
The runtime manifest records this policy. Use a new output directory and
recompute splits, candidate scores, edits and tuning after migrating; old
normalized references may have already lost distinctions that cannot be recovered.

```bash
conda activate b2txt_pt
python -B benchmarks/run_evaluation.py --config my_evaluation.yaml
```

Change datasets without editing the YAML:

```bash
python -B benchmarks/run_evaluation.py --config my_evaluation.yaml \
  --validation-logits /data/new_validation.pkl \
  --test-logits /data/new_test.pkl \
  --output benchmark_runs/new_comparison --device cuda:0
```

The current workstation configuration is `benchmark_runs/g_t15_t22_accuracy1000.yaml`: t15
validation, t22 provisional development test. Both source filenames are upstream
validation exports. Treat t22 results as development evidence, not production
generalization or permission to repeatedly tune against that test set.

## What the workflow does

1. Validate settings and data; save source indices, split rules and resource,
   model, runtime, hardware and code identities.
2. Screen every named configuration on the same randomly selected validation
   subset. Tune LLM alpha and length penalty using raw LLM sums and exact target
   counts. The named baseline has `tune: false`, preserving its original weights.
3. Evaluate finalists on full validation and retune their scoring weights there.
   Keep the baseline, best weighted candidate, best accuracy candidate and fastest
   candidate on the accuracy/latency Pareto frontier, filling any remaining slots
   by weighted score. `finalists: null` evaluates every candidate on full validation.
4. Freeze the recommended, accuracy, fastest and baseline configurations in
   `selection.json`. Overlapping roles share one test run.
5. Evaluate those fixed configurations on test data. Test labels never choose
   alpha, penalty, model or recommendation.

The default comparison set is bounded and explicit; it does not exhaustively
search every combination of decoder settings. Add named `overrides` to test
combined settings, adapters, contexts, n-best counts, other resource files or
other supported decoder parameters. For example:

```yaml
configurations:
  - {name: baseline, tune: false}
  - {name: retuned}
  - name: smaller_and_faster
    overrides:
      llm_model_name: Qwen/Qwen3.5-2B
      beam_size: 2500
      n_best: 50
```

Include an explicit `do_llm_rescoring: false` case to measure n-gram-only decoding.
LLM-enabled cases still compute LLM scores when alpha is zero; their measured
cost and exported configuration reflect that behavior. There is no automatic
conversion to a different execution path based on a tuned alpha.

## Accuracy/latency weighting

```text
score = wer_weight × WER_fraction + latency_weight × mean_seconds
```

With `{wer: 1000, latency: 1}`, a 0.1 percentage-point WER reduction is worth
one second. Thus 1.1% WER at 100 ms scores **11.1**, whereas 1.0% at one second
scores **11.0**. The latter wins under the revised accuracy preference.
Lower is better. No rolling or min-max normalization is applied. The report
always shows WER and latency separately, plus accuracy and speed choices.
Change weights in a new experiment if this tradeoff no longer fits your use case.

## Datasets and splits

Input is the same list-of-records pickle supported by the sweep worker:

- `logits`: numeric `(T, C)` array.
- `adjusted_len` or `adjusted_lens`: valid frame count; both must agree if present.
- `transcription`: text or zero-terminated integer character codes.
- Optional `context`: string, forwarded unchanged to LLM conditioning.
- Optional metadata such as `day_index` and `participant_id` for grouping.

References use the existing lowercase/punctuation/word normalization. Empty and
letter-by-letter spelling references are excluded; all counts and indices are
saved. Set `reorder_logit_columns: false` if inputs already follow tokens.txt;
the default reorders `[BLANK, phones, SIL]` to `[BLANK, SIL, phones]`.

For one-file splitting, set `test_logits: null` or use:

```bash
python -B benchmarks/run_evaluation.py --config my_evaluation.yaml \
  --validation-logits /data/all_logits.pkl --split-test-fraction 0.2 \
  --output benchmark_runs/grouped_split
```

`split_group: reference` keeps repeated normalized prompts together.
`split_group: day_index` also keeps days together, joining days that share a
reference. Fractions apply to independent groups, so utterance proportions may
differ. Fewer than two remaining independent groups fails explicitly. Separate
files are audited for shared reference identities; duplicates are reported, not
silently removed. Identical files cannot masquerade as separate val/test sets.

`bootstrap_group` controls uncertainty estimates independently of splitting.
Use `day_index` for recordings grouped by day, or `utterance` when that metadata
is unavailable. Group keys include participant ID when provided. The report uses
2,000 seeded resamples by default and gives paired test WER differences against
the baseline. Few groups produce weak uncertainty estimates; one group produces
no interval. These intervals do not remove selection bias or domain mismatch.

## Runtime, resume and staged evaluation

```bash
# Check settings/splits without allocating models or decoding.
python -B benchmarks/run_evaluation.py --config my_evaluation.yaml --dry-run

# Tune and freeze choices, leaving test scores unseen.
python -B benchmarks/run_evaluation.py --config my_evaluation.yaml --phase validation

# Later, evaluate only those frozen choices.
python -B benchmarks/run_evaluation.py --config my_evaluation.yaml --phase test --resume

# Recover interrupted work, reusing only completed, verified measurements.
python -B benchmarks/run_evaluation.py --config my_evaluation.yaml --resume

# Rebuild reports from saved measurements; no model, GPU or logits loading.
python -B benchmarks/run_evaluation.py --config my_evaluation.yaml --phase report
```

`--screen-limit`, `--validation-limit` and `--test-limit` override evaluation
sizes; `0` means all eligible records. Use a fresh output directory when changing
limits, settings, data, models, runtime, hardware or source. Resume verifies
content identities and completed output hashes. Hashing large model files has
startup I/O cost. During a run, keep inputs/models/source files unchanged.

Every configuration runs in its own subprocess, serially, with a timeout and a
saved log. Optional candidate failures are reported; a failed baseline or frozen
test run stops the workflow. Resume retries incomplete jobs. Reusing completed
measurements is experiment recovery; it adds no runtime LLM score cache.

Screening latency comes from the accuracy pass after separate warmup calls.
Final validation/test latency uses the same seeded subset for each configuration,
with configurable `timing_limit` and `timing_repeats`. GPU work is synchronized at
the decoder boundary. Timing includes preprocessing/search/extraction/rescoring
inside `offline_decode`, but excludes pickle loading, input-column reordering,
edit-distance evaluation, model loading and warmup. Detailed load/first-call
times remain available. Avoid other workloads on the measured CPU/GPU.

## Outputs

| Artifact | Contents |
|---|---|
| `report.md` | Readable screening/validation/test tables and paired test intervals |
| `report.csv` | Flat comparison table for analysis tools |
| `report.json` | Complete metrics, per-group WER, failures and frozen selection |
| `accuracy_latency.svg` | Standalone validation/test comparison plot, if matplotlib is installed |
| `manifest.json`, `split.json` | Identities, hardware, settings and exact record indices |
| `selection.json` | Validation-only choices and their immutable scoring settings |
| `selected_configs/*_decoder.json` | Complete decoder constructor kwargs per role |
| `selected_configs/*_sweep.yaml` | Equivalent fixed configuration for the existing sweep tool |
| `<stage>/<name>/` | Job config, worker log, candidates, scoring grid, timing samples, summary and completion hashes |

N-gram WER measures candidate generation's top choice; oracle WER is the lowest
word-edit count available anywhere in the n-best set; final WER measures selected
output. Their gaps help distinguish candidate-generation limits from rescoring
limits. OOV is measured against the lexicon, not KenLM's hidden vocabulary.
RAM is peak process RSS including load; VRAM is peak PyTorch allocated/reserved
memory, not total device memory. Model-load time and latency percentiles are
reported separately. Bootstrap intervals, raw scores and diagnostics are in JSON.

Load a selected configuration directly:

```python
import json
from phoneme_to_words_lm import KenLMFlashlightTextLM

with open('benchmark_runs/new_comparison/selected_configs/recommended_decoder.json') as stream:
    decoder = KenLMFlashlightTextLM(**json.load(stream))
```

The exported sweep YAML preserves decoder/scoring settings and input column
order. The existing sweep tool uses its own evaluation/timing conventions and
reads the configured logits file; it does not reproduce this harness's saved
subset or within-file split automatically. Use this harness and its manifest
when reproducing the reported numbers.

No package defaults are changed by a benchmark. Review the evidence, then select
the operating point appropriate to your deployment.
