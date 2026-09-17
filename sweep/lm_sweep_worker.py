"""LM decoder hyperparameter sweep worker.

Executed as a subprocess by lm_sweep.py. Load precomputed phoneme logits,
decode sentences, select alpha post-hoc from native beam totals, and write
results and candidate scores to JSON files.

Usage (standalone testing):
    python sweep/lm_sweep_worker.py --config worker_config.json
"""
import argparse
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Resolve cached snapshots without fetching changing remote weights in a trial.
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import editdistance
import numpy as np
import torch

from phoneme_to_words_lm import KenLMFlashlightTextLM
from phoneme_to_words_lm.utils import remove_punctuation, replace_words
from phoneme_to_words_lm.sweep_contract import alpha_values, positive_integer, resolve_decoder_config


def load_and_filter_logits(cfg):
    """Accept text or zero-terminated character IDs and either length spelling."""
    every = positive_integer('eval_every_nth', cfg.get('eval_every_nth', 1))
    limit = cfg.get('max_sentences')
    if limit is not None:
        positive_integer('max_sentences', limit)
    with open(cfg['logits_pkl_path'], 'rb') as stream:
        rows = pickle.load(stream)
    filtered = []
    for index, row in enumerate(rows):
        ref = row['transcription']
        if not isinstance(ref, str):
            codes = np.asarray(ref)
            if codes.ndim != 1 or codes.dtype.kind not in 'iu':
                raise ValueError(f'Trial {index}: transcription must be text or integer character IDs')
            end = int(np.flatnonzero(codes == 0)[0]) if (codes == 0).any() else len(codes)
            ref = ''.join(chr(int(code)) for code in codes[:end])
        ref = ' '.join(replace_words(remove_punctuation(ref)).split())
        if not ref or max(map(len, ref.split())) == 1:
            continue
        length = row.get('adjusted_len', row.get('adjusted_lens'))
        if isinstance(length, (bool, np.bool_)) or not isinstance(length, (int, np.integer)):
            raise ValueError(f'Trial {index}: integer adjusted_len (or adjusted_lens) required')
        if 'adjusted_len' in row and 'adjusted_lens' in row and row['adjusted_len'] != row['adjusted_lens']:
            raise ValueError(f'Trial {index}: conflicting length fields')
        logits = np.asarray(row['logits'])
        if logits.ndim != 2 or logits.dtype.kind not in 'fi' or not 0 <= length <= len(logits):
            raise ValueError(f'Trial {index}: invalid logits shape/type/length')
        if not np.isfinite(logits[:length]).all():
            raise ValueError(f'Trial {index}: nonfinite valid logits')
        context = row.get('context')
        if context is not None and not isinstance(context, str):
            raise ValueError(f'Trial {index}: context must be a string or null')
        filtered.append(dict(row, transcription=ref, adjusted_len=int(length), context=context,
                             logits=logits, source_index=index))
    selected = filtered[::every]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError('No usable sentences after filtering/subsampling')
    return selected


def decode_sentences(decoder, sentences, reorder_logit_columns):
    """Decode all sentences and retain scores, metadata, and timing."""
    results = []
    for sent in sentences:
        logits = torch.as_tensor(sent['logits'], dtype=torch.float32).unsqueeze(0)
        if reorder_logit_columns:
            # [BLANK, ph1..phN, SIL] -> [BLANK, SIL, ph1..phN]
            logits = torch.cat(
                (logits[:, :, :1], logits[:, :, -1:], logits[:, :, 1:-1]), dim=-1,
            )
        lengths = torch.tensor([sent['adjusted_len']], dtype=torch.int64)
        context = sent.get('context')
        t = time.perf_counter()
        hypo = decoder.offline_decode(logits.contiguous(), lengths,
                                      contexts=[context] if context is not None else None)[0]
        elapsed = time.perf_counter()-t
        # Preserve all score fields and diagnostic metadata for later selection.
        result = dict(hypo, transcription=sent['transcription'], decode_time=elapsed,
                      source_index=sent['source_index'], context=context)
        result['edit_reference'] = sent['transcription']
        result['edit_counts_by_text'] = {text: int(editdistance.eval(sent['transcription'].split(), text.split()))
                                         for text in hypo['word_seqs']}
        results.append(result)
        if len(results) % 50 == 0 or len(results) == len(sentences):
            print(f'  Decoded {len(results)}/{len(sentences)}', flush=True)
    return results


def prepare_candidates(results, length_penalty=None):
    """Compute edit counts once, preserving the decoder's tie-breaking order."""
    if not results:
        raise ValueError('No sentences to evaluate')
    prepared = []
    for result in results:
        texts = result['word_seqs']
        n_words = len(result['transcription'].split())
        if not n_words or not texts:
            raise ValueError('Each result needs a nonempty reference and at least one candidate')
        if not any(text.strip() for text in texts):
            prepared.append(dict(empty=True, n_words=n_words, edits=np.array([n_words])))
            continue
        if 'beam_scores' not in result:
            raise ValueError('Post-hoc ranking requires native beam_scores; rerun legacy decodes')
        beam = np.asarray(result['beam_scores'], dtype=float)
        values = result.get('llm_scores', [])
        if length_penalty is not None:
            if 'raw_llm_scores' not in result:
                raise ValueError('Changing length penalty requires raw_llm_scores and llm_token_counts')
            values = result.get('raw_llm_scores', [])
        if not values or all(v is None for v in values):
            llm = np.zeros_like(beam)
        elif any(v is None for v in values):
            raise ValueError('Incomplete LLM scores for a nonempty result')
        else:
            llm = np.asarray(values, dtype=float)
            if length_penalty is not None:
                counts = np.asarray(result['llm_token_counts'])
                if counts.shape != beam.shape or counts.dtype.kind not in 'iu' or (counts < 0).any():
                    raise ValueError('Target token counts must be nonnegative integers aligned to candidates')
                llm = llm - length_penalty*counts
        if (
            beam.shape != (len(texts),)
            or llm.shape != beam.shape
            or not np.isfinite(beam).all()
            or not np.isfinite(llm).all()
        ):
            raise ValueError('Candidate scores must be finite and aligned')
        if result.get('edit_reference') != result['transcription']:
            result['edit_reference'] = result['transcription']
            result['edit_counts_by_text'] = {}
        cache = result.setdefault('edit_counts_by_text', {})
        for text in texts:
            if text not in cache:
                cache[text] = int(editdistance.eval(result['transcription'].split(), text.split()))
        edits = np.asarray([cache[text] for text in texts])
        if edits.shape != beam.shape or edits.dtype.kind not in 'iu' or (edits < 0).any():
            raise ValueError('Candidate edit counts must be nonnegative aligned integers')
        order = sorted(range(len(beam)), key=lambda j: (-beam[j], texts[j]))
        prepared.append(dict(empty=False, beam=beam[order], llm=llm[order],
                             n_words=n_words, edits=edits[order]))
    return prepared


def alpha_posthoc_sweep(all_results, lm_weight, alpha_low, alpha_high, alpha_step, *, length_penalty=None):
    """Select alpha using native totals; optionally reuse raw sums for a new penalty.

    lm_weight is retained for compatibility; changing search weights requires
    another decode. Candidate edit counts are reused across alpha/penalty runs.
    """
    if length_penalty is not None and not math.isfinite(length_penalty):
        raise ValueError('length_penalty must be finite')
    alphas = alpha_values(alpha_low, alpha_high, alpha_step)
    prepared = prepare_candidates(all_results, length_penalty)
    n_words = sum(p['n_words'] for p in prepared)
    curve, counts = [], []
    for alpha in alphas:
        errors = 0
        for candidate in prepared:
            if candidate['empty']:
                best_index = 0
            else:
                final_scores = candidate['beam'] + alpha * candidate['llm']
                best_index = np.argmax(final_scores)
            errors += int(candidate['edits'][best_index])
        curve.append([alpha, errors/n_words])
        counts.append(errors)
    best_errors = min(counts)
    tied = [a for a, errors in zip(alphas, counts) if errors == best_errors]
    return tied[len(tied)//2], best_errors/n_words, best_errors, n_words, curve


def evaluation_metrics(results, decoder, warmup_sentences=1):
    positive_integer('timing_warmup_sentences', warmup_sentences, allow_zero=True)
    prepared = prepare_candidates(results)
    words = sum(p['n_words'] for p in prepared)
    excluded = min(warmup_sentences, len(results)-1)
    times = np.asarray([r['decode_time'] for r in results[excluded:]])
    vocab = decoder._hotword_lookup  # Case-normalized lexicon membership, not KenLM's hidden vocabulary.
    oov = sum(w.casefold() not in vocab for r in results for w in r['transcription'].split())
    return dict(ngram_wer=sum(int(p['edits'][0]) for p in prepared)/words,
                oracle_wer=sum(int(p['edits'].min()) for p in prepared)/words,
                empty_rate=sum(p['empty'] for p in prepared)/len(results),
                median_candidates=float(np.median([len(r['word_seqs']) for r in results])),
                reference_oov_rate=oov/words,
                avg_decode_time=float(times.mean()), p50_decode_time=float(np.median(times)),
                p95_decode_time=float(np.percentile(times, 95)),
                sentences_per_second=float(1/times.mean()),
                timing_warmup_sentences=excluded, first_decode_seconds=results[0]['decode_time'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    effective = resolve_decoder_config(cfg)
    alphas = (cfg.get('alpha_low', 0.), cfg.get('alpha_high', 3.), cfg.get('alpha_step', .05))
    alpha_values(*alphas)  # Validate before loading models or allocating CUDA.
    warmup = positive_integer('timing_warmup_sentences', cfg.get('timing_warmup_sentences', 1), allow_zero=True)
    torch.set_num_threads(1)
    started = time.perf_counter()
    sentences = load_and_filter_logits(cfg)
    fraction = cfg.get('max_vram_fraction')
    if fraction is not None and effective['do_llm_rescoring']:
        if not isinstance(fraction, (float, int)) or not 0 < fraction <= 1:
            raise ValueError('max_vram_fraction must be in (0,1]')
        torch.cuda.set_per_process_memory_fraction(fraction, torch.device(effective['llm_device']))
    decoder = KenLMFlashlightTextLM(**effective)
    load_seconds = time.perf_counter()-started
    rows = decode_sentences(decoder, sentences, cfg.get('reorder_logit_columns', True))
    alpha, wer, errors, words, curve = alpha_posthoc_sweep(rows, effective['lm_weight'], *alphas)
    results = dict(wer=wer, best_alpha=alpha, n_edits_total=errors, n_words_total=words,
                   n_sentences_evaluated=len(rows), alpha_wer_curve=curve,
                   effective_decoder_config=effective, source_indices=[r['source_index'] for r in rows],
                   hotword_backend=decoder.hotword_backend, load_seconds=load_seconds,
                   metric_version='sweep_v2', **evaluation_metrics(rows, decoder, warmup))
    output = Path(cfg['output_path'])
    output.parent.mkdir(parents=True, exist_ok=True)
    for path, value in ((output.with_name('candidates.json'), rows), (output, results)):
        temporary = path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(value, indent=2))
        temporary.replace(path)
    print('RESULTS', json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
