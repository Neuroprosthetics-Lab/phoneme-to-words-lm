"""Validation-only selection and paired evaluation statistics (no model cache)."""
from collections import defaultdict
import hashlib
import math

import numpy as np


def objective(wer, seconds, weights):
    """WER is a fraction; latency is seconds per utterance."""
    return weights['wer'] * wer + weights['latency'] * seconds


def subset_indices(rows, limit, seed):
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError('Subset limit must be a positive integer or null')
    if limit is None or limit >= len(rows):
        return [r['source_index'] for r in rows]
    chosen = sorted(np.random.default_rng(seed).choice(len(rows), limit, replace=False))
    return [rows[i]['source_index'] for i in chosen]


def group_key(row, field):
    if field == 'reference':
        return hashlib.sha256(row['transcription'].encode()).hexdigest()
    if field == 'utterance':
        return str(row['source_index'])
    if field not in row or row[field] is None:
        raise ValueError(f'Missing grouping field {field!r}; use reference or utterance explicitly')
    return f"{row.get('participant_id', '')}:{row[field]}"


def split_indices(rows, fraction, seed, group='reference'):
    """Keep groups and duplicate normalized references together within one file."""
    if isinstance(fraction, bool) or not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError('test_fraction must be between zero and one')
    # Join groups sharing a reference so repeated prompts cannot leak across a split.
    parent = {}

    def find(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    references = {}
    keys = []
    for row in rows:
        key = group_key(row, group)
        keys.append(key)
        root = find(key)
        ref = row['transcription']
        if ref in references:
            parent[root] = find(references[ref])
        references[ref] = key
    components = sorted({find(key) for key in keys})
    if len(components) < 2:
        raise ValueError('Need at least two independent groups after joining repeated references')
    np.random.default_rng(seed).shuffle(components)
    n_test = min(len(components)-1, max(1, round(len(components)*fraction)))
    test_groups = set(components[:n_test])
    validation, test = [], []
    for row, key in zip(rows, keys):
        if find(key) in test_groups:
            test.append(row['source_index'])
        else:
            validation.append(row['source_index'])
    return validation, test


def prepare_ranking(rows):
    """Prepare arrays once; ties follow native beam score, then candidate text."""
    import editdistance
    arrays = []
    for row in rows:
        texts = row['word_seqs']
        words = len(row['transcription'].split())
        if not words or not texts:
            raise ValueError('References and candidate lists must be nonempty')
        if not any(text.strip() for text in texts):
            arrays.append(dict(empty=True, words=words, edits=np.array([words]), order=[0]))
            continue
        order = sorted(range(len(texts)), key=lambda j: (-row['beam_scores'][j], texts[j]))
        beam = np.asarray(row['beam_scores'], dtype=float)[order]
        raw_values = row['raw_llm_scores']
        if all(value is None for value in raw_values):
            raw = np.zeros(len(texts))
        elif any(value is None for value in raw_values):
            raise ValueError('Partially missing LLM scores')
        else:
            raw = np.asarray(raw_values, dtype=float)[order]
        counts = np.asarray(row['llm_token_counts'])[order]
        if (beam.shape != (len(texts),) or raw.shape != beam.shape or counts.shape != beam.shape
                or not np.isfinite(beam).all() or not np.isfinite(raw).all()
                or counts.dtype.kind not in 'iu' or (counts < 0).any()):
            raise ValueError('Invalid candidate scores or target counts')
        cached = {}
        if row.get('edit_reference') == row['transcription']:
            cached = row.get('edit_counts_by_text', {})
        edits = []
        for j in order:
            text = texts[j]
            if text in cached:
                edits.append(cached[text])
            else:
                edits.append(int(editdistance.eval(row['transcription'].split(), text.split())))
        arrays.append(dict(empty=False, words=words, edits=np.asarray(edits), order=order,
                           beam=beam, raw=raw, counts=counts))
    return arrays


def rank(arrays, alpha, penalty):
    if not math.isfinite(alpha) or alpha < 0 or not math.isfinite(penalty):
        raise ValueError('Invalid alpha or length penalty')
    edits, indices = [], []
    for row in arrays:
        if row['empty']:
            index = 0
        else:
            llm_scores = row['raw'] - penalty * row['counts']
            index = int(np.argmax(row['beam'] + alpha * llm_scores))
        edits.append(int(row['edits'][index]))
        indices.append(row['order'][index])
    return edits, indices


def select_weights(arrays, alphas, penalties, anchor=None):
    """Tune on validation only. Prefer the anchor policy when edit counts tie."""
    if anchor is None:
        from .sweep_contract import decoder_defaults
        defaults = decoder_defaults()
        anchor = (defaults['llm_alpha'], defaults['llm_length_penalty'])
    curve = []
    for penalty in penalties:
        for alpha in alphas:
            errors = sum(rank(arrays, alpha, penalty)[0])
            curve.append(dict(alpha=alpha, length_penalty=penalty, edits=errors))
    if not curve:
        raise ValueError('Empty scoring-policy grid')
    best = min(curve, key=lambda r: (r['edits'], abs(r['alpha']-anchor[0]),
                                     abs(r['length_penalty']-anchor[1]), r['alpha'], r['length_penalty']))
    return best, curve


def percentile_stats(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Timing samples must be finite and positive')
    return dict(mean_seconds=float(values.mean()), p50_seconds=float(np.median(values)),
                p95_seconds=float(np.percentile(values, 95)),
                utterances_per_second=float(1/values.mean()), samples=len(values))


def bootstrap_interval(errors, words, groups, samples=2000, seed=42):
    """Percentile interval by resampling whole groups, never individual words."""
    if len(errors) != len(words) or len(words) != len(groups) or samples < 1:
        raise ValueError('Bootstrap inputs must align and samples must be positive')
    sums = defaultdict(lambda: [0, 0])
    for error, count, group in zip(errors, words, groups):
        sums[group][0] += error
        sums[group][1] += count
    if not sums or sum(v[1] for v in sums.values()) <= 0:
        raise ValueError('Bootstrap requires reference words')
    if len(sums) < 2:
        return dict(low=None, high=None, groups=len(sums), samples=0)
    values = np.asarray(list(sums.values()), dtype=float)
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, samples, 256):
        draw = rng.integers(len(values), size=(min(256, samples-start), len(values)))
        totals = values[draw].sum(axis=1)
        estimates.extend((totals[:, 0]/totals[:, 1]).tolist())
    low, high = np.percentile(estimates, [2.5, 97.5])
    return dict(low=float(low), high=float(high), groups=len(sums), samples=samples)


def summarize(rows, arrays, alpha, penalty, timing, vocab, *, bootstrap_samples=2000, seed=42):
    errors, winners = rank(arrays, alpha, penalty)
    words = [row['words'] for row in arrays]
    groups = [row['evaluation_group'] for row in rows]
    group_totals = defaultdict(lambda: dict(utterances=0, words=0, edits=0))
    for row, error, count, winner in zip(rows, errors, words, winners):
        row.update(selected_text=row['word_seqs'][winner], selected_edits=error, n_words=count)
        entry = group_totals[row['evaluation_group']]
        entry['utterances'] += 1
        entry['words'] += count
        entry['edits'] += error
    for entry in group_totals.values():
        entry['wer'] = entry['edits']/entry['words']
    total_words = sum(words)
    total_edits = sum(errors)
    oov_words = sum(
        word.casefold() not in vocab
        for row in rows for word in row['transcription'].split()
    )
    search_frames = sum(row['search_frames'] for row in rows)
    total_frames = sum(row['total_frames'] for row in rows)
    return {
        'utterances': len(rows),
        'words': total_words,
        'edits': total_edits,
        'wer': total_edits / total_words,
        'alpha': alpha,
        'length_penalty': penalty,
        'ngram_wer': sum(int(row['edits'][0]) for row in arrays) / total_words,
        'oracle_wer': sum(int(row['edits'].min()) for row in arrays) / total_words,
        'empty_rate': sum(row['empty'] for row in arrays) / len(rows),
        'median_candidates': float(np.median([len(row['word_seqs']) for row in rows])),
        'reference_oov_rate': oov_words / total_words,
        'retained_frame_fraction': search_frames / max(1, total_frames),
        'timing': percentile_stats(timing),
        'wer_interval': bootstrap_interval(errors, words, groups, bootstrap_samples, seed),
        'by_group': dict(group_totals),
    }


def pareto_front(results):
    """Return nondominated configurations using validation WER and mean latency."""
    frontier = []
    for result in results:
        wer = result['wer']
        latency = result['timing']['mean_seconds']
        dominated = False
        for other in results:
            other_wer = other['wer']
            other_latency = other['timing']['mean_seconds']
            if (other_wer <= wer and other_latency <= latency
                    and (other_wer < wer or other_latency < latency)):
                dominated = True
                break
        if not dominated:
            frontier.append(result)
    return frontier


def choose_finalists(results, baseline, limit, weights):
    if any(r['stage'] not in ('screen', 'validation') for r in results):
        raise ValueError('Test results must never select configurations')
    ordered = sorted(results, key=lambda r: (objective(r['wer'], r['timing']['mean_seconds'], weights), r['name']))
    accurate = min(results, key=lambda r: (r['wer'], r['timing']['mean_seconds'], r['name']))
    # Minimum latency, breaking ties by WER, is already on the Pareto frontier.
    fastest = min(results, key=lambda r: (r['timing']['mean_seconds'], r['wer'], r['name']))
    if limit is None:
        return [r['name'] for r in ordered]
    chosen = []
    priority = [baseline, ordered[0]['name'], accurate['name'], fastest['name']]
    for name in priority + [r['name'] for r in ordered]:
        if name not in chosen:
            chosen.append(name)
        if len(chosen) == limit:
            break
    return chosen


def freeze_selection(results, baseline, weights):
    if any(r['stage'] != 'validation' for r in results):
        raise ValueError('Only full validation results can freeze test configurations')
    weighted = min(results, key=lambda r: (objective(r['wer'], r['timing']['mean_seconds'], weights), r['name']))
    accurate = min(results, key=lambda r: (r['wer'], r['timing']['mean_seconds'], r['name']))
    fast = min(results, key=lambda r: (r['timing']['mean_seconds'], r['wer'], r['name']))
    return dict(recommended=weighted['name'], accuracy=accurate['name'], fastest=fast['name'], baseline=baseline)


def paired_comparison(candidate, baseline, samples=2000, seed=42):
    if [r['source_index'] for r in candidate] != [r['source_index'] for r in baseline]:
        raise ValueError('Paired comparisons require identical ordered utterance indices')
    if any(a['transcription'] != b['transcription'] or a['evaluation_group'] != b['evaluation_group']
           for a, b in zip(candidate, baseline)):
        raise ValueError('Paired comparison references/groups differ')
    errors = [a['selected_edits']-b['selected_edits'] for a, b in zip(candidate, baseline)]
    words = [r['n_words'] for r in candidate]
    return dict(wer_difference=sum(errors)/sum(words),
                interval=bootstrap_interval(errors, words, [r['evaluation_group'] for r in candidate], samples, seed),
                improved=sum(e < 0 for e in errors), worsened=sum(e > 0 for e in errors),
                unchanged=sum(e == 0 for e in errors))
