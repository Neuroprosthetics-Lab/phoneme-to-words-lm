"""One isolated batch-G measurement; test jobs accept only frozen scoring weights."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')

import editdistance
import numpy as np
import torch

from phoneme_to_words_lm import KenLMFlashlightTextLM
from phoneme_to_words_lm.evaluation import (group_key, prepare_ranking, rank, select_weights,
                                           subset_indices, summarize)
from phoneme_to_words_lm.sweep_contract import file_identity
from sweep.lm_sweep_worker import load_and_filter_logits


def decode_one(decoder, row, reorder, device):
    logits = torch.as_tensor(row['logits'], dtype=torch.float32).unsqueeze(0)
    if reorder:
        logits = torch.cat((logits[:, :, :1], logits[:, :, -1:], logits[:, :, 1:-1]), dim=-1)
    logits = logits.contiguous()
    lengths = torch.tensor([row['adjusted_len']], dtype=torch.int64)
    if device is not None:
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    context = row.get('context')
    result = decoder.offline_decode(logits, lengths, contexts=[context] if context is not None else None)[0]
    if device is not None:
        torch.cuda.synchronize(device)
    return result, time.perf_counter()-started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text())
    if job['stage'] == 'test' and job['tune']:
        raise ValueError('Test jobs cannot tune scoring weights')
    # Load the selected records and initialize one isolated decoder.
    torch.set_num_threads(job['torch_threads'])
    lookup = {row['source_index']: row for row in load_and_filter_logits({'logits_pkl_path': job['logits']})}
    if len(set(job['indices'])) != len(job['indices']):
        raise ValueError('Duplicate evaluation indices')
    rows = [lookup[index] for index in job['indices']]
    if not rows:
        raise ValueError('No evaluation utterances')
    for row in rows:
        row['evaluation_group'] = group_key(row, job['bootstrap_group'])
    config = dict(job['decoder'])
    # Candidate dumps retain native ordering; final policy selection uses raw scores.
    config['llm_alpha'] = 0.
    device = None
    if config['do_llm_rescoring'] and config['llm_device'].startswith('cuda'):
        device = torch.device(config['llm_device'])
    if device is not None:
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    decoder = KenLMFlashlightTextLM(**config)
    load_seconds = time.perf_counter()-started

    # Warm up separately, then collect candidates and immutable raw scores.
    first_decode = None
    for row in rows[:job['warmup_sentences']]:
        _, elapsed = decode_one(decoder, row, job['reorder_logit_columns'], device)
        if first_decode is None:
            first_decode = elapsed
    records = []
    for position, row in enumerate(rows):
        result, elapsed = decode_one(decoder, row, job['reorder_logit_columns'], device)
        if first_decode is None:
            first_decode = elapsed
        result.update(transcription=row['transcription'], source_index=row['source_index'],
                      evaluation_group=row['evaluation_group'], context=row.get('context'),
                      decode_time=elapsed, edit_reference=row['transcription'],
                      edit_counts_by_text={text: int(editdistance.eval(row['transcription'].split(), text.split()))
                                           for text in result['word_seqs']})
        records.append(result)
        if (position+1) % 64 == 0 or position+1 == len(rows):
            print(f"{job['stage']} {job['name']}: {position+1}/{len(rows)}", flush=True)

    # Only validation jobs may select new scoring weights.
    arrays = prepare_ranking(records)
    anchor = (job['decoder']['llm_alpha'], job['decoder']['llm_length_penalty'])
    if job['tune'] and config['do_llm_rescoring']:
        policy, curve = select_weights(arrays, job['alphas'], job['length_penalties'], anchor)
    else:
        alpha, penalty = anchor if config['do_llm_rescoring'] else (0., 0.)
        policy = dict(alpha=alpha, length_penalty=penalty, edits=sum(rank(arrays, alpha, penalty)[0]))
        curve = [policy]
    # Dedicated finalist timing uses a common random subset, with repeated calls
    # after model warmup. Screening uses the accuracy pass's observed timings.
    if job['timing_repeats']:
        timings = []
        timed_indices = subset_indices(rows, job['timing_limit'], job['seed']+1)
        by_index = {row['source_index']: row for row in rows}
        decoder.llm_alpha = policy['alpha']
        decoder.llm_length_penalty = policy['length_penalty']
        _, expected_indices = rank(arrays, policy['alpha'], policy['length_penalty'])
        expected = {row['source_index']: row['word_seqs'][index] for row, index in zip(records, expected_indices)}
        for repeat in range(job['timing_repeats']):
            order = list(timed_indices)
            np.random.default_rng(job['seed']+repeat+7).shuffle(order)
            for index in order:
                actual, elapsed = decode_one(decoder, by_index[index], job['reorder_logit_columns'], device)
                if actual['word_seqs'][0] != expected[index]:
                    raise RuntimeError(f'Direct/frozen post-hoc winner changed for source index {index}')
                timings.append(dict(source_index=index, repeat=repeat, seconds=elapsed))
    else:
        timings = [
            dict(source_index=row['source_index'], repeat=0, seconds=row['decode_time'])
            for row in records
        ]

    # Publish metrics and candidate dumps before the completion marker.
    summary = summarize(records, arrays, policy['alpha'], policy['length_penalty'],
                        [row['seconds'] for row in timings], decoder._hotword_lookup,
                        bootstrap_samples=job['bootstrap_samples'], seed=job['seed'])
    summary.update(
        name=job['name'],
        stage=job['stage'],
        load_seconds=load_seconds,
        first_decode_seconds=first_decode,
        warmup_sentences=min(job['warmup_sentences'], len(rows)),
        hotword_backend=decoder.hotword_backend,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        peak_vram_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30 if device else None,
        peak_vram_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30 if device else None,
        timed_utterances=len({row['source_index'] for row in timings}),
        timing_repeats=job['timing_repeats'] or 1,
        decoder={
            **job['decoder'],
            'llm_alpha': policy['alpha'],
            'llm_length_penalty': policy['length_penalty'],
        },
    )
    output = Path(args.job).parent
    files = {
        'candidates.json': records,
        'summary.json': summary,
        'scoring_curve.json': curve,
        'timings.json': timings,
    }
    for name, value in files.items():
        path = output/name
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2))
        temporary.replace(path)
    complete = {
        'job_sha256': file_identity(args.job)['sha256'],
        'outputs': {name: file_identity(output/name)['sha256'] for name in files},
    }
    (output/'complete.json').write_text(json.dumps(complete, indent=2))
    print('COMPLETE', json.dumps({key:summary[key] for key in ('name','stage','wer','alpha','length_penalty','timing')}), flush=True)


if __name__ == '__main__':
    main()
