"""Reproducible validation runner; no Redis, training, or remote model downloads.

Run from the repository root. Use --package-root to evaluate a preserved checkout.
Detailed candidate outputs stay in the explicitly selected output directory.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--logits', required=True)
    ap.add_argument('--resources', required=True)
    ap.add_argument('--model', required=True, help='Local Qwen snapshot path')
    ap.add_argument('--output', required=True)
    ap.add_argument('--package-root', default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument('--limit', type=int, default=256, help='0 evaluates all eligible trials')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--blank-threshold', type=float, default=1.0)
    ap.add_argument('--no-llm', action='store_true')
    ap.add_argument('--prefix', default=None)
    ap.add_argument('--score-eos', action='store_true')
    ap.add_argument('--beam-size', type=int, default=6000)
    ap.add_argument('--hotwords-path', help='Optional word-to-bonus YAML file')
    args = ap.parse_args()
    sys.path.insert(0, args.package_root)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    import torch
    import numpy as np
    import editdistance
    from phoneme_to_words_lm import KenLMFlashlightTextLM
    from phoneme_to_words_lm.utils import remove_punctuation, replace_words
    from transformers.utils.logging import disable_progress_bar
    disable_progress_bar()
    torch.set_num_threads(1)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    payload = Path(args.logits).read_bytes()
    rows = pickle.loads(payload)
    eligible = []
    for idx, row in enumerate(rows):
        ref = row['transcription']
        if not isinstance(ref, str):
            codes = list(ref); end = codes.index(0) if 0 in codes else len(codes)
            ref = ''.join(chr(int(c)) for c in codes[:end])
        ref = replace_words(remove_punctuation(ref))
        if ref and max(map(len, ref.split())) > 1:
            eligible.append((idx, row, ref))
    if args.limit and len(eligible) > args.limit:
        selected = sorted(np.random.default_rng(args.seed).choice(len(eligible), args.limit, replace=False))
        eligible = [eligible[i] for i in selected]
    root = Path(args.resources)
    cfg = dict(lexicon_path=str(root/'lexicon.txt'), tokens_path=str(root/'tokens.txt'),
               kenlm_model_path=str(root/'lm_unpruned.bin'), beam_size=args.beam_size,
               token_beam_size=26, beam_threshold=20., lm_weight=1.5,
               temperature=2., blank_penalty=4., blank_skip_threshold=args.blank_threshold,
               n_best=100, do_llm_rescoring=not args.no_llm, llm_model_name=args.model,
               llm_device='cuda:0', llm_alpha=.45, llm_batch_size=100)
    if args.prefix is not None: cfg['llm_prefix'] = args.prefix
    if args.score_eos: cfg['llm_score_eos'] = True
    if args.hotwords_path: cfg['hotwords_path'] = args.hotwords_path
    manifest = dict(arguments=vars(args), config=cfg, data_sha256=hashlib.sha256(payload).hexdigest(),
                    indices=[i for i, _, _ in eligible], total_trials=len(rows),
                    torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if not args.no_llm else None)
    package = Path(args.package_root)/'phoneme_to_words_lm'
    manifest['source_sha256'] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.glob('*.py'))
    }
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('INIT', json.dumps(manifest), flush=True)
    start = time.perf_counter(); d = KenLMFlashlightTextLM(**cfg)
    manifest['load_seconds'] = time.perf_counter()-start
    import flashlight.lib.text.flashlight_lib_text_decoder as native_extension
    extension_dir = Path(native_extension.__file__).parent
    manifest['flashlight'] = dict(
        version=importlib.metadata.version('flashlight-text'),
        module_path=native_extension.__file__,
        binaries_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(extension_dir.glob('*.so'))})
    manifest['hotword_backend'] = getattr(d, 'hotword_backend', 'python')
    manifest['hotwords'] = d.get_hotwords()
    # Preserve native scores in baseline evidence without changing its behavior.
    native = {}; original_extract = d._extract_nbest
    def capture(results, *a, **kw):
        native.clear()
        for r in results:
            s = replace_words(remove_punctuation(' '.join(d.word_dict.get_entry(w) for w in r.words if w >= 0)))
            native[s] = max(native.get(s, -float('inf')), r.score)
        return original_extract(results, *a, **kw)
    d._extract_nbest = capture
    records = []
    for pos, (idx, row, ref) in enumerate(eligible):
        x = torch.as_tensor(row['logits'], dtype=torch.float32).unsqueeze(0)
        x = torch.cat((x[:, :, :1], x[:, :, -1:], x[:, :, 1:-1]), -1).contiguous()
        length = int(row.get('adjusted_len', row.get('adjusted_lens')))
        t = time.perf_counter(); result = d.offline_decode(x, torch.tensor([length]))[0]
        elapsed = time.perf_counter()-t
        edits = [int(editdistance.eval(ref.split(), s.split())) for s in result['word_seqs']]
        result.update(index=idx, reference=ref, day_index=int(row.get('day_index', -1)),
                      n_words=len(ref.split()), edits=edits, seconds=elapsed,
                      native_beam_scores=[native.get(s, -float('inf')) for s in result['word_seqs']])
        records.append(result)
        if (pos+1) % 16 == 0 or pos+1 == len(eligible):
            print('PROGRESS', pos+1, 'of', len(eligible), 'WER', sum(r['edits'][0] for r in records)/sum(r['n_words'] for r in records), flush=True)
            (output/'candidates.json').write_text(json.dumps(records))
    nwords = sum(r['n_words'] for r in records)
    # Natural ngram ranking from whichever search score contract produced this result.
    def ngram_index(r):
        if 'beam_scores' in r: return int(np.argmax(r['beam_scores']))
        return int(np.argmax(np.array(r['acoustic_scores'])+1.5*np.array(r['ngram_scores'])))
    summary = dict(n_sentences=len(records), n_words=nwords,
                   final_edits=sum(r['edits'][0] for r in records),
                   ngram_edits=sum(r['edits'][ngram_index(r)] for r in records),
                   oracle_edits=sum(min(r['edits']) for r in records),
                   empty_count=sum(not r['word_seqs'][0] for r in records),
                   median_candidates=float(np.median([len(r['word_seqs']) for r in records])),
                   median_seconds=float(np.median([r['seconds'] for r in records[1:]])),
                   p95_seconds=float(np.percentile([r['seconds'] for r in records[1:]],95)),
                   median_ngram_seconds=float(np.median([r['ngram_time'] for r in records[1:]])),
                   p95_ngram_seconds=float(np.percentile([r['ngram_time'] for r in records[1:]],95)),
                   peak_vram_gib=torch.cuda.max_memory_allocated()/2**30 if not args.no_llm else None)
    for k in ['final','ngram','oracle']: summary[k+'_wer']=summary[k+'_edits']/nwords
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('SUMMARY',json.dumps(summary),flush=True)

if __name__ == '__main__': main()
