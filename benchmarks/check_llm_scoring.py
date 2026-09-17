"""Check a local Qwen scorer against a full-logsoftmax reference and policies.

Optional --candidates reuses evaluate_logits.py output for validation-only
prefix/EOS ablations. It performs no beam search or model downloads.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--candidates')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    os.environ.update(HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                      HF_HUB_DISABLE_PROGRESS_BARS='1')
    import numpy as np
    import torch
    from transformers.utils.logging import disable_progress_bar
    from phoneme_to_words_lm.decoder import _build_llm
    from phoneme_to_words_lm.llm_scoring import (
        SCORER_VERSION, prepare_scoring_inputs, padded_inputs,
        _score_batch, score_sentences,
    )
    disable_progress_bar()
    torch.set_num_threads(1)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model, tokenizer = _build_llm(args.model, device='cuda:0', dtype='bfloat16')
    texts = ['hello', 'yes', 'the cat is on the mat', 'a longer candidate with several words']
    contexts = ['', 'previous sentence.', '', 'history\n']
    report = dict(model=args.model, scorer_version=SCORER_VERSION,
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(), checks=[])
    for eos in [False, True]:
        requests = prepare_scoring_inputs(tokenizer, texts, contexts, score_eos=eos)
        # Same padded shape and mask, intentionally independent score formula.
        with torch.inference_mode():
            ids, attention, targets = padded_inputs(requests, tokenizer, model.device)
            logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits
            token_logp = logits.float().log_softmax(-1)[:, :-1].gather(-1, ids[:, 1:, None]).squeeze(-1)
            reference = (token_logp * targets[:, 1:]).sum(-1).cpu()
            del logits, token_logp
        actual = torch.tensor(_score_batch(model, tokenizer, requests, 16))
        torch.testing.assert_close(actual, reference, atol=2e-4, rtol=1e-5)
        alone = [score_sentences(model, tokenizer, [s], [c], score_eos=eos)[0][0]
                 for s, c in zip(texts, contexts)]
        report['checks'].append(dict(score_eos=eos,
            counts=[r.token_count for r in requests], scores=actual.tolist(),
            max_reference_difference=float((actual-reference).abs().max()),
            max_singleton_batch_difference=float((actual-torch.tensor(alone)).abs().max())))
    no_context = prepare_scoring_inputs(tokenizer, ['hello'])
    empty_context = prepare_scoring_inputs(tokenizer, ['hello'], [''])
    mixed_context = prepare_scoring_inputs(tokenizer, ['hello', 'yes'], ['', 'history'])
    assert no_context == empty_context == mixed_context[:1]
    raw, counts = score_sentences(model, tokenizer, ['hello', 'yes'])
    assert counts == [1, 1] and all(s < 0 for s in raw) and raw[0] != raw[1]
    report['single_token'] = dict(scores=raw, counts=counts)
    print('REFERENCE', json.dumps(report), flush=True)
    if args.candidates:
        payload = Path(args.candidates).read_bytes()
        records = json.loads(payload)
        report['candidate_sha256'] = hashlib.sha256(payload).hexdigest()
        report['n_trials'] = len(records)
        report['n_words'] = sum(r['n_words'] for r in records)
        report['policies'] = {}
        for name, prefix, eos in [('newline_eos', '\n', True),
                                  ('sentence_prefix', 'Sentence:\n', False)]:
            scored = []
            start = time.perf_counter()
            for i, row in enumerate(records):
                raw, counts = score_sentences(model, tokenizer, row['word_seqs'],
                                               prefix=prefix, score_eos=eos)
                scored.append(dict(index=row['index'], raw_llm_scores=raw,
                                   llm_token_counts=counts))
                if (i+1) % 64 == 0:
                    print('PROGRESS', name, i+1, flush=True)
            curve = []
            for alpha in np.round(np.arange(0, 1.501, .05), 2):
                errors = sum(row['edits'][int(np.argmax(
                    np.asarray(row['beam_scores']) + alpha*np.asarray(s['raw_llm_scores'])))]
                    for row, s in zip(records, scored))
                curve.append(dict(alpha=float(alpha), errors=errors))
            report['policies'][name] = dict(prefix=prefix, score_eos=eos,
                seconds=time.perf_counter()-start, alpha_curve=curve,
                best=min(curve, key=lambda x:x['errors']))
            (output/(name+'.json')).write_text(json.dumps(scored))
            print('POLICY', name, json.dumps(report['policies'][name]), flush=True)
    (output/'summary.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
