"""Paired native/Python hotword agreement and timing with one real KenLM load.

Requires the rebuilt Flashlight extension. Alternates execution order, excludes
the first pair per configuration from timings, and checks all returned n-best
scores. Use --manifest from evaluate_logits.py to select a fixed source subset.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logits', required=True)
    parser.add_argument('--resources', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=32)
    args = parser.parse_args()
    if args.limit < 2:
        parser.error('--limit must be at least 2')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    import torch
    from phoneme_to_words_lm.decoder import KenLMFlashlightTextLM, BiasingLM, HotwordLM
    if HotwordLM is None:
        raise RuntimeError('Rebuild flashlight_text to run native hotword verification')
    torch.set_num_threads(1)
    payload = Path(args.logits).read_bytes()
    manifest = json.loads(Path(args.manifest).read_text())
    assert hashlib.sha256(payload).hexdigest() == manifest['data_sha256']
    rows = pickle.loads(payload)
    indices = sorted(map(int, np.random.default_rng(42).choice(
        manifest['indices'], min(args.limit, len(manifest['indices'])), replace=False)))
    root = Path(args.resources)
    decoder = KenLMFlashlightTextLM(
        lexicon_path=str(root/'lexicon.txt'), tokens_path=str(root/'tokens.txt'),
        kenlm_model_path=str(root/'lm_unpruned.bin'), beam_size=6000,
        token_beam_size=26, beam_threshold=20., lm_weight=1.5,
        temperature=2., blank_penalty=4., n_best=100, do_llm_rescoring=False,
        blank_skip_threshold=.98)
    fields = ('word_seqs', 'raw_word_seqs', 'word_ids', 'beam_scores',
              'ngram_scores', 'acoustic_scores', 'final_scores',
              'search_frames', 'total_frames', 'preprocessing_config')
    report = dict(arguments=vars(args), indices=indices, data_sha256=manifest['data_sha256'],
                  flashlight_version=importlib.metadata.version('flashlight-text'),
                  configurations=[], single_word_checks=0)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    previous_indices = set()
    kenlm = decoder._kenlm
    configs = [('empty', {}), ('positive', {'the': .1, 'you': 2.}),
               ('zero', {'the': 0., 'you': 0.}),
               ('negative', {'the': -.1, 'you': -1.23456789123}),
               ('membership', {'and': .2, 'to': -.5}), ('cleared', {})]
    empty_results = {}
    for name, hotwords in configs:
        trie, lm, search = decoder.trie, decoder.lm, decoder.ngram_decoder
        t = time.perf_counter()
        decoder.set_hotwords(hotwords)
        decoder._apply_pending_hotwords()
        update_seconds = time.perf_counter()-t
        current_indices, bonuses = decoder._resolve_hotwords(hotwords)
        assert decoder._kenlm is kenlm
        if current_indices == previous_indices:
            assert decoder.trie is trie and decoder.lm is lm and decoder.ngram_decoder is search
        else:
            assert decoder.trie is not trie and decoder.ngram_decoder is not search
        previous_indices = current_indices
        native_lm, native_decoder = decoder.lm, decoder.ngram_decoder
        reference_lm = BiasingLM(kenlm)
        reference_lm.hotword_bonus = bonuses
        reference_decoder = decoder._make_ngram_decoder(decoder.trie, reference_lm)
        # Check real nonzero KenLM values at multiple states, including finish.
        for start_with_nothing in (False, True):
            state = native_lm.start(start_with_nothing)
            for word in ('the', 'you', 'and', 'to', 'hello'):
                label = decoder._hotword_lookup[word][0]
                label = decoder.word_dict.get_index(label)
                got_state, score = native_lm.score(state, label)
                ref_state, expected = reference_lm.score(state, label)
                assert score == float(np.float32(expected)), (name, word, score, expected)
                assert got_state.compare(ref_state) == 0
                state = got_state
                report['single_word_checks'] += 1
            got_state, score = native_lm.finish(state)
            ref_state, expected = reference_lm.finish(state)
            assert score == expected and got_state.compare(ref_state) == 0
        records = []
        for pos, index in enumerate(indices):
            row = rows[index]
            x = torch.as_tensor(row['logits'], dtype=torch.float32)[:int(row['adjusted_len'])]
            x = torch.cat((x[:, :1], x[:, -1:], x[:, 1:-1]), -1).unsqueeze(0).contiguous()
            results, seconds = {}, {}
            order = ('native', 'python') if pos % 2 == 0 else ('python', 'native')
            for backend in order:
                decoder.lm, decoder.ngram_decoder = (
                    (native_lm, native_decoder) if backend == 'native' else (reference_lm, reference_decoder))
                t = time.perf_counter()
                results[backend] = decoder.offline_decode(x)[0]
                seconds[backend] = time.perf_counter()-t
            decoder.lm, decoder.ngram_decoder = native_lm, native_decoder
            for field in fields:
                assert results['native'][field] == results['python'][field], (name, index, field)
            if name == 'empty':
                empty_results[index] = {key: results['native'][key] for key in fields}
            if name == 'cleared':
                for field in fields:
                    assert results['native'][field] == empty_results[index][field], ('clear', index, field)
            # Exercise streaming for both backends with actual hotwords. Queue
            # an update mid-stream and ensure end() still uses the old table.
            if pos == 0:
                for backend, active_lm, active_decoder in (
                    ('native', native_lm, native_decoder), ('python', reference_lm, reference_decoder)):
                    decoder.lm, decoder.ngram_decoder = active_lm, active_decoder
                    for size in (1, 17, 64):
                        decoder.online_decode_begin()
                        for start in range(0, x.shape[1], size):
                            decoder.online_decode_step(x[:, start:start+size])
                            decoder.set_hotwords({'hello': 9.})
                        online = decoder.online_decode_end()
                        for field in fields:
                            assert online[field] == results[backend][field], (name, backend, size, field)
                        # Cancel the pending test update without applying it.
                        decoder._pending_hotwords = None
                decoder.lm, decoder.ngram_decoder = native_lm, native_decoder
            records.append(dict(index=index, seconds=seconds, candidates=len(results['native']['word_seqs'])))
        timings = {}
        for backend in ('native', 'python'):
            values = [r['seconds'][backend] for r in records[1:]]
            timings[backend] = dict(median_ms=float(np.median(values)*1000),
                                    p95_ms=float(np.percentile(values, 95)*1000))
        report['configurations'].append(dict(name=name, hotwords=hotwords,
            backend=decoder.hotword_backend, update_seconds=update_seconds,
            candidate_pairs=sum(r['candidates'] for r in records), timings=timings, trials=records))
        output.write_text(json.dumps(report, indent=2))
        print('PASS', name, json.dumps(timings), flush=True)


if __name__ == '__main__':
    main()
