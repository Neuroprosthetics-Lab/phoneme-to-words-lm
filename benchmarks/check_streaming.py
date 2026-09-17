"""Verify offline/chunked equality with optional blank-run compression."""
import argparse
import json
from pathlib import Path
import pickle
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logits', required=True)
    parser.add_argument('--resources', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--blank-threshold', type=float, default=1.0)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    import torch
    from phoneme_to_words_lm import KenLMFlashlightTextLM
    torch.set_num_threads(1)
    with open(args.logits, 'rb') as stream:
        rows = pickle.load(stream)
    indices = sorted(set([0, len(rows)-1, int(np.argmax([r['adjusted_len'] for r in rows]))]
                         + list(map(int, np.random.default_rng(42).choice(len(rows), 5, replace=False)))))
    root = Path(args.resources)
    decoder = KenLMFlashlightTextLM(
        lexicon_path=str(root/'lexicon.txt'), tokens_path=str(root/'tokens.txt'),
        kenlm_model_path=str(root/'lm_unpruned.bin'), beam_size=6000,
        token_beam_size=26, beam_threshold=20., lm_weight=1.5,
        temperature=2., blank_penalty=4., n_best=100, do_llm_rescoring=False,
        blank_skip_threshold=args.blank_threshold)
    report = dict(indices=indices, blank_skip_threshold=args.blank_threshold, trials=[])
    for index in indices:
        row = rows[index]
        x = torch.as_tensor(row['logits'], dtype=torch.float32)[:row['adjusted_len']]
        x = torch.cat((x[:, :1], x[:, -1:], x[:, 1:-1]), -1).contiguous()
        offline = decoder.offline_decode(x.unsqueeze(0))[0]
        for size in [1, 17, 64, len(x)]:
            decoder.online_decode_begin()
            for start in range(0, len(x), size):
                decoder.online_decode_step(x[start:start+size])
            online = decoder.online_decode_end()
            for key in ['word_seqs', 'word_ids', 'raw_word_seqs', 'beam_scores',
                        'acoustic_scores', 'ngram_scores', 'final_scores']:
                assert online[key] == offline[key], (index, size, key)
            assert online['total_frames'] == len(x)
            assert online['search_frames'] == offline['search_frames']
            assert online['preprocessing_config'] == offline['preprocessing_config']
        report['trials'].append(dict(index=index, frames=len(x), search_frames=offline['search_frames'],
                                    candidates=len(offline['word_seqs'])))
        print('PASS', index, len(x), offline['search_frames'], flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
