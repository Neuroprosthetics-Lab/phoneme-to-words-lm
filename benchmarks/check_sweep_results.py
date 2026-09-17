"""Compare saved sweep candidates with direct decoding in a fresh process."""
import argparse
import json
import os
from pathlib import Path
import pickle
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--package-root', help='Optional frozen package root for before/after comparison')
    args = parser.parse_args()
    package_root = args.package_root or str(Path(__file__).resolve().parents[1])
    sys.path.insert(0, package_root)
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    import torch
    from phoneme_to_words_lm import KenLMFlashlightTextLM
    torch.set_num_threads(1)
    cfg = json.loads(Path(args.worker_config).read_text())
    saved = json.loads(Path(cfg['output_path']).with_name('candidates.json').read_text())
    with open(cfg['logits_pkl_path'], 'rb') as stream:
        records = pickle.load(stream)
    decoder = KenLMFlashlightTextLM(**cfg['decoder_config'])
    checked, candidates = set(), 0
    for expected in saved:
        record = records[expected['source_index']]
        logits = torch.as_tensor(record['logits'], dtype=torch.float32).unsqueeze(0)
        if cfg.get('reorder_logit_columns', True):
            logits = torch.cat((logits[:, :, :1], logits[:, :, -1:], logits[:, :, 1:-1]), dim=-1)
        length = record.get('adjusted_len', record.get('adjusted_lens', logits.shape[1]))
        actual = decoder.offline_decode(logits.contiguous(), torch.tensor([length]),
                                        contexts=[record.get('context')])[0]
        for key, value in actual.items():
            if key.endswith('_time'):
                continue
            # Normalize tuple/list representations across JSON serialization.
            assert json.loads(json.dumps(value)) == expected[key], (expected['source_index'], key)
            checked.add(key)
        candidates += len(actual['word_seqs'])
    report = dict(sentences=len(saved), candidates=candidates, exact_fields=sorted(checked),
                  hotword_backend=decoder.hotword_backend, adapter=decoder.llm_lora_path,
                  contexts=sum(row['context'] is not None for row in saved), package_root=package_root)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
