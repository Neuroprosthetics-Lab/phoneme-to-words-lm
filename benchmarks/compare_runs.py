"""Compare evaluate_logits.py outputs and isolate score-only alpha changes."""
import argparse
import json
from pathlib import Path


def read(directory, name):
    return json.loads((Path(directory)/name).read_text())


def alpha_curve(rows, base_key, llm_key, lm_weight):
    curve = []
    for step in range(31):
        alpha = round(step*.05, 2)
        errors = 0
        for row in rows:
            base = (row[base_key] if base_key else
                    [a+lm_weight*n for a, n in zip(row['acoustic_scores'], row['ngram_scores'])])
            scores = [b+alpha*(s or 0.) for b, s in zip(base, row[llm_key])]
            errors += row['edits'][max(range(len(scores)), key=scores.__getitem__)]
        curve.append(dict(alpha=alpha, errors=errors))
    return dict(at_045=curve[9], best=min(curve, key=lambda r:r['errors']), curve=curve)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--corrected', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    baseline = read(args.baseline, 'candidates.json')
    corrected = read(args.corrected, 'candidates.json')
    bm = read(args.baseline, 'manifest.json')
    cm = read(args.corrected, 'manifest.json')
    assert bm['indices'] == cm['indices'], 'Mismatched trial subsets'
    assert bm['data_sha256'] == cm['data_sha256'], 'Mismatched input data'
    weight = cm['config']['lm_weight']
    assert weight == bm['config']['lm_weight']
    differences, changed, n_candidates, same_sets = [], [], 0, 0
    for old, new in zip(baseline, corrected):
        native = dict(zip(old['word_seqs'], old['native_beam_scores']))
        same_sets += set(native) == set(new['word_seqs'])
        for text, beam in zip(new['word_seqs'], new['beam_scores']):
            if text in native:
                differences.append(abs(beam-native[text]))
                n_candidates += 1
        if old['word_seqs'][0] != new['word_seqs'][0]:
            changed.append(dict(index=old['index'], old_errors=old['edits'][0], new_errors=new['edits'][0]))
    report = dict(n_trials=len(corrected), identical_candidate_sets=same_sets,
        compared_candidates=n_candidates, max_native_score_difference=max(differences, default=0.),
        changed_final_predictions=changed,
        baseline_summary=read(args.baseline, 'summary.json'),
        corrected_summary=read(args.corrected, 'summary.json'),
        alpha_ablations=dict(
            legacy=alpha_curve(baseline, None, 'llm_scores', weight),
            native_with_legacy_llm=alpha_curve(baseline, 'native_beam_scores', 'llm_scores', weight),
            corrected=alpha_curve(corrected, 'beam_scores', 'llm_scores', weight)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='alpha_ablations'}, indent=2))
    for name, result in report['alpha_ablations'].items():
        print(name, 'at .45:', result['at_045']['errors'], 'best:', result['best'])


if __name__ == '__main__':
    main()
