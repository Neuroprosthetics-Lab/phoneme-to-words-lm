"""Screen on validation, freeze choices, then evaluate a separate development test.

Example: python benchmarks/run_evaluation.py --config benchmarks/example_evaluation.yaml
Use --phase validation to leave the test set unevaluated until a later invocation.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from omegaconf import OmegaConf

from phoneme_to_words_lm.evaluation import (choose_finalists, freeze_selection, group_key,
    objective, paired_comparison, pareto_front, split_indices, subset_indices)
from phoneme_to_words_lm.sweep_contract import (alpha_values, file_identity,
    decoder_defaults, positive_integer, resolve_decoder_config, runtime_identity, RESOURCE_FIELDS)
from sweep.lm_sweep_worker import load_and_filter_logits


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def load_config(path, overrides=None):
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    defaults = {
        'seed': 42,
        'screen_limit': 128,
        'validation_limit': None,
        'test_limit': None,
        'test_fraction': .2,
        'split_group': 'reference',
        'bootstrap_group': 'day_index',
        'bootstrap_samples': 2000,
        'finalists': 4,
        'warmup_sentences': 2,
        'timing_limit': 128,
        'timing_repeats': 2,
        'torch_threads': 1,
        'worker_timeout_seconds': 1800,
        'reorder_logit_columns': True,
        'objective': {'wer': 1000., 'latency': 1.},
        'alphas': {'low': 0., 'high': 1.2, 'step': .05},
        'length_penalties': [-.25, 0., .25],
        'baseline': 'baseline',
        'test_logits': None,
    }
    unknown = set(cfg) - set(defaults) - {'validation_logits', 'output_dir', 'decoder', 'configurations'}
    if unknown:
        raise ValueError(f'Unknown evaluation settings: {sorted(unknown)}')
    cfg = {**defaults, **cfg, **(overrides or {})}
    for key in ('seed', 'warmup_sentences'):
        positive_integer(key, cfg[key], allow_zero=True)
    for key in ('bootstrap_samples', 'timing_repeats', 'torch_threads', 'worker_timeout_seconds'):
        positive_integer(key, cfg[key])
    for key in ('screen_limit', 'validation_limit', 'test_limit', 'timing_limit', 'finalists'):
        if cfg[key] is not None:
            positive_integer(key, cfg[key])
    if cfg['finalists'] is not None and cfg['finalists'] < 4:
        raise ValueError('finalists must be at least four (baseline, weighted, accuracy and speed) or null for all')
    weights = cfg['objective']
    if set(weights) != {'wer','latency'} or any(isinstance(v, bool) or not isinstance(v, (int,float))
            or not math.isfinite(v) or v < 0 for v in weights.values()) or not any(weights.values()):
        raise ValueError('objective needs finite nonnegative wer/latency weights, not both zero')
    if not isinstance(cfg['reorder_logit_columns'], bool):
        raise ValueError('reorder_logit_columns must be boolean')
    alpha_values(**cfg['alphas'])
    penalties = cfg['length_penalties']
    if not isinstance(penalties, list) or not penalties or any(isinstance(p, bool)
            or not isinstance(p, (int,float)) or not math.isfinite(p) for p in penalties):
        raise ValueError('length_penalties must be a nonempty list of finite numbers')
    for key in ('split_group','bootstrap_group'):
        if not isinstance(cfg[key], str) or not cfg[key]:
            raise ValueError(f'{key} must be a field name, reference or utterance')
    for key in ('validation_logits','test_logits','output_dir'):
        if cfg[key] is not None:
            cfg[key] = str(Path(cfg[key]).expanduser().resolve())
            if key != 'output_dir' and not Path(cfg[key]).is_file():
                raise FileNotFoundError(cfg[key])
    cases = {}
    if not isinstance(cfg['configurations'], list) or not cfg['configurations']:
        raise ValueError('configurations must be a nonempty list of named decoder overrides')
    for entry in cfg['configurations']:
        if set(entry)-{'name','overrides','tune'}:
            raise ValueError(f'Unknown configuration fields: {entry}')
        name = entry['name']
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name) or name in cases:
            raise ValueError(f'Duplicate or invalid configuration name: {name!r}')
        tune = entry.get('tune', True)
        if not isinstance(tune, bool):
            raise ValueError('tune must be boolean')
        supplied = {**cfg['decoder'], **entry.get('overrides', {})}
        effective = resolve_decoder_config({'decoder_config':supplied})
        effective['llm_alpha'] = supplied.get('llm_alpha', decoder_defaults()['llm_alpha'])
        if (isinstance(effective['llm_alpha'], bool) or not isinstance(effective['llm_alpha'], (int,float))
                or not math.isfinite(effective['llm_alpha']) or effective['llm_alpha'] < 0):
            raise ValueError('llm_alpha must be finite and nonnegative')
        cases[name] = dict(decoder=effective, tune=tune)
    if cfg['baseline'] not in cases or cases[cfg['baseline']]['tune']:
        raise ValueError('baseline must name a configuration with tune: false')
    return cfg, cases


def prepare_data(cfg):
    validation = load_and_filter_logits({'logits_pkl_path':cfg['validation_logits']})
    if cfg['test_logits']:
        if file_identity(cfg['validation_logits'])['sha256'] == file_identity(cfg['test_logits'])['sha256']:
            raise ValueError('Validation/test files contain identical bytes; use an explicit grouped split instead')
        test = load_and_filter_logits({'logits_pkl_path':cfg['test_logits']})
        test_path = cfg['test_logits']
        split = 'separate_files'
    else:
        val_ids, test_ids = split_indices(validation, cfg['test_fraction'], cfg['seed'], cfg['split_group'])
        val_set, test_set = set(val_ids), set(test_ids)
        test = [row for row in validation if row['source_index'] in test_set]
        validation = [row for row in validation if row['source_index'] in val_set]
        test_path = cfg['validation_logits']
        split = f"grouped_within_file:{cfg['split_group']}"
    val_indices = subset_indices(validation, cfg['validation_limit'], cfg['seed'])
    test_indices = subset_indices(test, cfg['test_limit'], cfg['seed'] + 2)
    val_set, test_set = set(val_indices), set(test_indices)
    validation = [row for row in validation if row['source_index'] in val_set]
    test = [row for row in test if row['source_index'] in test_set]
    for row in validation + test:
        group_key(row, cfg['bootstrap_group'])
    shared = set(row['transcription'] for row in validation) & set(row['transcription'] for row in test)
    if shared:
        print(f'NOTE: {len(shared)} normalized reference identities occur in both files; reported in the split manifest.', flush=True)
    return {
        'split': split,
        'validation_path': cfg['validation_logits'],
        'test_path': test_path,
        'validation_indices': val_indices,
        'test_indices': test_indices,
        'screen_indices': subset_indices(validation, cfg['screen_limit'], cfg['seed'] + 3),
        'validation_words': sum(len(row['transcription'].split()) for row in validation),
        'test_words': sum(len(row['transcription'].split()) for row in test),
        'shared_reference_identities': len(shared),
        'validation_groups': len({group_key(row, cfg['bootstrap_group']) for row in validation}),
        'test_groups': len({group_key(row, cfg['bootstrap_group']) for row in test}),
    }


def experiment_identity(cfg, cases, data):
    """Hash each distinct input/model/native/source file once per invocation."""
    from huggingface_hub import snapshot_download
    import flashlight.lib.text.flashlight_lib_text_decoder as native
    import torch
    files = {data['validation_path'], data['test_path']}
    for case in cases.values():
        decoder = case['decoder']
        files.update(decoder[key] for key in RESOURCE_FIELDS)
        if decoder['hotwords_path']:
            files.add(decoder['hotwords_path'])
        if decoder['do_llm_rescoring']:
            model = decoder['llm_model_name']
            if not Path(model).is_dir():
                model = snapshot_download(model, cache_dir=decoder['llm_cache_dir'], local_files_only=True)
            decoder['llm_model_name'] = str(Path(model).resolve())
            for directory in (model, decoder['llm_lora_path']):
                if directory:
                    files.update(str(p.resolve()) for p in Path(directory).rglob('*') if p.is_file())
    root = Path(__file__).resolve().parents[1]
    for folder in ('phoneme_to_words_lm','sweep'):
        files.update(str(p) for p in (root/folder).glob('*.py'))
    files.update(str(root/'benchmarks'/name) for name in ('run_evaluation.py','evaluation_worker.py'))
    files.update(str(p) for p in Path(native.__file__).parent.glob('*.so'))
    devices = sorted({case['decoder']['llm_device'] for case in cases.values() if case['decoder']['do_llm_rescoring']})
    gpu = {}
    for device in devices:
        if device.startswith('cuda'):
            properties = torch.cuda.get_device_properties(torch.device(device))
            gpu[device] = dict(name=properties.name, total_memory=properties.total_memory,
                               capability=[properties.major,properties.minor], uuid=str(getattr(properties,'uuid','unavailable')))
    cpu_name = next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                     if line.startswith('model name')), platform.processor())
    value = dict(version='evaluation_v1', config={k:v for k,v in cfg.items() if k != 'output_dir'},
                 cases=cases, data=data, runtime=runtime_identity(),
                 hardware=dict(cpu=cpu_name, logical_cpus=os.cpu_count(), gpu=gpu,
                               platform=platform.platform(), cuda=torch.version.cuda),
                 files=[file_identity(p) for p in sorted({str(Path(p).resolve()) for p in files})])
    return dict(sha256=digest(value), manifest=value)


def verified_summary(directory, job):
    directory = Path(directory)
    marker = directory/'complete.json'
    if not marker.exists():
        return None
    complete = json.loads(marker.read_text())
    if complete['job_sha256'] != hashlib.sha256(json.dumps(job, indent=2).encode()).hexdigest():
        raise ValueError(f'Completed job configuration changed: {directory}')
    for name, expected in complete['outputs'].items():
        if file_identity(directory/name)['sha256'] != expected:
            raise ValueError(f'Completed output changed: {directory/name}')
    return json.loads((directory/'summary.json').read_text())


def run_job(root, stage, name, case, cfg, data):
    is_test = stage == 'test'
    if stage == 'screen':
        indices = data['screen_indices']
    elif is_test:
        indices = data['test_indices']
    else:
        indices = data['validation_indices']
    job = dict(stage=stage, name=name, decoder=case['decoder'], tune=case['tune'] and not is_test,
               logits=data['test_path'] if is_test else data['validation_path'],
               indices=indices,
               alphas=alpha_values(**cfg['alphas']) if not is_test else [],
               length_penalties=cfg['length_penalties'] if not is_test else [],
               timing_repeats=0 if stage == 'screen' else cfg['timing_repeats'],
               **{key:cfg[key] for key in ('seed','torch_threads','warmup_sentences','reorder_logit_columns',
                                          'bootstrap_group','bootstrap_samples','timing_limit')})
    directory = root/stage/name
    existing = verified_summary(directory, job)
    if existing is not None:
        print(f'REUSE completed measurement: {stage}/{name}', flush=True)
        return existing
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory/'job.json', job)
    print(f'START {stage}/{name}: {len(job["indices"])} utterances', flush=True)
    try:
        with (directory/'worker.log').open('w') as log:
            subprocess.run([sys.executable, str(Path(__file__).with_name('evaluation_worker.py')),
                            '--job', str(directory/'job.json')], stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=cfg['worker_timeout_seconds'])
    except (subprocess.SubprocessError, OSError) as exc:
        write_json(directory/'failure.json', dict(error=str(exc), stage=stage, name=name))
        print(f'FAILED {stage}/{name}; see {directory}/worker.log', flush=True)
        return None
    result = verified_summary(directory, job)
    if result is None:
        raise RuntimeError(f'Worker exited without a completed result: {directory}')
    (directory/'failure.json').unlink(missing_ok=True)
    print(f"DONE {stage}/{name}: WER {100*result['wer']:.3f}%, mean {1000*result['timing']['mean_seconds']:.1f} ms", flush=True)
    return result


def measure_validation_stage(root, stage, cases, cfg, data):
    """Measure screening or full validation; optional failures remain in the report."""
    results = []
    for name, case in cases.items():
        result = run_job(root, stage, name, case, cfg, data)
        if result is not None:
            results.append(result)
        elif name == cfg['baseline']:
            label = 'screening' if stage == 'screen' else 'validation'
            raise RuntimeError(f'Baseline {label} failed; see worker log')
        build_report(root, cfg, data)
    return results


def save_selection(root, selection):
    path = root/'selection.json'
    value = dict(selection, sha256=digest(selection))
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('Selection is already frozen; use a new output directory to change it')
    write_json(path, value)


def load_selection(root, identity):
    value = json.loads((root/'selection.json').read_text())
    expected = value.pop('sha256')
    if digest(value) != expected or value['experiment_sha256'] != identity:
        raise ValueError('Frozen selection or experiment identity changed')
    return value


def build_report(root, cfg, data, selection=None):
    all_results = {}
    for stage in ('screen','validation','test'):
        all_results[stage] = [json.loads(path.read_text()) for path in sorted((root/stage).glob('*/summary.json'))
                              if path.with_name('complete.json').is_file()]
    comparisons = {}
    if all_results['test']:
        baseline_path = root/'test'/cfg['baseline']/'candidates.json'
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text())
            for result in all_results['test']:
                candidate = json.loads((root/'test'/result['name']/'candidates.json').read_text())
                comparisons[result['name']] = paired_comparison(candidate, baseline, cfg['bootstrap_samples'], cfg['seed'])
    failures = [json.loads(path.read_text()) for path in root.glob('*/*/failure.json')]
    report = dict(objective=cfg['objective'], data=data, selection=selection, results=all_results,
                   validation_pareto=[r['name'] for r in pareto_front(all_results['validation'])],
                   test_paired_vs_baseline=comparisons, failures=failures)
    write_json(root/'report.json', report)
    fields = ['stage','name','wer_percent','ngram_wer_percent','oracle_wer_percent','mean_ms','p50_ms','p95_ms',
              'score','alpha','length_penalty','peak_rss_gib','peak_vram_allocated_gib','words','edits']
    csv_rows = []
    lines = ['# Decoder validation and development-test results', '',
             f"Objective: `{cfg['objective']['wer']} × WER_fraction + {cfg['objective']['latency']} × mean_seconds` (lower is better).", '',
             f"Validation: {len(data['validation_indices'])} utterances / {data['validation_words']} words; "
             f"test: {len(data['test_indices'])} utterances / {data['test_words']} words.",
             f"Split: {data['split']}; shared normalized reference identities: {data['shared_reference_identities']}.", '',
             'Choices use validation only. Test scores below do not change the selected configuration.', '']
    if selection:
        lines += [f"Recommended by validation: **{selection['roles']['recommended']}**. "
                  f"Accuracy choice: **{selection['roles']['accuracy']}**; fastest Pareto choice: **{selection['roles']['fastest']}**.", '']
    for stage, results in all_results.items():
        lines += [f'## {stage.title()}', '', '| Configuration | WER % (edits) | N-gram / oracle % | Mean / p50 / p95 ms | Score | Alpha / penalty | RAM / VRAM GiB |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        # Keep test display in name order to avoid presenting a test-selected winner.
        ordered = results
        if stage != 'test':
            ordered = sorted(results, key=lambda r: objective(
                r['wer'], r['timing']['mean_seconds'], cfg['objective']))
        for r in ordered:
            t = r['timing']
            score = objective(r['wer'], t['mean_seconds'], cfg['objective'])
            vram = f"{r['peak_vram_allocated_gib']:.2f}" if r['peak_vram_allocated_gib'] is not None else '—'
            lines.append(f"| {r['name']} | {100*r['wer']:.3f} ({r['edits']}) | {100*r['ngram_wer']:.3f} / {100*r['oracle_wer']:.3f} | "
                         f"{1000*t['mean_seconds']:.1f} / {1000*t['p50_seconds']:.1f} / {1000*t['p95_seconds']:.1f} | {score:.4f} | "
                         f"{r['alpha']:.2f} / {r['length_penalty']:.2f} | {r['peak_rss_gib']:.2f} / {vram} |")
            csv_row = {
                'stage': stage,
                'name': r['name'],
                'wer_percent': 100 * r['wer'],
                'ngram_wer_percent': 100 * r['ngram_wer'],
                'oracle_wer_percent': 100 * r['oracle_wer'],
                'mean_ms': 1000 * t['mean_seconds'],
                'p50_ms': 1000 * t['p50_seconds'],
                'p95_ms': 1000 * t['p95_seconds'],
                'score': score,
            }
            for key in fields:
                if key in r and key not in ('stage', 'name'):
                    csv_row[key] = r[key]
            csv_rows.append(csv_row)
        lines.append('')
    if comparisons:
        lines += ['## Paired test differences versus the fixed baseline', '',
                  f"95% percentile bootstrap intervals resample `{cfg['bootstrap_group']}` groups. Negative differences favor the candidate.", '',
                  '| Configuration | WER difference (percentage points) | 95% interval | Groups |', '|---|---:|---:|---:|']
        for name, result in comparisons.items():
            ci = result['interval']
            interval = 'unavailable' if ci['low'] is None else f"[{100*ci['low']:.3f}, {100*ci['high']:.3f}]"
            lines.append(f"| {name} | {100*result['wer_difference']:.3f} | {interval} | {ci['groups']} |")
        lines.append('')
    lines += ['## Interpretation and files', '',
              '- This is a bounded candidate search, not a globally optimal setting or a production accuracy claim.',
              '- Screening tunes scoring weights on a fixed validation subset; finalists retune on full validation. The baseline keeps its fixed policy.',
              '- Model load and warmup are separate. Finalist timings use the same randomly selected utterances and repeat counts for each configuration, serially.',
              '- RAM is process peak RSS including model loading; VRAM is peak PyTorch allocated memory. These are not total system/device usage.',
              '- Bootstrap intervals describe this set of groups; few groups and validation selection limit their interpretation.',
              '- JSON contains per-group WER, WER intervals, OOV/empty/candidate metrics, load times, retained frames, full effective configs and failures.',
              '- Each job saves candidates, raw scores/counts, scoring curves, timings and content-checked completion metadata. No runtime score cache is added.',
              '- `selection.json` freezes validation choices; `selected_configs/` contains decoder and sweep configurations for reuse.', '']
    if failures:
        lines += ['Failed measurements: '+', '.join(f"{r['stage']}/{r['name']}" for r in failures)+'. See worker logs.', '']
    (root/'report.md').write_text('\n'.join(lines))
    with (root/'report.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    if selection:
        plot_report(root, report)
    return report


def plot_report(root, report):
    os.environ.setdefault('MPLCONFIGDIR', str(root/'.matplotlib'))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('Plot skipped: install matplotlib to generate accuracy_latency.svg', flush=True)
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    chosen = report['selection']['roles']['recommended']
    for axis, stage in zip(axes, ('validation','test')):
        if not report['results'][stage]:
            axis.text(.5, .5, 'Not evaluated yet', ha='center', va='center', transform=axis.transAxes)
            axis.set_axis_off()
            continue
        for row in report['results'][stage]:
            color = '#1565c0' if row['name'] == chosen else '#666666'
            x, y = 1000*row['timing']['mean_seconds'], 100*row['wer']
            axis.scatter(x, y, c=color, s=60)
            axis.annotate(row['name'], (x,y), xytext=(5,7), textcoords='offset points', fontsize=8)
        axis.set(title='Validation (selection)' if stage=='validation' else 'Development test (frozen choices)',
                 xlabel='Mean decoder latency (ms)', ylabel='Word error rate (%)')
        axis.grid(alpha=.2)
        axis.margins(x=.25, y=.3)
        if axis.get_ylim()[0] < 0:
            axis.set_ylim(bottom=0)
        if axis.get_xlim()[0] < 0:
            axis.set_xlim(left=0)
    fig.savefig(root/'accuracy_latency.svg')
    plt.close(fig)


def export_configs(root, selection, cfg, data):
    for role, name in selection['roles'].items():
        decoder = selection['configurations'][name]['decoder']
        write_json(root/'selected_configs'/f'{role}_decoder.json', decoder)
        sweep = {**decoder, 'llm_alpha':0., 'logits_pkl_path':data['validation_path'],
                 'reorder_logit_columns':cfg['reorder_logit_columns'],
                 'output_dir':str(root/f'{role}_sweep'), 'study_name':f'{role}_followup', 'n_trials':1,
                 'parameters':{'beam_size':{'type':'fixed','value':decoder['beam_size']}},
                 'alpha_range':dict(low=decoder['llm_alpha'], high=decoder['llm_alpha'], step=.05),
                 'objective_weights':{'wer':cfg['objective']['wer'], 'time':cfg['objective']['latency']}}
        OmegaConf.save(OmegaConf.create(sweep), root/'selected_configs'/f'{role}_sweep.yaml')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--validation-logits')
    parser.add_argument('--test-logits')
    parser.add_argument('--output')
    parser.add_argument('--device', help='Override every configuration device, e.g. cuda:0')
    parser.add_argument('--split-test-fraction', type=float, help='Split validation file instead of using a test file')
    parser.add_argument('--screen-limit', type=int, help='Override screening size; 0 means all')
    parser.add_argument('--validation-limit', type=int, help='Override validation size; 0 means all')
    parser.add_argument('--test-limit', type=int, help='Override test size; 0 means all')
    parser.add_argument('--phase', choices=['all','validation','test','report'], default='all')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Validate configs and splits without model loads or measurements')
    args = parser.parse_args()
    if args.phase == 'report':
        raw = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
        root = Path(args.output or raw['output_dir']).expanduser().resolve()
        stored = json.loads((root/'manifest.json').read_text())
        if digest(stored['manifest']) != stored['sha256']:
            raise ValueError('Saved experiment manifest changed')
        selection = load_selection(root, stored['sha256']) if (root/'selection.json').exists() else None
        for marker in root.glob('*/*/complete.json'):
            verified_summary(marker.parent, json.loads(marker.with_name('job.json').read_text()))
        build_report(root, stored['manifest']['config'], stored['manifest']['data'], selection)
        print(f"Regenerated {root/'report.md'} from verified saved measurements; no model/data load.")
        return
    overrides = {key:getattr(args,key) for key in ('validation_logits','test_logits') if getattr(args,key) is not None}
    if args.output:
        overrides['output_dir'] = args.output
    for key in ('screen_limit','validation_limit','test_limit'):
        if getattr(args,key) is not None:
            overrides[key] = getattr(args,key) or None
    if args.split_test_fraction is not None:
        overrides.update(test_logits=None, test_fraction=args.split_test_fraction)
    cfg, cases = load_config(args.config, overrides)
    if args.device:
        for case in cases.values():
            case['decoder']['llm_device'] = args.device
    data = prepare_data(cfg)
    if args.dry_run:
        print(json.dumps(dict(configurations=list(cases), objective=cfg['objective'], data=data), indent=2))
        return
    root = Path(cfg['output_dir'])
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise ValueError('Output directory is not empty; use --resume or choose a new directory')
    print('Hashing data, models, runtime and source for a reproducible experiment...', flush=True)
    identity = experiment_identity(cfg, cases, data)
    manifest_path = root/'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())['sha256'] != identity['sha256']:
            raise ValueError('Experiment identity changed; use a new output directory')
    elif args.phase in ('test','report'):
        raise ValueError('Run validation and freeze a selection before test/report')
    write_json(manifest_path, identity)
    write_json(root/'split.json', data)
    if args.phase in ('all','validation'):
        screening = measure_validation_stage(root, 'screen', cases, cfg, data)
        names = choose_finalists(screening, cfg['baseline'], cfg['finalists'], cfg['objective'])
        finalists = {name: cases[name] for name in names}
        validation = measure_validation_stage(root, 'validation', finalists, cfg, data)
        roles = freeze_selection(validation, cfg['baseline'], cfg['objective'])
        selected = {r['name']:dict(decoder=r['decoder'], tune=False) for r in validation if r['name'] in roles.values()}
        selection = dict(experiment_sha256=identity['sha256'], roles=roles, configurations=selected,
                         validation_objective=cfg['objective'])
        save_selection(root, selection)
        export_configs(root, selection, cfg, data)
    else:
        selection = load_selection(root, identity['sha256'])
    if args.phase in ('all','test'):
        # Selection is sealed on disk before the first test decode starts.
        selection = load_selection(root, identity['sha256'])
        for name, case in selection['configurations'].items():
            if run_job(root, 'test', name, case, cfg, data) is None:
                build_report(root, cfg, data, selection)
                raise RuntimeError(f'Test measurement failed for frozen configuration {name}; rerun with --resume')
            build_report(root, cfg, data, selection)
    build_report(root, cfg, data, selection)
    print(f"Finished. Recommended by validation: {selection['roles']['recommended']}. Report: {root/'report.md'}", flush=True)


if __name__ == '__main__':
    main()
