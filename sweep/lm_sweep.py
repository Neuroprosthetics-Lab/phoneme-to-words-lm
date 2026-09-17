"""
Optuna hyperparameter sweep for the LM decoder.

Optimizes a weighted combination of WER (accuracy) and decode time (speed)
as a single scalar objective. Each trial is executed as a subprocess
(lm_sweep_worker.py) for GPU isolation.

Usage:
    python sweep/lm_sweep.py --sweep_config lm_sweep.yaml --devices cuda:0

Examples:
    # Single GPU sweep
    python sweep/lm_sweep.py --sweep_config my_lm_sweep.yaml --devices cuda:0

    # 2 concurrent jobs on GPU 0, 1 on GPU 1 (3 total)
    python sweep/lm_sweep.py --sweep_config my_lm_sweep.yaml --devices cuda:0:2,cuda:1

    # Resume a previous sweep
    python sweep/lm_sweep.py --sweep_config my_lm_sweep.yaml --resume
"""

import argparse
import json
import math
import warnings
import os
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import optuna
from omegaconf import OmegaConf

from phoneme_to_words_lm.sweep_utils import (
    GPUPool,
    suggest_param,
    enqueue_initial_configs,
    _build_grid_search_space,
)
from phoneme_to_words_lm.sweep_contract import (
    decoder_defaults, resolve_decoder_config, validate_parameter, alpha_values,
    positive_integer, run_identity, verify_resume, RESOURCE_FIELDS,
)


class WorkerExecutionError(RuntimeError):
    """A failed subprocess or invalid result, recorded as an Optuna FAIL."""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Optuna sweep for the LM decoder (weighted WER + speed)")
    parser.add_argument("--sweep_config", type=str, required=True,
                        help="Path to LM sweep config YAML")
    parser.add_argument("--devices", type=str, default="cuda:0",
                        help="Comma-separated CUDA devices with optional concurrency, "
                             "e.g. cuda:0:3,cuda:1 (3 jobs on GPU 0, 1 on GPU 1)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing SQLite DB")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Device spec parsing
# ---------------------------------------------------------------------------

def parse_device_specs(devices_str):
    """Parse device string into list of (gpu_id, n_jobs) tuples.

    Format: cuda:<id>[:<n_jobs>], comma-separated.
    Examples: "cuda:0" -> [(0,1)], "cuda:0:3,cuda:1" -> [(0,3),(1,1)]
    """
    device_strs = [d.strip() for d in devices_str.split(",")]
    specs = []
    for d in device_strs:
        parts = d.split(":")
        if parts[0] != "cuda" or len(parts) not in (2, 3):
            raise ValueError(f"Invalid device format: {d}. Expected 'cuda:<id>[:<n_jobs>]'.")
        gpu_id = int(parts[1])
        n_jobs = int(parts[2]) if len(parts) > 2 else 1
        if gpu_id < 0 or any(g == gpu_id for g, _ in specs):
            raise ValueError("GPU indices must be nonnegative and unique")
        if n_jobs < 1:
            raise ValueError(f"Concurrency for cuda:{gpu_id} must be >= 1, got {n_jobs}")
        specs.append((gpu_id, n_jobs))
    return specs


# ---------------------------------------------------------------------------
# Sweep config loading
# ---------------------------------------------------------------------------

def load_lm_sweep_config(path):
    """Load and validate LM sweep config YAML."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Sweep config not found: {path}")
    cfg = OmegaConf.load(path)

    # Validate required fields
    required = ["lexicon_path", "tokens_path", "kenlm_model_path", "logits_pkl_path"]
    for field in required:
        if field not in cfg:
            raise ValueError(f"sweep config must contain '{field}'")
    if "parameters" not in cfg or len(cfg.parameters) == 0:
        raise ValueError("sweep config must contain at least one entry in 'parameters'")

    # Validate file paths exist
    for field in required:
        fpath = str(Path(cfg[field]).expanduser().resolve())
        cfg[field] = fpath
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"{field} not found: {fpath}")

    # Defaults
    cfg.setdefault("search_strategy", "tpe")
    cfg.setdefault("n_trials", 50)
    cfg.setdefault("output_dir", os.path.join(os.path.dirname(path), "lm_sweep_output"))
    cfg.setdefault("study_name", "lm_decoder_sweep")
    cfg.setdefault("n_startup_trials", 10)
    cfg.setdefault("eval_every_nth", 1)
    cfg.setdefault("reorder_logit_columns", True)
    cfg.setdefault("llm_cache_dir", None)  # resolved to HF_CACHE_DIR at runtime
    cfg.setdefault("llm_dtype", "bfloat16")
    cfg.setdefault("llm_batch_size", 100)
    cfg.setdefault("llm_prefix", "\n")
    cfg.setdefault("llm_score_eos", False)
    cfg.setdefault("llm_max_length", 512)
    cfg.setdefault("llm_max_batch_tokens", 2048)
    cfg.setdefault("llm_logprob_chunk_size", 16)

    # Alpha range defaults
    if "alpha_range" not in cfg:
        cfg.alpha_range = OmegaConf.create({"low": 0.0, "high": 3.0, "step": 0.05})
    else:
        cfg.alpha_range.setdefault("low", 0.0)
        cfg.alpha_range.setdefault("high", 3.0)
        cfg.alpha_range.setdefault("step", 0.05)

    # Objective weights (for weighted ranking of results)
    if "objective_weights" not in cfg or cfg.objective_weights is None:
        cfg.objective_weights = OmegaConf.create({"wer": 1.0, "time": 1.0})
    else:
        cfg.objective_weights.setdefault("wer", 1.0)
        cfg.objective_weights.setdefault("time", 1.0)
        if cfg.objective_weights.wer < 0 or cfg.objective_weights.time < 0:
            raise ValueError("objective_weights values must be non-negative")

    # Validate initial_configs structure if present
    if "initial_configs" in cfg and cfg.initial_configs is not None:
        if not OmegaConf.is_list(cfg.initial_configs):
            raise ValueError("'initial_configs' must be a list")
        for i, entry in enumerate(cfg.initial_configs):
            if "params" not in entry:
                name = entry.get("name", f"entry {i}")
                raise ValueError(
                    f"initial_configs: '{name}' is missing required 'params' key"
                )

    cfg.setdefault("timing_warmup_sentences", 1)
    cfg.setdefault("max_sentences", None)
    cfg.setdefault("worker_timeout_seconds", None)
    cfg.setdefault("seed", 42)
    cfg.output_dir = str(Path(cfg.output_dir).expanduser().resolve())
    plain = OmegaConf.to_container(cfg, resolve=True)
    allowed = set(decoder_defaults()) | set(RESOURCE_FIELDS)
    controls = {'parameters', 'search_strategy', 'n_trials', 'output_dir', 'study_name',
                'n_startup_trials', 'eval_every_nth', 'reorder_logit_columns', 'alpha_range',
                'objective_weights', 'initial_configs', 'logits_pkl_path', 'timing_warmup_sentences',
                'max_sentences', 'worker_timeout_seconds', 'seed'}
    unknown = set(plain) - allowed - controls
    if unknown:
        raise ValueError(f'Unknown sweep configuration fields: {sorted(unknown)}')
    for key in ('n_trials', 'eval_every_nth'):
        positive_integer(key, plain[key])
    for key in ('n_startup_trials', 'timing_warmup_sentences', 'seed'):
        positive_integer(key, plain[key], allow_zero=True)
    if plain['max_sentences'] is not None:
        positive_integer('max_sentences', plain['max_sentences'])
    if plain['worker_timeout_seconds'] is not None:
        positive_integer('worker_timeout_seconds', plain['worker_timeout_seconds'])
    if not isinstance(plain['reorder_logit_columns'], bool):
        raise ValueError('reorder_logit_columns must be boolean')
    if plain['search_strategy'] not in ('tpe', 'grid', 'nsga2', 'random'):
        raise ValueError('Unknown search_strategy')
    if plain.get('llm_alpha', 0) != 0:
        raise ValueError('Use alpha_range for sweep selection; llm_alpha must be omitted or zero')
    if set(plain['alpha_range']) != {'low', 'high', 'step'}:
        raise ValueError('alpha_range supports only low, high and step')
    alpha_values(**{k: plain['alpha_range'][k] for k in ('low', 'high', 'step')})
    weights = plain['objective_weights']
    if set(weights) != {'wer', 'time'} or any(isinstance(v, bool) or not isinstance(v, (int, float))
        or not math.isfinite(v) or v < 0 for v in weights.values()) or not any(weights.values()):
        raise ValueError('Objective weights must be finite, nonnegative and not both zero')
    for name, spec in plain['parameters'].items():
        if name not in allowed or name in RESOURCE_FIELDS + ('llm_alpha', 'llm_device'):
            raise ValueError(f'Unsupported swept parameter: {name}')
        for value in validate_parameter(name, spec):
            resolve_decoder_config(plain, {name: value})
    effective = resolve_decoder_config(plain)
    for key in RESOURCE_FIELDS + ('llm_lora_path', 'hotwords_path'):
        if key in plain:
            cfg[key] = effective[key]
    # Validate supplied initial values before a trial acquires a device.
    from phoneme_to_words_lm.sweep_utils import encode_param_value
    for entry in plain.get('initial_configs') or []:
        for name, value in entry['params'].items():
            if name not in plain['parameters']:
                raise ValueError(f'Unknown initial parameter: {name}')
            spec = plain['parameters'][name]
            if spec['type'] == 'fixed' and value != spec['value']:
                raise ValueError(f'Initial parameter {name} differs from its fixed value')
            encode_param_value(name, spec, value)
        resolve_decoder_config(plain, entry['params'])
    return cfg


# ---------------------------------------------------------------------------
# Sampler creation
# ---------------------------------------------------------------------------

def create_lm_sampler(strategy, parameters=None, n_initial_configs=0, n_startup_trials=10, seed=42):
    """Create an Optuna sampler for optimization."""
    strategy = strategy.lower()
    if strategy == "tpe":
        effective_startup = max(0, n_startup_trials - n_initial_configs)
        return optuna.samplers.TPESampler(n_startup_trials=effective_startup, seed=seed)
    elif strategy == "grid":
        search_space = _build_grid_search_space(parameters)
        return optuna.samplers.GridSampler(search_space, seed=seed)
    elif strategy == "nsga2":
        return optuna.samplers.NSGAIISampler(seed=seed)
    elif strategy == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    else:
        raise ValueError(f"Unknown search strategy: {strategy}. "
                         f"Supported: tpe, grid, nsga2, random")


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def make_lm_objective(sweep_cfg, gpu_pool, worker_script_path, gpu_concurrency):
    """Return a single-objective closure for study.optimize().

    Returns a weighted combination of WER and decode time:
        score = wer_weight * wer + time_weight * avg_decode_time

    Args:
        gpu_concurrency: dict mapping gpu_id -> n_jobs, used to compute
            per-process VRAM fraction (e.g. 2 jobs on one GPU -> 0.5 each).
    """
    parameters = sweep_cfg.parameters
    wer_weight = float(sweep_cfg.objective_weights.wer)
    time_weight = float(sweep_cfg.objective_weights.time)

    def objective(trial):
        # Suggest hyperparameters
        param_values = {}
        for name, spec in parameters.items():
            param_values[name] = suggest_param(trial, name, spec)

        base_config = OmegaConf.to_container(sweep_cfg, resolve=True)
        effective = resolve_decoder_config(base_config, param_values)
        trial.set_user_attr('effective_decoder_config', effective)
        trial_dir = os.path.join(sweep_cfg.output_dir, f"trial_{trial.number}")
        gpu_id = gpu_pool.acquire()
        try:
            os.makedirs(trial_dir, exist_ok=True)

            print(f"\n{'=' * 60}")
            print(f"Trial {trial.number} | GPU {gpu_id} | Params: {param_values}")
            print(f"{'=' * 60}\n")

            # Build worker config JSON
            n_jobs_on_gpu = gpu_concurrency[gpu_id]
            vram_fraction = 1.0 / n_jobs_on_gpu if n_jobs_on_gpu > 1 else None

            effective['llm_device'] = f'cuda:{gpu_id}'
            worker_config = {
                **base_config,
                "decoder_config": effective,
                "llm_device": f"cuda:{gpu_id}",
                "alpha_low": float(sweep_cfg.alpha_range.low),
                "alpha_high": float(sweep_cfg.alpha_range.high),
                "alpha_step": float(sweep_cfg.alpha_range.step),
                "max_vram_fraction": vram_fraction,
                "trial_params": dict(param_values),
                "output_path": os.path.join(trial_dir, "results.json"),
                "gpu_concurrency": n_jobs_on_gpu,
            }
            trial.set_user_attr('effective_decoder_config', effective)
            trial.set_user_attr('gpu_concurrency', n_jobs_on_gpu)
            # A retry must not accidentally consume a stale success artifact.
            Path(worker_config['output_path']).unlink(missing_ok=True)

            config_path = os.path.join(trial_dir, "worker_config.json")
            with open(config_path, "w") as f:
                json.dump(worker_config, f, indent=2)

            # Spawn worker subprocess
            try:
                result = subprocess.run(
                    [sys.executable, worker_script_path, "--config", config_path],
                    capture_output=True,
                    text=True,
                    timeout=sweep_cfg.worker_timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                for name, value in (('stdout', exc.stdout), ('stderr', exc.stderr)):
                    Path(trial_dir, name+'.log').write_text(
                        value.decode(errors='replace') if isinstance(value, bytes) else value or '')
                raise WorkerExecutionError(f'Worker timed out; see {trial_dir}') from exc

            # Save stdout/stderr
            with open(os.path.join(trial_dir, "stdout.log"), "w") as f:
                f.write(result.stdout)
            with open(os.path.join(trial_dir, "stderr.log"), "w") as f:
                f.write(result.stderr)

            # Check for failure
            results_path = os.path.join(trial_dir, "results.json")
            if result.returncode != 0 or not os.path.isfile(results_path):
                error_msg = (f"Worker failed (rc={result.returncode}). "
                             f"See {trial_dir}/stderr.log")
                with open(os.path.join(trial_dir, "error.log"), "w") as f:
                    f.write(error_msg + "\n")
                    f.write(f"STDERR:\n{result.stderr}\n")
                print(f"  Trial {trial.number} FAILED: {error_msg}")
                raise WorkerExecutionError(error_msg)

            # Parse results
            try:
                with open(results_path, "r") as f:
                    results = json.load(f)
                for key in ('wer', 'avg_decode_time', 'best_alpha'):
                    if isinstance(results[key], bool) or not isinstance(results[key], (int, float)) or not math.isfinite(results[key]) or results[key] < 0:
                        raise ValueError(f'Invalid {key}')
                for key in ('n_sentences_evaluated', 'n_words_total'):
                    positive_integer(key, results[key])
                positive_integer('n_edits_total', results['n_edits_total'], allow_zero=True)
            except (KeyError, ValueError, TypeError) as exc:
                raise WorkerExecutionError(f'Invalid worker results in {results_path}: {exc}') from exc

            wer = results["wer"]
            avg_decode_time = results["avg_decode_time"]

            # Store individual metrics and metadata as trial attributes
            trial.set_user_attr("wer", wer)
            trial.set_user_attr("avg_decode_time", avg_decode_time)
            trial.set_user_attr("best_alpha", results["best_alpha"])
            trial.set_user_attr("n_sentences", results["n_sentences_evaluated"])
            trial.set_user_attr("n_words", results["n_words_total"])
            trial.set_user_attr("n_edits", results["n_edits_total"])

            for key in ('ngram_wer', 'oracle_wer', 'empty_rate', 'median_candidates',
                        'p50_decode_time', 'p95_decode_time', 'sentences_per_second',
                        'load_seconds', 'first_decode_seconds', 'timing_warmup_sentences', 'reference_oov_rate'):
                if key in results:
                    trial.set_user_attr(key, results[key])
            score = wer_weight * wer + time_weight * avg_decode_time
            print(f"  Trial {trial.number} | WER = {wer:.5f} | "
                  f"Time = {avg_decode_time:.4f}s/sent | "
                  f"Score = {score:.5f} | "
                  f"Alpha = {results['best_alpha']:.3f}")

            return score

        finally:
            gpu_pool.release(gpu_id)

    return objective


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def generate_lm_visualizations(study, output_dir):
    """Generate single-objective visualizations."""
    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
        )
    except ImportError:
        print("Install plotly for visualizations: pip install plotly")
        return

    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < 2:
        print("Not enough completed trials for visualizations.")
        return

    # Optimization history (weighted score)
    try:
        fig = plot_optimization_history(study)
        fig.write_html(os.path.join(vis_dir, "optimization_history.html"))
        print("  Saved optimization_history.html")
    except Exception as e:
        print(f"  Could not generate optimization_history: {e}")

    # Per-metric optimization history via user attributes
    for attr, label in [("wer", "WER"), ("avg_decode_time", "Avg Decode Time (s)")]:
        try:
            fig = plot_optimization_history(
                study,
                target=lambda t, a=attr: t.user_attrs[a],
                target_name=label,
            )
            fig.write_html(os.path.join(vis_dir, f"optimization_history_{attr}.html"))
            print(f"  Saved optimization_history_{attr}.html")
        except Exception as e:
            print(f"  Could not generate optimization_history_{attr}: {e}")

    # Param importances (weighted score)
    try:
        fig = plot_param_importances(study)
        fig.write_html(os.path.join(vis_dir, "param_importances.html"))
        print("  Saved param_importances.html")
    except Exception as e:
        print(f"  Could not generate param_importances: {e}")

    # Parallel coordinate (colored by WER)
    try:
        fig = plot_parallel_coordinate(
            study,
            target=lambda t: t.user_attrs["wer"],
            target_name="WER",
        )
        fig.write_html(os.path.join(vis_dir, "parallel_coordinate.html"))
        print("  Saved parallel_coordinate.html")
    except Exception as e:
        print(f"  Could not generate parallel_coordinate: {e}")


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def print_lm_results_table(study, objective_weights=None):
    """Print results table sorted by objective score (lower is better).

    Args:
        objective_weights: dict with 'wer' and 'time' keys (for display).
            If None, defaults to equal weights (1.0, 1.0).
    """
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("No completed trials.")
        return

    if objective_weights is None:
        objective_weights = {"wer": 1.0, "time": 1.0}
    wer_weight = float(objective_weights["wer"])
    time_weight = float(objective_weights["time"])

    # Sort by objective value (lower is better)
    completed.sort(key=lambda t: t.value)

    param_names = sorted({k for t in completed for k in t.params.keys()})

    # Header
    header = (f"{'Rank':<5} {'Trial':<7} {'Score':<10} {'WER':<10} {'Time(s)':<10} "
              f"{'Alpha':<8} ")
    header += " ".join(f"{p:<20}" for p in param_names)
    sep = "=" * len(header)

    print(f"\nObjective: score = WER * {wer_weight} + Time * {time_weight}")

    print(f"\n{sep}")
    print("ALL TRIALS (sorted by score, best first)")
    print(sep)
    print(header)
    print("-" * len(header))
    for rank, trial in enumerate(completed, 1):
        _print_trial_row(rank, trial, param_names)

    print()
    pruned = [t for t in study.trials
              if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials
              if t.state == optuna.trial.TrialState.FAIL]
    print(f"Total trials: {len(study.trials)} | "
          f"Completed: {len(completed)} | "
          f"Pruned/Failed: {len(pruned) + len(failed)}")


def _print_trial_row(rank, trial, param_names):
    """Print a single row of the results table."""
    score = trial.value
    wer = trial.user_attrs.get("wer", float("nan"))
    decode_time = trial.user_attrs.get("avg_decode_time", float("nan"))
    alpha = trial.user_attrs.get("best_alpha", "N/A")
    alpha_str = f"{alpha:<8.3f}" if isinstance(alpha, float) else f"{alpha:<8}"
    params_str = " ".join(
        f"{str(trial.params.get(p, 'N/A')):<20}" for p in param_names
    )
    config_label = ""
    if "initial_config_name" in trial.user_attrs:
        config_label = f" [{trial.user_attrs['initial_config_name']}]"
    print(f"{rank:<5} {trial.number:<7} {score:<10.5f} {wer:<10.5f} {decode_time:<10.4f} "
          f"{alpha_str}{params_str}{config_label}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Parse devices
    device_specs = parse_device_specs(args.devices)
    total_jobs = sum(n for _, n in device_specs)
    print(f"Using {len(device_specs)} GPU(s): "
          + ", ".join(f"cuda:{gid} x{nj}" for gid, nj in device_specs)
          + f" ({total_jobs} total concurrent jobs)")

    # Load sweep config
    sweep_cfg = load_lm_sweep_config(args.sweep_config)
    os.makedirs(sweep_cfg.output_dir, exist_ok=True)

    # Worker script path
    worker_script_path = str(
        Path(__file__).resolve().parent / "lm_sweep_worker.py"
    )

    if total_jobs > 1:
        warnings.warn('Concurrent workers contend for CPU/GPU resources. Use these runs for accuracy exploration; benchmark finalists separately for latency.')
    print('Fingerprinting input, model, native and source files for safe resume...', flush=True)
    identity = run_identity(OmegaConf.to_container(sweep_cfg, resolve=True), device_specs)
    sweep_cfg.resolved_model_paths = OmegaConf.create(identity['model_snapshots'])

    # Create or load study
    db_path = os.path.join(sweep_cfg.output_dir, "sweep_study.db")
    storage = f"sqlite:///{db_path}"

    initial_configs = (list(sweep_cfg.initial_configs)
                       if "initial_configs" in sweep_cfg and sweep_cfg.initial_configs
                       else [])
    n_initial_configs = len(initial_configs)

    sampler = create_lm_sampler(
        sweep_cfg.search_strategy,
        parameters=sweep_cfg.parameters,
        n_initial_configs=n_initial_configs,
        n_startup_trials=sweep_cfg.n_startup_trials,
        seed=sweep_cfg.seed,
    )

    if args.resume:
        print(f"Resuming study from {db_path}")
        study = optuna.load_study(
            study_name=sweep_cfg.study_name,
            storage=storage,
            sampler=sampler,
        )
    else:
        study = optuna.create_study(
            study_name=sweep_cfg.study_name,
            storage=storage,
            sampler=sampler,
            direction="minimize",
            load_if_exists=False,
        )

    verify_resume(study, identity, resume=args.resume)
    Path(sweep_cfg.output_dir, 'run_manifest.json').write_text(json.dumps(identity, indent=2))
    is_grid = sweep_cfg.search_strategy.lower() == "grid"
    if initial_configs and not is_grid:
        print(f"\nEnqueuing {n_initial_configs} initial config(s)...")
        enqueue_initial_configs(study, initial_configs, sweep_cfg.parameters)
    elif initial_configs and is_grid:
        print("Note: initial_configs ignored for grid search")

    gpu_pool = GPUPool(device_specs)
    gpu_concurrency = {gpu_id: n_jobs for gpu_id, n_jobs in device_specs}
    objective = make_lm_objective(sweep_cfg, gpu_pool, worker_script_path,
                                  gpu_concurrency)

    n_trials = sweep_cfg.n_trials
    if is_grid:
        search_space = _build_grid_search_space(sweep_cfg.parameters)
        n_trials = 1
        for vals in search_space.values():
            n_trials *= len(vals)
        if args.resume:
            n_trials = max(0, n_trials-len(study.trials))
        print(f"\nGrid search: {n_trials} remaining combinations")

    print(f"\nStarting LM decoder sweep: {sweep_cfg.study_name}")
    initial_info = (f" | Initial configs: {n_initial_configs}"
                    if n_initial_configs > 0 and not is_grid else "")
    print(f"Strategy: {sweep_cfg.search_strategy} | Trials: {n_trials} | "
          f"Concurrent jobs: {total_jobs}{initial_info}")
    print(f"Objective: minimize score = WER * {sweep_cfg.objective_weights.wer} "
          f"+ Time * {sweep_cfg.objective_weights.time}")
    print(f"Output: {sweep_cfg.output_dir}")
    print(f"DB: {db_path}\n")

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=total_jobs,
        catch=(WorkerExecutionError,),
    )

    # Print results
    obj_weights = OmegaConf.to_container(sweep_cfg.objective_weights)
    print_lm_results_table(study, objective_weights=obj_weights)

    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_lm_visualizations(study, sweep_cfg.output_dir)

    # Summary
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"\nSweep complete!")
    print(f"  SQLite DB: {db_path}")
    print(f"  Output dir: {sweep_cfg.output_dir}")
    if completed:
        best = study.best_trial
        print(f"  Best trial: {best.number} "
              f"(Score={best.value:.5f}, "
              f"WER={best.user_attrs['wer']:.5f}, "
              f"Time={best.user_attrs['avg_decode_time']:.4f}s)")
    print(f"\nTo explore interactively:")
    print(f"  pip install optuna-dashboard")
    print(f"  optuna-dashboard sqlite:///{db_path}")


if __name__ == "__main__":
    main()
