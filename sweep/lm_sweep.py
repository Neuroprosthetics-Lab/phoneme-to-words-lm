"""
Multi-objective Optuna hyperparameter sweep for the LM decoder.

Optimizes both WER (accuracy) and decode time (speed) using Optuna's
TPESampler with multi-objective support. Each trial is executed as a
subprocess (lm_sweep_worker.py) for GPU isolation.

Usage:
    python hyperparameter_sweep/lm_sweep.py --sweep_config lm_sweep.yaml --devices cuda:0

Examples:
    # Single GPU sweep
    python hyperparameter_sweep/lm_sweep.py --sweep_config my_lm_sweep.yaml --devices cuda:0

    # 2 concurrent jobs on GPU 0, 1 on GPU 1 (3 total)
    python hyperparameter_sweep/lm_sweep.py --sweep_config my_lm_sweep.yaml --devices cuda:0:2,cuda:1

    # Resume a previous sweep
    python hyperparameter_sweep/lm_sweep.py --sweep_config my_lm_sweep.yaml --resume
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import optuna
from omegaconf import OmegaConf

from phoneme_to_words_lm.sweep_utils import (
    GPUPool,
    suggest_param,
    enqueue_initial_configs,
    _build_grid_search_space,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-objective Optuna sweep for the LM decoder (WER + speed)")
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
        if parts[0] != "cuda" or len(parts) < 2:
            raise ValueError(f"Invalid device format: {d}. Expected 'cuda:<id>[:<n_jobs>]'.")
        gpu_id = int(parts[1])
        n_jobs = int(parts[2]) if len(parts) > 2 else 1
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
        fpath = os.path.expanduser(str(cfg[field]))
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
    cfg.setdefault("llm_cache_dir", "~/brand/huggingface")
    cfg.setdefault("llm_dtype", "bfloat16")
    cfg.setdefault("llm_batch_size", 100)

    # Alpha range defaults
    if "alpha_range" not in cfg:
        cfg.alpha_range = OmegaConf.create({"low": 0.0, "high": 3.0, "step": 0.05})
    else:
        cfg.alpha_range.setdefault("low", 0.0)
        cfg.alpha_range.setdefault("high", 3.0)
        cfg.alpha_range.setdefault("step", 0.05)

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

    return cfg


# ---------------------------------------------------------------------------
# Sampler creation
# ---------------------------------------------------------------------------

def create_lm_sampler(strategy, parameters=None, n_initial_configs=0, n_startup_trials=10):
    """Create an Optuna sampler for multi-objective optimization."""
    strategy = strategy.lower()
    if strategy == "tpe":
        effective_startup = max(0, n_startup_trials - n_initial_configs)
        return optuna.samplers.TPESampler(n_startup_trials=effective_startup)
    elif strategy == "grid":
        search_space = _build_grid_search_space(parameters)
        return optuna.samplers.GridSampler(search_space)
    elif strategy == "nsga2":
        return optuna.samplers.NSGAIISampler()
    elif strategy == "random":
        return optuna.samplers.RandomSampler()
    else:
        raise ValueError(f"Unknown search strategy: {strategy}. "
                         f"Supported: tpe, grid, nsga2, random")


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def make_lm_objective(sweep_cfg, gpu_pool, worker_script_path, gpu_concurrency):
    """Return a multi-objective closure for study.optimize().

    Args:
        gpu_concurrency: dict mapping gpu_id -> n_jobs, used to compute
            per-process VRAM fraction (e.g. 2 jobs on one GPU -> 0.5 each).
    """
    parameters = sweep_cfg.parameters

    def objective(trial):
        # Suggest hyperparameters
        param_values = {}
        for name, spec in parameters.items():
            param_values[name] = suggest_param(trial, name, spec)

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

            worker_config = {
                "lexicon_path": str(sweep_cfg.lexicon_path),
                "tokens_path": str(sweep_cfg.tokens_path),
                "kenlm_model_path": str(sweep_cfg.kenlm_model_path),
                "logits_pkl_path": str(sweep_cfg.logits_pkl_path),
                "llm_cache_dir": str(sweep_cfg.llm_cache_dir),
                "llm_device": f"cuda:{gpu_id}",
                "llm_dtype": str(sweep_cfg.llm_dtype),
                "llm_batch_size": int(sweep_cfg.llm_batch_size),
                "reorder_logit_columns": bool(sweep_cfg.reorder_logit_columns),
                "eval_every_nth": int(sweep_cfg.eval_every_nth),
                "alpha_low": float(sweep_cfg.alpha_range.low),
                "alpha_high": float(sweep_cfg.alpha_range.high),
                "alpha_step": float(sweep_cfg.alpha_range.step),
                "max_vram_fraction": vram_fraction,
                "trial_params": dict(param_values),
                "output_path": os.path.join(trial_dir, "results.json"),
            }

            config_path = os.path.join(trial_dir, "worker_config.json")
            with open(config_path, "w") as f:
                json.dump(worker_config, f, indent=2)

            # Spawn worker subprocess
            result = subprocess.run(
                [sys.executable, worker_script_path, "--config", config_path],
                capture_output=True,
                text=True,
            )

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
                raise optuna.TrialPruned(error_msg)

            # Parse results
            with open(results_path, "r") as f:
                results = json.load(f)

            wer = results["wer"]
            avg_decode_time = results["avg_decode_time"]

            # Store metadata as trial attributes
            trial.set_user_attr("best_alpha", results["best_alpha"])
            trial.set_user_attr("n_sentences", results["n_sentences_evaluated"])
            trial.set_user_attr("n_words", results["n_words_total"])
            trial.set_user_attr("n_edits", results["n_edits_total"])

            print(f"  Trial {trial.number} | WER = {wer:.5f} | "
                  f"Time = {avg_decode_time:.4f}s/sent | "
                  f"Alpha = {results['best_alpha']:.3f}")

            return wer, avg_decode_time

        finally:
            gpu_pool.release(gpu_id)

    return objective


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def generate_lm_visualizations(study, output_dir):
    """Generate multi-objective visualizations including Pareto front."""
    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
            plot_pareto_front,
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

    # Pareto front
    try:
        fig = plot_pareto_front(
            study,
            target_names=["WER", "Avg Decode Time (s)"],
        )
        fig.write_html(os.path.join(vis_dir, "pareto_front.html"))
        print("  Saved pareto_front.html")
    except Exception as e:
        print(f"  Could not generate pareto_front: {e}")

    # Per-objective optimization history
    for idx, name in enumerate(["wer", "time"]):
        try:
            fig = plot_optimization_history(
                study,
                target=lambda t, i=idx: t.values[i],
                target_name="WER" if idx == 0 else "Avg Decode Time (s)",
            )
            fig.write_html(os.path.join(vis_dir, f"optimization_history_{name}.html"))
            print(f"  Saved optimization_history_{name}.html")
        except Exception as e:
            print(f"  Could not generate optimization_history_{name}: {e}")

    # Param importances per objective
    for idx, name in enumerate(["wer", "time"]):
        try:
            fig = plot_param_importances(
                study,
                target=lambda t, i=idx: t.values[i],
                target_name="WER" if idx == 0 else "Avg Decode Time (s)",
            )
            fig.write_html(os.path.join(vis_dir, f"param_importances_{name}.html"))
            print(f"  Saved param_importances_{name}.html")
        except Exception as e:
            print(f"  Could not generate param_importances_{name}: {e}")

    # Parallel coordinate (colored by WER)
    try:
        fig = plot_parallel_coordinate(
            study,
            target=lambda t: t.values[0],
            target_name="WER",
        )
        fig.write_html(os.path.join(vis_dir, "parallel_coordinate.html"))
        print("  Saved parallel_coordinate.html")
    except Exception as e:
        print(f"  Could not generate parallel_coordinate: {e}")


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def _get_pareto_trial_numbers(study):
    """Return set of trial numbers on the Pareto front."""
    try:
        return {t.number for t in study.best_trials}
    except Exception:
        return set()


def print_lm_results_table(study):
    """Print a multi-objective results table with Pareto-optimal trials marked."""
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("No completed trials.")
        return

    # Sort by WER (first objective)
    completed.sort(key=lambda t: t.values[0])
    pareto_numbers = _get_pareto_trial_numbers(study)

    param_names = sorted({k for t in completed for k in t.params.keys()})

    # Header
    header = (f"{'Rank':<5} {'Trial':<7} {'WER':<10} {'Time(s)':<10} "
              f"{'Alpha':<8} ")
    header += " ".join(f"{p:<20}" for p in param_names)
    sep = "=" * len(header)

    # Pareto front section
    pareto_trials = [t for t in completed if t.number in pareto_numbers]
    if pareto_trials:
        print(f"\n{sep}")
        print("PARETO FRONT (sorted by WER, best first)")
        print(sep)
        print(header)
        print("-" * len(header))
        for rank, trial in enumerate(pareto_trials, 1):
            _print_trial_row(rank, trial, param_names, pareto=True)

    # All trials section
    print(f"\n{sep}")
    print("ALL TRIALS (sorted by WER, best first)")
    print(sep)
    print(header)
    print("-" * len(header))
    for rank, trial in enumerate(completed, 1):
        is_pareto = trial.number in pareto_numbers
        _print_trial_row(rank, trial, param_names, pareto=is_pareto)

    print()
    pruned = [t for t in study.trials
              if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials
              if t.state == optuna.trial.TrialState.FAIL]
    print(f"Total trials: {len(study.trials)} | "
          f"Completed: {len(completed)} | "
          f"Pareto-optimal: {len(pareto_numbers)} | "
          f"Pruned/Failed: {len(pruned) + len(failed)}")


def _print_trial_row(rank, trial, param_names, pareto=False):
    """Print a single row of the results table."""
    marker = "*" if pareto else " "
    wer = trial.values[0]
    decode_time = trial.values[1]
    alpha = trial.user_attrs.get("best_alpha", "N/A")
    alpha_str = f"{alpha:<8.3f}" if isinstance(alpha, float) else f"{alpha:<8}"
    params_str = " ".join(
        f"{str(trial.params.get(p, 'N/A')):<20}" for p in param_names
    )
    config_label = ""
    if "initial_config_name" in trial.user_attrs:
        config_label = f" [{trial.user_attrs['initial_config_name']}]"
    print(f"{rank:<4}{marker} {trial.number:<7} {wer:<10.5f} {decode_time:<10.4f} "
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
            directions=["minimize", "minimize"],  # WER, decode_time
            load_if_exists=False,
        )

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
        print(f"\nGrid search: {n_trials} total combinations")

    print(f"\nStarting LM decoder sweep: {sweep_cfg.study_name}")
    initial_info = (f" | Initial configs: {n_initial_configs}"
                    if n_initial_configs > 0 and not is_grid else "")
    print(f"Strategy: {sweep_cfg.search_strategy} | Trials: {n_trials} | "
          f"Concurrent jobs: {total_jobs}{initial_info}")
    print(f"Objectives: minimize WER, minimize decode time")
    print(f"Output: {sweep_cfg.output_dir}")
    print(f"DB: {db_path}\n")

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=total_jobs,
        catch=(Exception,),
    )

    # Print results
    print_lm_results_table(study)

    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_lm_visualizations(study, sweep_cfg.output_dir)

    # Summary
    pareto = study.best_trials
    print(f"\nSweep complete!")
    print(f"  SQLite DB: {db_path}")
    print(f"  Output dir: {sweep_cfg.output_dir}")
    print(f"  Pareto-optimal trials: {len(pareto)}")
    if pareto:
        # Show the Pareto trial with lowest WER
        best_wer_trial = min(pareto, key=lambda t: t.values[0])
        print(f"  Lowest WER on Pareto front: Trial {best_wer_trial.number} "
              f"(WER={best_wer_trial.values[0]:.5f}, "
              f"Time={best_wer_trial.values[1]:.4f}s)")
        # Show the Pareto trial with fastest decode time
        best_time_trial = min(pareto, key=lambda t: t.values[1])
        print(f"  Fastest on Pareto front:    Trial {best_time_trial.number} "
              f"(WER={best_time_trial.values[0]:.5f}, "
              f"Time={best_time_trial.values[1]:.4f}s)")
    print(f"\nTo explore interactively:")
    print(f"  pip install optuna-dashboard")
    print(f"  optuna-dashboard sqlite:///{db_path}")


if __name__ == "__main__":
    main()
