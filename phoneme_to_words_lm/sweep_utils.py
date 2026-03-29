"""Shared Optuna sweep utilities.

Extracted from the npl-davis encoder hyperparameter sweep for reuse by
the LM decoder sweep (lm_sweep.py).
"""

import json
import queue

import optuna
from omegaconf import OmegaConf


def _build_grid_search_space(parameters):
    """Convert parameter specs into a dict of discrete lists for GridSampler."""
    import numpy as np
    search_space = {}
    for name, spec in parameters.items():
        ptype = spec["type"]
        if ptype == "choice":
            search_space[name] = [json.dumps(list(v)) if OmegaConf.is_list(v) else v for v in spec["values"]]
        elif ptype == "fixed":
            search_space[name] = [spec["value"]]
        elif ptype == "float_range":
            n_pts = spec.get("n_grid_points", 5)
            low, high = spec["low"], spec["high"]
            if spec.get("log", False):
                search_space[name] = list(np.geomspace(low, high, n_pts))
            else:
                search_space[name] = list(np.linspace(low, high, n_pts))
        elif ptype == "int_range":
            n_pts = spec.get("n_grid_points", None)
            low, high = spec["low"], spec["high"]
            if n_pts is not None:
                import math
                step = max(1, math.ceil((high - low) / (n_pts - 1)))
                search_space[name] = list(range(low, high + 1, step))
            else:
                search_space[name] = list(range(low, high + 1))
        elif ptype == "boolean":
            search_space[name] = [False, True]
        else:
            raise ValueError(f"Unknown param type '{ptype}' for grid search param '{name}'")
    return search_space


def encode_param_value(name: str, spec, value):
    """Encode a user-provided value into Optuna's internal representation.

    Returns (encoded_value, skip) where skip=True means the param should be
    omitted from the enqueued dict (e.g. fixed params handled outside Optuna).
    Raises ValueError if the value is invalid for the param spec.
    """
    ptype = spec["type"]

    if ptype == "fixed":
        # Fixed params are applied directly in build_trial_config, not via Optuna
        return None, True

    elif ptype == "choice":
        valid_choices = []
        for v in spec["values"]:
            if OmegaConf.is_list(v):
                valid_choices.append(json.dumps(list(v)))
            else:
                valid_choices.append(v)
        # If the user value is a list, JSON-encode it to match suggest_categorical
        if isinstance(value, (list, tuple)) or OmegaConf.is_list(value):
            encoded = json.dumps(list(value))
        else:
            encoded = value
        if encoded not in valid_choices:
            raise ValueError(
                f"initial_configs: value {value!r} for '{name}' is not in "
                f"choices {spec['values']}"
            )
        return encoded, False

    elif ptype == "float_range":
        fval = float(value)
        if fval < spec["low"] or fval > spec["high"]:
            raise ValueError(
                f"initial_configs: value {fval} for '{name}' is outside "
                f"range [{spec['low']}, {spec['high']}]"
            )
        return fval, False

    elif ptype == "int_range":
        ival = int(value)
        if ival < spec["low"] or ival > spec["high"]:
            raise ValueError(
                f"initial_configs: value {ival} for '{name}' is outside "
                f"range [{spec['low']}, {spec['high']}]"
            )
        return ival, False

    elif ptype == "boolean":
        return bool(value), False

    else:
        raise ValueError(f"Unknown param type '{ptype}' for param '{name}'")


def enqueue_initial_configs(study, initial_configs, parameters):
    """Enqueue user-specified initial configs into the Optuna study.

    Each config's params are encoded via encode_param_value and enqueued with
    skip_if_exists=True so that --resume doesn't create duplicates.
    """
    if not initial_configs:
        return 0

    count = 0
    for i, cfg_entry in enumerate(initial_configs):
        cfg_name = cfg_entry.get("name", f"initial_config_{i}")
        params_dict = cfg_entry.get("params", {})

        encoded = {}
        for pname, pvalue in params_dict.items():
            if pname not in parameters:
                print(f"  Warning: '{pname}' in initial config '{cfg_name}' "
                      f"is not in sweep parameters — skipping")
                continue
            spec = parameters[pname]
            enc_val, skip = encode_param_value(pname, spec, pvalue)
            if not skip:
                encoded[pname] = enc_val

        study.enqueue_trial(
            params=encoded,
            user_attrs={"initial_config_name": cfg_name},
            skip_if_exists=True,
        )
        count += 1
        print(f"  Enqueued initial config '{cfg_name}': {encoded}")

    return count


def suggest_param(trial, name: str, spec):
    ptype = spec["type"]
    if ptype == "fixed":
        return spec["value"]
    elif ptype == "choice":
        choices = [json.dumps(list(v)) if OmegaConf.is_list(v) else v for v in spec["values"]]
        result = trial.suggest_categorical(name, choices)
        # Decode JSON strings back to lists
        if isinstance(result, str):
            try:
                decoded = json.loads(result)
                if isinstance(decoded, list):
                    return decoded
            except (json.JSONDecodeError, TypeError):
                pass
        return result
    elif ptype == "float_range":
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    elif ptype == "int_range":
        return trial.suggest_int(name, spec["low"], spec["high"])
    elif ptype == "boolean":
        return trial.suggest_categorical(name, [False, True])
    else:
        raise ValueError(f"Unknown param type: {ptype}")


class GPUPool:
    """Thread-safe pool of GPU device indices with per-GPU concurrency."""

    def __init__(self, device_specs: list[tuple[int, int]]):
        self._queue = queue.Queue()
        for gpu_id, n_jobs in device_specs:
            for _ in range(n_jobs):
                self._queue.put(gpu_id)

    def acquire(self) -> int:
        return self._queue.get()  # blocks until available

    def release(self, gpu_id: int):
        self._queue.put(gpu_id)
