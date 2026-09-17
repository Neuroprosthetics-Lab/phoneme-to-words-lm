"""Validation and reproducibility shared by the sweep launcher and worker."""
import hashlib
import importlib.metadata
import inspect
import json
import math
import platform
from collections.abc import Mapping
from pathlib import Path

from phoneme_to_words_lm.utils import HF_CACHE_DIR, TEXT_NORMALIZATION_VERSION


def positive_integer(name, value, *, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise ValueError(f'{name} must be a {"nonnegative" if allow_zero else "positive"} integer')
    return value


def alpha_values(low, high, step):
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
           for v in (low, high, step)) or low < 0 or high < low or step <= 0:
        raise ValueError('alpha bounds must be finite, nonnegative and ordered, with a positive step')
    count = math.floor((high-low)/step + 1e-10) + 1
    if count > 1_000_000:
        raise ValueError('alpha grid has more than one million points')
    return [round(low+i*step, 10) for i in range(count)]


def decoder_defaults():
    from phoneme_to_words_lm.decoder import KenLMFlashlightTextLM
    return {name: p.default for name, p in inspect.signature(KenLMFlashlightTextLM).parameters.items()
            if p.default is not inspect.Parameter.empty}


RESOURCE_FIELDS = ('lexicon_path', 'tokens_path', 'kenlm_model_path')
PATH_FIELDS = RESOURCE_FIELDS + ('llm_lora_path', 'hotwords_path', 'llm_cache_dir')


def resolve_decoder_config(cfg, params=None):
    """Defaults < fixed top-level config < sampled parameters; no model load."""
    from phoneme_to_words_lm.decoder import KenLMFlashlightTextLM
    defaults = decoder_defaults()
    allowed = set(defaults) | set(RESOURCE_FIELDS)
    if 'decoder_config' in cfg:
        supplied = dict(cfg['decoder_config'])
        unknown = set(supplied)-allowed
        if unknown:
            raise ValueError(f'Unknown decoder settings: {sorted(unknown)}')
    else:
        supplied = {key: cfg[key] for key in allowed if key in cfg}
    params = dict(cfg.get('trial_params', {})) if params is None else dict(params)
    unknown = set(params)-allowed
    if unknown:
        raise ValueError(f'Unknown decoder parameters: {sorted(unknown)}')
    fixed_params = {k: spec['value'] for k, spec in cfg.get('parameters', {}).items()
                    if spec.get('type') == 'fixed'}
    result = {**defaults, **supplied, **fixed_params, **params}
    result['llm_model_name'] = cfg.get('resolved_model_paths', {}).get(
        result['llm_model_name'], result['llm_model_name'])
    result['llm_alpha'] = 0.0  # Alpha is selected from immutable scores post hoc.
    result['llm_cache_dir'] = result['llm_cache_dir'] or HF_CACHE_DIR
    for key in PATH_FIELDS:
        value = result.get(key)
        if key in RESOURCE_FIELDS and not value:
            raise ValueError(f'Missing required {key}')
        if value is not None:
            if not isinstance(value, str) or not value:
                raise ValueError(f'{key} must be a nonempty path string or null')
            result[key] = str(Path(value).expanduser().resolve())
            if key != 'llm_cache_dir' and not Path(result[key]).exists():
                raise FileNotFoundError(f'{key} not found: {result[key]}')
            if key in RESOURCE_FIELDS + ('hotwords_path',) and not Path(result[key]).is_file():
                raise ValueError(f'{key} must be a file')
            if key == 'llm_lora_path' and not Path(result[key]).is_dir():
                raise ValueError('llm_lora_path must be a directory')
    model = result['llm_model_name']
    if not isinstance(model, str) or not model:
        raise ValueError('llm_model_name must be a local snapshot or Hugging Face model ID')
    if Path(model).expanduser().exists() or model.startswith(('~', '/', './', '../')):
        result['llm_model_name'] = str(Path(model).expanduser().resolve())
        if not Path(result['llm_model_name']).is_dir():
            raise FileNotFoundError(f'LLM snapshot not found: {model}')
    for key in ('do_llm_rescoring', 'log_add', 'llm_score_eos'):
        if not isinstance(result[key], bool):
            raise ValueError(f'{key} must be boolean')
    if not isinstance(result['llm_device'], str):
        raise ValueError('llm_device must be a device string')
    for key in ('sil_token', 'blank_token', 'unk_token'):
        if not isinstance(result[key], str) or not result[key]:
            raise ValueError(f'{key} must be a nonempty token string')
    # Reuse the decoder's numeric/policy validation without native allocation.
    probe = KenLMFlashlightTextLM.__new__(KenLMFlashlightTextLM)
    probe.__dict__.update(result)
    probe._validate_settings()
    hotwords = result['hotwords']
    if hotwords is not None:
        if not isinstance(hotwords, Mapping):
            raise ValueError('hotwords must be a mapping')
        result['hotwords'] = dict(hotwords)
        normalized = set()
        for word, bonus in hotwords.items():
            if not isinstance(word, str) or not word.strip() or word.strip().casefold() in normalized:
                raise ValueError('hotword names must be nonempty and unique after normalization')
            normalized.add(word.strip().casefold())
            if isinstance(bonus, bool) or not isinstance(bonus, (int, float)) or not math.isfinite(bonus):
                raise ValueError('hotword bonuses must be finite numbers')
    return result


def validate_parameter(name, spec):
    kind = spec.get('type')
    if kind == 'fixed':
        return [spec['value']]
    if kind == 'boolean':
        return [False, True]
    if kind == 'choice':
        values = spec.get('values')
        if not isinstance(values, list) or not values:
            raise ValueError(f'{name}: choice requires a nonempty values list')
        if any(isinstance(v, (dict, list)) for v in values):
            raise ValueError(f'{name}: composite settings must use type: fixed')
        return values
    if kind not in ('float_range', 'int_range'):
        raise ValueError(f'{name}: unknown parameter type {kind!r}')
    low, high = spec['low'], spec['high']
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in (low, high)) or high < low:
        raise ValueError(f'{name}: invalid bounds')
    if kind == 'int_range' and any(not isinstance(v, int) for v in (low, high)):
        raise ValueError(f'{name}: integer bounds required')
    step = spec.get('step')
    if step is not None and (isinstance(step, bool) or not isinstance(step, (float, int))
                             or not math.isfinite(step) or step <= 0):
        raise ValueError(f'{name}: step must be finite and positive')
    if kind == 'int_range' and step is not None and not isinstance(step, int):
        raise ValueError(f'{name}: integer step required')
    if spec.get('log', False) and (low <= 0 or step is not None):
        raise ValueError(f'{name}: log sampling requires positive bounds and no step')
    if 'n_grid_points' in spec:
        positive_integer(f'{name}.n_grid_points', spec['n_grid_points'])
        if spec['n_grid_points'] < 2 and high != low:
            raise ValueError(f'{name}: at least two grid points required for a range')
    return [low, high]


def file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    before = path.stat()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f'Resource changed while fingerprinting: {path}')
    return dict(path=str(path), bytes=after.st_size, sha256=digest.hexdigest())


def runtime_identity():
    versions = {'python': platform.python_version(), 'text_normalization': TEXT_NORMALIZATION_VERSION}
    for name in ('torch', 'transformers', 'tokenizers', 'flashlight-text', 'optuna',
                 'numpy', 'editdistance', 'omegaconf', 'huggingface-hub', 'peft'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None  # Optional PEFT absence is part of the identity.
    return versions


def run_identity(cfg, device_specs):
    """Hash actual input/model/native/code content once before starting workers."""
    from huggingface_hub import snapshot_download
    import flashlight.lib.text.flashlight_lib_text_decoder as native
    import torch
    fixed = {k: v for k, v in cfg.items()
             if k not in ('n_trials', 'output_dir', 'initial_configs', 'study_name')}
    resources, snapshots, model_snapshots = set(), set(), {}
    resources.add(cfg['logits_pkl_path'])
    configs = [resolve_decoder_config(cfg)]
    for key, spec in cfg['parameters'].items():
        configs.extend(resolve_decoder_config(cfg, {key: value})
                       for value in validate_parameter(key, spec))
    needs_llm = any(config['do_llm_rescoring'] for config in configs)
    for config in configs:
        resources.update(config[k] for k in RESOURCE_FIELDS)
        if config['hotwords_path']:
            resources.add(config['hotwords_path'])
        if needs_llm:
            model = config['llm_model_name']
            if not Path(model).is_dir():
                try:
                    model = snapshot_download(model, cache_dir=config['llm_cache_dir'], local_files_only=True)
                except Exception as exc:
                    raise ValueError(f'Cache or provide a local immutable LLM snapshot before sweeping: {model}') from exc
            snapshots.add(model)
            model_snapshots[config['llm_model_name']] = str(Path(model).resolve())
            if config['llm_lora_path']:
                snapshots.add(config['llm_lora_path'])
    for directory in snapshots:
        resources.update(str(p) for p in Path(directory).rglob('*') if p.is_file())
    package = Path(__file__).parent
    resources.update(str(p) for p in package.glob('*.py'))
    resources.update(str(p) for p in (package.parent/'sweep').glob('*.py'))
    resources.update(str(p) for p in Path(native.__file__).parent.glob('*.so'))
    identity = dict(version='sweep_v2', config=fixed, devices=device_specs,
                    runtime=runtime_identity(),
                    cuda=torch.version.cuda,
                    files=[file_identity(p) for p in sorted(resources)])
    canonical = json.dumps(identity, sort_keys=True).encode()
    return dict(sha256=hashlib.sha256(canonical).hexdigest(), manifest=identity,
                model_snapshots=model_snapshots)


def verify_resume(study, identity, *, resume):
    previous = study.user_attrs.get('run_identity_sha256')
    if resume and previous != identity['sha256']:
        raise ValueError('Cannot resume: dataset, resources, code, runtime or effective sweep configuration changed (or study lacks a v2 identity). Use a new study/output directory.')
    if not resume:
        study.set_user_attr('run_identity_sha256', identity['sha256'])
