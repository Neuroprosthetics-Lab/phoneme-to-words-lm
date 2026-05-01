"""Train KenLM n-gram language models for brain-to-text BCI decoding.

Supports word-level and character-level (spelling) LMs, multiple corpora
with weighted SRILM interpolation, and per-corpus n-gram orders.

Usage:
    python train_ngram_lm.py --config config.yaml [--verbose]

See README.md for full documentation and example_config.yaml for config reference.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import re
import shutil
import subprocess
import sys
import time

from omegaconf import OmegaConf
from tqdm import tqdm

from phoneme_to_words_lm.utils import LOGIT_PHONE_DEF
from text_normalize import collect_vocab_from_normalized, normalize_corpus

# Default to the cmu_dict.pkl bundled with the phoneme_to_words_lm package
# when the config doesn't specify cmu_dict_path.
import phoneme_to_words_lm
_BUNDLED_CMU_DICT_PATH = os.path.join(
    os.path.dirname(phoneme_to_words_lm.__file__), 'cmu_dict.pkl'
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Spoken letter-name phoneme sequences (from CMU dict).
# SIL is NOT included here -- it is appended during lexicon generation.
SPOKEN_LETTER_PHONEMES = {
    'a': ['EY'],
    'b': ['B', 'IY'],
    'c': ['S', 'IY'],
    'd': ['D', 'IY'],
    'e': ['IY'],
    'f': ['EH', 'F'],
    'g': ['JH', 'IY'],
    'h': ['EY', 'CH'],
    'i': ['AY'],
    'j': ['JH', 'EY'],
    'k': ['K', 'EY'],
    'l': ['EH', 'L'],
    'm': ['EH', 'M'],
    'n': ['EH', 'N'],
    'o': ['OW'],
    'p': ['P', 'IY'],
    'q': ['K', 'Y', 'UW'],
    'r': ['AA', 'R'],
    's': ['EH', 'S'],
    't': ['T', 'IY'],
    'u': ['Y', 'UW'],
    'v': ['V', 'IY'],
    'w': ['D', 'AH', 'B', 'AH', 'L', 'Y', 'UW'],
    'x': ['EH', 'K', 'S'],
    'y': ['W', 'AY'],
    'z': ['Z', 'IY'],
}


# ---------------------------------------------------------------------------
# External tool wrappers
# ---------------------------------------------------------------------------

def _run_with_live_stderr(
    cmd: list[str],
    tool_name: str,
    stdin=None,
    stdout=subprocess.DEVNULL,
) -> tuple[int, str]:
    """Run a subprocess, forwarding its stderr to the logger.

    The external tools lmplz / SRILM ngram / build_binary write their own
    progress output to stderr. Streaming it live (instead of capturing and
    emitting after exit) gives the user real-time visibility into
    long-running steps. Captured stderr is still returned so callers can
    attach it to a CalledProcessError on failure.

    lmplz in particular draws an ASCII progress bar by emitting '*'
    characters one at a time with no trailing newline until the bar is
    complete. A line-buffered reader (``for line in pipe``) blocks on
    readline() and hides the whole bar until it finishes -- defeating
    its purpose. So we read stderr byte-by-byte and, when sys.stderr is
    a TTY, stream each printable byte directly for real-time animation.
    On '\\n' we clear the partial display (so it isn't duplicated) and
    route the complete line through the logger so it gets the normal
    timestamp formatting and lands in any file handlers.
    """
    process = subprocess.Popen(
        cmd,
        stdin=stdin,
        stdout=stdout,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    assert process.stderr is not None
    fd = process.stderr.fileno()

    prefix = f'    [{tool_name}] '
    is_tty = sys.stderr.isatty()
    captured: list[str] = []
    line_buf = bytearray()
    partial_written = 0  # chars streamed on the current partial line

    def _clear_partial() -> None:
        nonlocal partial_written
        if partial_written:
            cols = shutil.get_terminal_size((80, 24)).columns
            n_lines = 1 + (partial_written - 1) // cols
            sys.stderr.write('\r')
            if n_lines > 1:
                sys.stderr.write(f'\033[{n_lines - 1}A')
            sys.stderr.write('\033[J')  # Erase from cursor to end of screen.
            sys.stderr.flush()
            partial_written = 0

    def _finalize() -> None:
        nonlocal line_buf
        text = line_buf.decode('utf-8', errors='replace')
        _clear_partial()
        if text:
            log.info(f'{prefix}{text}')
            captured.append(text)
        line_buf = bytearray()

    try:
        while True:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            for b in chunk:
                if b == 0x0A:  # \n
                    _finalize()
                elif b == 0x0D:  # \r (tools that use CR for in-place updates)
                    if line_buf:
                        _finalize()
                else:
                    line_buf.append(b)
                    if is_tty and 0x20 <= b < 0x80:
                        if partial_written == 0:
                            sys.stderr.write(prefix)
                            partial_written = len(prefix)
                        sys.stderr.write(chr(b))
                        sys.stderr.flush()
                        partial_written += 1
    finally:
        if line_buf:
            _finalize()

    returncode = process.wait()
    return returncode, '\n'.join(captured)


def train_single_lm(
    text_path: str,
    output_path: str,
    order: int,
    pruning: list[int] | None,
    memory: str,
    lmplz_path: str,
    intermediate: bool = False,
    discount_fallback: bool = False,
) -> None:
    """Train a single n-gram LM from a normalized text file using lmplz.

    When ``intermediate`` is False (default), writes an ARPA to
    ``output_path``. When True, writes KenLM intermediate-format files
    using ``output_path`` as a prefix -- lmplz will produce
    ``<prefix>.1``, ..., ``<prefix>.{order}``, plus ``<prefix>.vocab``
    and ``<prefix>.kenlm_intermediate``. Intermediate mode suppresses
    ARPA output (lmplz's ``--intermediate`` turns off ``--arpa`` by
    default); downstream interpolation consumes the prefix directly.

    ``discount_fallback`` passes ``--discount_fallback`` to lmplz, which
    substitutes default Kneser-Ney discounts when the data is too dense
    to estimate them (e.g. char-level spelling LMs with only ~30 unigram
    types have no adjusted-count-1 unigrams, so the KN discount fit fails).
    """
    cmd = [lmplz_path, '-o', str(order), '-S', memory]
    if pruning:
        cmd.extend(['--prune'] + [str(p) for p in pruning])
    if intermediate:
        cmd += ['--intermediate', output_path]
    if discount_fallback:
        cmd.append('--discount_fallback')

    kind = 'intermediate' if intermediate else 'ARPA'
    log.info(f'  Training {order}-gram ({kind}): {" ".join(cmd)}')
    log.info(f'    input:  {text_path}')
    log.info(f'    output: {output_path}{".*" if intermediate else ""}')

    with open(text_path, 'r') as fin:
        if intermediate:
            returncode, stderr = _run_with_live_stderr(
                cmd, 'lmplz', stdin=fin, stdout=subprocess.DEVNULL,
            )
        else:
            with open(output_path, 'w') as fout:
                returncode, stderr = _run_with_live_stderr(
                    cmd, 'lmplz', stdin=fin, stdout=fout,
                )

    if returncode != 0:
        log.error(f'lmplz failed (exit {returncode})')
        raise subprocess.CalledProcessError(returncode, cmd, stderr=stderr)

    if intermediate:
        total = sum(
            os.path.getsize(f'{output_path}.{i}') for i in range(1, order + 1)
        )
        log.info(f'    Intermediate files written: {total / 1e6:.1f} MB total')
    else:
        arpa_size = os.path.getsize(output_path)
        log.info(f'    ARPA written: {arpa_size / 1e6:.1f} MB')


def interpolate_lms_srilm(
    arpa_paths: list[str],
    weights: list[float],
    output_arpa: str,
    max_order: int,
    srilm_path: str,
) -> None:
    """Interpolate 2+ ARPA models using SRILM's ngram tool.

    Weights are already normalized to sum to 1.0.

    SRILM semantics:
    - ``-lambda`` is the weight for ``-lm`` (the primary model).
    - The remainder ``(1 - lambda)`` goes to ``-mix-lm``.
    - For 3+ models, ``-mix-lm2``/``-mix-lambda2``, etc. specify additional
      models with absolute weights taken from the ``(1 - lambda)`` pool.
    """
    assert len(arpa_paths) >= 2
    assert len(arpa_paths) == len(weights)

    cmd = [srilm_path, '-order', str(max_order)]
    cmd += ['-lm', arpa_paths[0]]
    cmd += ['-mix-lm', arpa_paths[1]]
    cmd += ['-lambda', str(weights[0])]
    for i in range(2, len(arpa_paths)):
        cmd += [f'-mix-lm{i}', arpa_paths[i]]
        cmd += [f'-mix-lambda{i}', str(weights[i])]
    cmd += ['-write-lm', output_arpa]

    log.info(f'  Interpolating {len(arpa_paths)} models: {" ".join(cmd)}')

    returncode, stderr = _run_with_live_stderr(cmd, 'ngram')

    if returncode != 0:
        log.error(f'SRILM ngram failed (exit {returncode})')
        raise subprocess.CalledProcessError(returncode, cmd, stderr=stderr)

    arpa_size = os.path.getsize(output_arpa)
    log.info(f'    Interpolated ARPA written: {arpa_size / 1e6:.1f} MB')


def interpolate_lms_kenlm(
    intermediate_paths: list[str],
    weights: list[float],
    output_arpa: str,
    kenlm_interpolate_path: str,
    memory: str,
    temp_prefix: str,
) -> None:
    """Interpolate 2+ LMs using KenLM's streaming `interpolate` tool.

    Inputs are KenLM intermediate-format file-prefixes (produced by
    ``lmplz --intermediate <prefix>``; each prefix fans out into
    ``<prefix>.N`` per order, plus ``.vocab`` and ``.kenlm_intermediate``
    sidecars). Unlike SRILM, KenLM interpolate sort-merges on disk and
    keeps peak RAM proportional to the sort buffer, so it scales to
    model sets that don't fit in memory simultaneously.

    Writes the interpolated ARPA to ``output_arpa`` (stdout of the tool).
    Weights are passed as absolute mixture coefficients and must already
    sum to 1.0.
    """
    assert len(intermediate_paths) >= 2
    assert len(intermediate_paths) == len(weights)

    os.makedirs(temp_prefix, exist_ok=True)

    cmd = [kenlm_interpolate_path,
           '-m', *intermediate_paths,
           '-w', *[str(w) for w in weights],
           '-S', memory,
           '-T', temp_prefix]

    log.info(f'  Interpolating {len(intermediate_paths)} models: {" ".join(cmd)}')

    with open(output_arpa, 'w') as fout:
        returncode, stderr = _run_with_live_stderr(
            cmd, 'kenlm-interpolate', stdout=fout,
        )

    if returncode != 0:
        log.error(f'KenLM interpolate failed (exit {returncode})')
        raise subprocess.CalledProcessError(returncode, cmd, stderr=stderr)

    arpa_size = os.path.getsize(output_arpa)
    log.info(f'    Interpolated ARPA written: {arpa_size / 1e6:.1f} MB')


def compile_to_binary(arpa_path: str, bin_path: str, build_binary_path: str) -> None:
    """Compile an ARPA file to KenLM binary (trie format)."""
    cmd = [build_binary_path, 'trie', arpa_path, bin_path]
    log.info(f'  Compiling: {" ".join(cmd)}')

    returncode, stderr = _run_with_live_stderr(cmd, 'build_binary')

    if returncode != 0:
        log.error(f'build_binary failed (exit {returncode})')
        raise subprocess.CalledProcessError(returncode, cmd, stderr=stderr)

    bin_size = os.path.getsize(bin_path)
    log.info(f'    Binary written: {bin_size / 1e6:.1f} MB')


# ---------------------------------------------------------------------------
# Lexicon and tokens generation
# ---------------------------------------------------------------------------

def load_cmu_dict(path: str) -> dict[str, list[list[str]]]:
    """Load the CMU pronunciation dictionary from a .pkl or text file.

    Both formats yield the same structure: ``{lowercase_word: [[phones], ...]}``
    with stress markers (trailing digits) stripped from phones.

    Text format follows the upstream cmudict-0.7b convention:
    ``;;;`` comment lines, one ``WORD  P1 P2 ...`` entry per line, and
    ``WORD(N)`` naming the N-th alternate pronunciation.
    """
    if path.endswith('.pkl'):
        with open(path, 'rb') as f:
            return pickle.load(f)

    cmu_dict: dict[str, list[list[str]]] = {}
    alt_pron_re = re.compile(r'\(\d+\)$')
    with open(path, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';;;'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            word = alt_pron_re.sub('', parts[0]).lower()
            phones = [re.sub(r'[0-9]', '', p) for p in parts[1:]]
            prons = cmu_dict.setdefault(word, [])
            # Dedupe: alternates that only differed in stress collapse to the
            # same phone list after stripping. Matches the pkl's semantics and
            # avoids duplicate lines in the generated lexicon.
            if phones not in prons:
                prons.append(phones)
    return cmu_dict


def _g2p_phonemize(word: str) -> list[str] | None:
    """Use g2p-en to phonemize a word, returning stress-stripped ARPABET phones.

    Returns None if g2p-en produces no valid phonemes.
    """
    from g2p_en import G2p
    if not hasattr(_g2p_phonemize, '_g2p'):
        _g2p_phonemize._g2p = G2p()

    raw = _g2p_phonemize._g2p(word)
    phonemes = []
    for p in raw:
        p = re.sub(r'[0-9]', '', p)  # Remove stress markers
        if re.match(r'^[A-Z]+$', p):  # Only keep valid phonemes
            phonemes.append(p)
    return phonemes if phonemes else None


def generate_lexicon(
    lm_type: str,
    vocab: set[str],
    cmu_dict: dict | None,
    output_path: str,
    english_words: set[str] | None = None,
) -> None:
    """Generate lexicon.txt mapping words to phoneme sequences.

    Word mode: looks up each vocab word in the CMU pronunciation dictionary.
    Words with multiple pronunciations get multiple lines. If
    ``english_words`` is provided, OOV words are phonemized with g2p-en
    when they appear in that set -- others are dropped from the lexicon
    entirely. This keeps misspellings, concatenated tokens, and garbage
    strings out of the decoder vocabulary. If ``english_words`` is None,
    every OOV word is dropped (no g2p-en fallback), so the decoder
    vocabulary is exactly the CMU-dict intersection of the corpus vocab.
    OOV words and their g2p-en phone sequences are saved to oov_g2p.txt
    in the same directory as the lexicon.

    Spelling mode: 26 entries (a-z) with hardcoded spoken-letter phonemes.
    """
    lines = []

    if lm_type == 'spelling':
        for letter in sorted(SPOKEN_LETTER_PHONEMES):
            phonemes = ' '.join(SPOKEN_LETTER_PHONEMES[letter]) + ' SIL'
            lines.append(f'{letter}\t{phonemes}\n')
        log.info(f'  Lexicon: {len(lines)} spelling entries written')

    else:
        assert cmu_dict is not None
        n_found = 0
        n_g2p = 0
        # Not in CMU dict and either english_words is None or word not in it.
        n_filtered = 0
        n_skipped = 0   # Passed filter but g2p produced no valid phonemes.
        oov_lines = []
        rejected_lines = []
        sorted_vocab = sorted(vocab)
        pbar = tqdm(
            sorted_vocab,
            desc='  Building lexicon',
            unit='word',
            disable=not sys.stderr.isatty(),
        )
        for word in pbar:
            if word in cmu_dict:
                for pronunciation in cmu_dict[word]:
                    phonemes = ' '.join(pronunciation) + ' SIL'
                    lines.append(f'{word}\t{phonemes}\n')
                n_found += 1
            elif english_words is None or word not in english_words:
                reason = 'no_english_words_list' if english_words is None else 'not_in_english_words'
                rejected_lines.append(f'{word}\t{reason}\n')
                n_filtered += 1
            else:
                g2p_phones = _g2p_phonemize(word)
                if g2p_phones is not None:
                    phonemes = ' '.join(g2p_phones) + ' SIL'
                    lines.append(f'{word}\t{phonemes}\n')
                    oov_lines.append(f'{word}\t{" ".join(g2p_phones)}\n')
                    n_g2p += 1
                else:
                    rejected_lines.append(f'{word}\tg2p_failed\n')
                    n_skipped += 1
            pbar.set_postfix(cmu=n_found, g2p=n_g2p, filt=n_filtered, skip=n_skipped)
        pbar.close()

        # Write OOV words and their g2p-en pronunciations.
        out_dir = os.path.dirname(output_path)
        oov_path = os.path.join(out_dir, 'oov_g2p.txt')
        with open(oov_path, 'w') as f:
            f.writelines(oov_lines)

        # Write rejected words (filtered or g2p-failed) for debugging.
        rejected_path = os.path.join(out_dir, 'rejected_oov.txt')
        with open(rejected_path, 'w') as f:
            f.writelines(rejected_lines)

        total = n_found + n_g2p + n_filtered + n_skipped
        filter_reason = ('no english_words_path set' if english_words is None
                         else 'not in english word list')
        log.info(f'  Lexicon: {n_found} from CMU, {n_g2p} from g2p-en, '
                 f'{n_filtered} filtered ({filter_reason}), '
                 f'{n_skipped} skipped (g2p failed) '
                 f'({total} total vocab)')
        if n_g2p > 0:
            log.info(f'  OOV pronunciations written to {oov_path}')
        if rejected_lines:
            log.info(f'  Rejected words written to {rejected_path}')

    with open(output_path, 'w') as f:
        f.writelines(lines)


def generate_tokens(output_path: str) -> None:
    """Write the static tokens.txt file (BLANK, SIL, AA..ZH)."""
    with open(output_path, 'w') as f:
        for token in LOGIT_PHONE_DEF:
            f.write(token + '\n')
    log.info(f'  Tokens: {len(LOGIT_PHONE_DEF)} tokens written')


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_CONFIG_DEFAULTS = {
    'memory': '80%',
    'lmplz_path': 'lmplz',
    'build_binary_path': 'build_binary',
    'srilm_ngram_path': 'ngram',
    'kenlm_interpolate_path': 'interpolate',
    'cmu_dict_path': None,
    'english_words_path': None,
    # None -> auto-detect at validation time; see load_and_validate_config.
    'normalize_workers': None,
}

_INTERPOLATOR_BACKENDS = ('srilm', 'kenlm')


_SAFE_NAME_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')
_UNSAFE_NAME_CHAR_RE = re.compile(r'[^A-Za-z0-9_.-]+')


def _as_path_list(x) -> list[str] | None:
    """Coerce a single path or list-of-paths into a list. None passes through.

    Accepts plain strings, Python lists, and OmegaConf ListConfig (which is
    iterable but not a ``list`` instance).
    """
    if x is None:
        return None
    if isinstance(x, str):
        return [x]
    return list(x)


def _derive_corpus_name(c) -> str:
    """Derive a default corpus name from its source file's basename (sans extension).

    Priority matches the source priority:
    arpa_path > intermediate_path > normalized_path > path.
    Any characters outside ``[A-Za-z0-9_.-]`` are collapsed to ``_``.
    Only callable when the chosen source is a single string -- list-form
    sources require an explicit ``name`` (enforced in ``_resolve_corpus_names``).
    """
    src = (OmegaConf.select(c, 'arpa_path')
           or OmegaConf.select(c, 'intermediate_path')
           or OmegaConf.select(c, 'normalized_path')
           or OmegaConf.select(c, 'path'))
    stem = os.path.splitext(os.path.basename(src))[0]
    sanitized = _UNSAFE_NAME_CHAR_RE.sub('_', stem).strip('_')
    return sanitized or 'corpus'


def _resolve_corpus_names(corpora) -> None:
    """Assign a unique, filesystem-safe ``name`` to each corpus in-place.

    If a corpus already has ``name``, it is validated. Otherwise a name is
    derived from the source filename. Collisions are resolved by appending
    ``_1``, ``_2``, etc. to later corpora. List-form ``path`` /
    ``normalized_path`` require an explicit ``name`` since deriving from
    one of N filenames would be arbitrary.
    """
    seen: set[str] = set()
    for i, c in enumerate(corpora):
        explicit = OmegaConf.select(c, 'name')
        if explicit is not None:
            if not isinstance(explicit, str) or not _SAFE_NAME_RE.match(explicit):
                raise ValueError(
                    f'Corpus [{i}] name {explicit!r} is invalid. '
                    f'Must match {_SAFE_NAME_RE.pattern} (letters, digits, _, -, .)'
                )
            name = explicit
            if name in seen:
                raise ValueError(f'Corpus [{i}] name {name!r} is duplicated')
        else:
            for field in ('path', 'normalized_path'):
                val = OmegaConf.select(c, field)
                if val is not None and not isinstance(val, str):
                    raise ValueError(
                        f'Corpus [{i}] uses a multi-file {field}; an explicit '
                        f'`name` field is required when {field} is a list.'
                    )
            base = _derive_corpus_name(c)
            name = base
            suffix = 1
            while name in seen:
                name = f'{base}_{suffix}'
                suffix += 1
        seen.add(name)
        OmegaConf.update(c, 'name', name)


def load_and_validate_config(config_path: str) -> OmegaConf:
    """Load YAML config, apply defaults, and validate all fields."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file not found: {config_path}')

    cfg = OmegaConf.load(config_path)

    # Apply defaults for optional fields.
    for key, default in _CONFIG_DEFAULTS.items():
        if key not in cfg or cfg[key] is None:
            OmegaConf.update(cfg, key, default)

    # Expand ~ in paths.
    for key in ['output_dir', 'cmu_dict_path', 'english_words_path',
                'lmplz_path', 'build_binary_path', 'srilm_ngram_path',
                'kenlm_interpolate_path']:
        val = OmegaConf.select(cfg, key)
        if val and isinstance(val, str):
            OmegaConf.update(cfg, key, os.path.expanduser(val))

    # --- Validate lm_type ---
    if cfg.lm_type not in ('word', 'spelling'):
        raise ValueError(f"lm_type must be 'word' or 'spelling', got '{cfg.lm_type}'")

    # --- Resolve normalize_workers (None -> auto) ---
    nw = OmegaConf.select(cfg, 'normalize_workers')
    if nw is None:
        # Cap at 16: nltk + num2words duplication per worker adds memory, and
        # past ~16 workers the pipeline becomes I/O-bound on most corpora.
        # Users with headroom can override explicitly.
        resolved = min(os.cpu_count() or 1, 16)
        OmegaConf.update(cfg, 'normalize_workers', resolved)
    elif not isinstance(nw, int) or nw < 1:
        raise ValueError(
            f'normalize_workers must be a positive int (or unset for auto), got {nw!r}'
        )

    # --- Validate interpolator_backend (required, no default) ---
    backend = OmegaConf.select(cfg, 'interpolator_backend')
    if backend is None:
        raise ValueError(
            f"Config must specify interpolator_backend (one of: "
            f"{', '.join(_INTERPOLATOR_BACKENDS)})"
        )
    if backend not in _INTERPOLATOR_BACKENDS:
        raise ValueError(
            f"interpolator_backend must be one of "
            f"{list(_INTERPOLATOR_BACKENDS)}, got '{backend}'"
        )

    # --- Validate corpora ---
    if 'corpora' not in cfg or not cfg.corpora:
        raise ValueError('Config must have a non-empty corpora list')

    for i, c in enumerate(cfg.corpora):
        # Expand ~ in corpus paths. `path` and `normalized_path` may be a single
        # string or a list of strings (group of files to concatenate);
        # `arpa_path` is always a single file.
        for key in ['path', 'normalized_path']:
            val = OmegaConf.select(c, key)
            if val is None:
                continue
            if isinstance(val, str):
                OmegaConf.update(c, key, os.path.expanduser(val))
            else:
                OmegaConf.update(c, key, [os.path.expanduser(p) for p in val])
        for key in ['arpa_path', 'intermediate_path']:
            val = OmegaConf.select(c, key)
            if val and isinstance(val, str):
                OmegaConf.update(c, key, os.path.expanduser(val))

        # At least one source must be provided.
        has_path = OmegaConf.select(c, 'path') is not None
        has_norm = OmegaConf.select(c, 'normalized_path') is not None
        has_arpa = OmegaConf.select(c, 'arpa_path') is not None
        has_inter = OmegaConf.select(c, 'intermediate_path') is not None
        if not (has_path or has_norm or has_arpa or has_inter):
            raise ValueError(
                f'Corpus [{i}] must have at least one of: '
                f'path, normalized_path, arpa_path, intermediate_path'
            )

        # Reuse fields are backend-specific.
        if has_arpa and backend == 'kenlm':
            raise ValueError(
                f"Corpus [{i}] specifies arpa_path but interpolator_backend "
                f"is 'kenlm'. KenLM interpolate cannot consume ARPA files. "
                f"Use intermediate_path instead, or omit it to retrain from "
                f"normalized_path / path."
            )
        if has_inter and backend == 'srilm':
            raise ValueError(
                f"Corpus [{i}] specifies intermediate_path but "
                f"interpolator_backend is 'srilm'. SRILM ngram cannot "
                f"consume KenLM intermediate files. Use arpa_path instead, "
                f"or omit it to retrain from normalized_path / path."
            )

        # Verify the highest-priority source file(s) exist. The highest
        # priority is backend-dependent: srilm uses arpa_path (already
        # rejected above for kenlm), kenlm uses intermediate_path.
        if has_arpa:
            if not os.path.exists(c.arpa_path):
                raise FileNotFoundError(f'Corpus [{i}] arpa_path not found: {c.arpa_path}')
        elif has_inter:
            # intermediate_path is a prefix; lmplz writes <prefix>.N,
            # <prefix>.vocab, and <prefix>.kenlm_intermediate. Validate
            # the sentinel file to catch typos without hardcoding the
            # n-gram order here.
            sentinel = f'{c.intermediate_path}.kenlm_intermediate'
            if not os.path.exists(sentinel):
                raise FileNotFoundError(
                    f'Corpus [{i}] intermediate_path prefix invalid: '
                    f'expected sentinel file {sentinel} from a prior '
                    f'lmplz --intermediate run.'
                )
        elif has_norm:
            norm_paths = _as_path_list(c.normalized_path)
            if not norm_paths:
                raise ValueError(f'Corpus [{i}] normalized_path is empty')
            for p in norm_paths:
                if not os.path.exists(p):
                    raise FileNotFoundError(f'Corpus [{i}] normalized_path not found: {p}')
        else:
            paths = _as_path_list(c.path)
            if not paths:
                raise ValueError(f'Corpus [{i}] path is empty')
            for p in paths:
                if not os.path.exists(p):
                    raise FileNotFoundError(f'Corpus [{i}] path not found: {p}')

        # Order is required.
        if 'order' not in c:
            raise ValueError(f'Corpus [{i}] must have an order field')
        if not isinstance(c.order, int) or not 1 <= c.order <= 10:
            raise ValueError(f'Corpus [{i}] order must be an int 1-10, got {c.order}')

        # Pruning validation.
        pruning = OmegaConf.select(c, 'pruning')
        if pruning is not None:
            pruning = list(pruning)
            if len(pruning) != c.order:
                raise ValueError(f'Corpus [{i}] pruning length ({len(pruning)}) must match order ({c.order})')
            if pruning[0] != 0:
                raise ValueError(f'Corpus [{i}] pruning[0] must be 0 (unigrams cannot be pruned)')

    # --- Resolve corpus names (explicit or derived from source filenames) ---
    _resolve_corpus_names(cfg.corpora)

    # --- Validate weights for multi-corpus ---
    if len(cfg.corpora) > 1:
        for i, c in enumerate(cfg.corpora):
            if 'weight' not in c:
                raise ValueError(f'Corpus [{i}] must have a weight field for multi-corpus interpolation')
            if c.weight <= 0:
                raise ValueError(f'Corpus [{i}] weight must be positive, got {c.weight}')

    # --- Validate tool paths ---
    def _check_tool(path: str, name: str, required: bool = True):
        if not required:
            return
        resolved = shutil.which(path) or (os.path.exists(path) and path)
        if not resolved:
            raise FileNotFoundError(
                f'{name} not found at "{path}". '
                f'Ensure it is on your PATH or provide an absolute path in the config.'
            )

    # lmplz is needed whenever any corpus lacks a backend-appropriate
    # pre-built artifact (arpa for srilm, intermediate for kenlm).
    prebuilt_key = 'arpa_path' if backend == 'srilm' else 'intermediate_path'
    needs_lmplz = any(OmegaConf.select(c, prebuilt_key) is None for c in cfg.corpora)
    _check_tool(cfg.lmplz_path, 'lmplz', required=needs_lmplz)
    _check_tool(cfg.build_binary_path, 'build_binary')
    multi = len(cfg.corpora) > 1
    _check_tool(cfg.srilm_ngram_path, 'srilm ngram',
                required=multi and backend == 'srilm')
    _check_tool(cfg.kenlm_interpolate_path, 'kenlm interpolate',
                required=multi and backend == 'kenlm')

    # --- CMU dict for word mode ---
    if cfg.lm_type == 'word':
        # Default to the cmu_dict.pkl bundled with phoneme_to_words_lm.
        if not cfg.cmu_dict_path:
            OmegaConf.update(cfg, 'cmu_dict_path', _BUNDLED_CMU_DICT_PATH)
        if not os.path.exists(cfg.cmu_dict_path):
            raise FileNotFoundError(f'CMU dict not found: {cfg.cmu_dict_path}')

        # Optional English word list filter for OOV words.
        if cfg.english_words_path and not os.path.exists(cfg.english_words_path):
            raise FileNotFoundError(f'English word list not found: {cfg.english_words_path}')

    # --- Create output directories ---
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, 'normalized'), exist_ok=True)
    if backend == 'srilm':
        os.makedirs(os.path.join(cfg.output_dir, 'arpa'), exist_ok=True)
    else:
        os.makedirs(os.path.join(cfg.output_dir, 'intermediate'), exist_ok=True)

    return cfg


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _concat_normalized(input_paths: list[str], output_path: str, lm_type: str) -> None:
    """Concatenate normalized text files into a single output file.

    For ``word`` mode this is a byte-level concat. For ``spelling`` mode the
    per-file outputs are deduplicated character-expanded words; we re-dedupe
    across files so a word appearing in multiple inputs still occupies one
    line in the merged output (matching ``normalize_corpus`` semantics).
    """
    if lm_type == 'spelling':
        seen: set[str] = set()
        for ip in input_paths:
            with open(ip, 'r') as f:
                for line in f:
                    entry = line.rstrip('\n')
                    if entry:
                        seen.add(entry)
        with open(output_path, 'w') as f:
            for entry in sorted(seen):
                f.write(entry + '\n')
    else:
        with open(output_path, 'wb') as fout:
            for ip in input_paths:
                with open(ip, 'rb') as fin:
                    shutil.copyfileobj(fin, fout)


def train_ngram_lm(cfg) -> None:
    """Run the full n-gram LM training pipeline."""
    t_start = time.time()

    # ---- Step 1: Normalize corpora ----
    log.info('Step 1: Text normalization')
    all_vocab: set[str] = set()
    normalized_paths: list[str] = []

    # Backend-dependent per-corpus "pre-built artifact" field: when set,
    # the corpus is already trained and step 2 skips lmplz for it.
    prebuilt_key = 'arpa_path' if cfg.interpolator_backend == 'srilm' else 'intermediate_path'

    for i, c in enumerate(cfg.corpora):
        norm_path = os.path.join(cfg.output_dir, 'normalized', f'{c.name}.txt')

        if OmegaConf.select(c, prebuilt_key) is not None:
            kind = 'ARPA' if prebuilt_key == 'arpa_path' else 'intermediate'
            log.info(f'  Corpus [{i}] ({c.name}): using existing {kind}, skipping normalization')
            # Still need to collect vocab for word-mode lexicon generation.
            # Try normalized_path, then path, then skip vocab collection.
            if cfg.lm_type == 'word':
                src = OmegaConf.select(c, 'normalized_path') or OmegaConf.select(c, 'path')
                if src:
                    for sp in _as_path_list(src):
                        log.info(f'  Corpus [{i}] ({c.name}): scanning {sp} for vocab')
                        all_vocab |= collect_vocab_from_normalized(sp)
            normalized_paths.append(None)  # No normalized file for this corpus.
            continue

        if OmegaConf.select(c, 'normalized_path') is not None:
            norm_srcs = _as_path_list(c.normalized_path)
            if len(norm_srcs) == 1:
                log.info(f'  Corpus [{i}] ({c.name}): using pre-normalized text from {norm_srcs[0]}')
                shutil.copy2(norm_srcs[0], norm_path)
            else:
                log.info(f'  Corpus [{i}] ({c.name}): concatenating {len(norm_srcs)} pre-normalized files into {norm_path}')
                for ns in norm_srcs:
                    log.info(f'    - {ns}')
                _concat_normalized(norm_srcs, norm_path, cfg.lm_type)
            if cfg.lm_type == 'word':
                all_vocab |= collect_vocab_from_normalized(norm_path)
            normalized_paths.append(norm_path)
            continue

        src_paths = _as_path_list(c.path)
        if len(src_paths) == 1:
            log.info(f'  Corpus [{i}] ({c.name}): normalizing {src_paths[0]} '
                     f'(workers={cfg.normalize_workers})')
            vocab = normalize_corpus(src_paths[0], norm_path, cfg.lm_type,
                                     workers=cfg.normalize_workers)
            all_vocab |= vocab
        else:
            log.info(f'  Corpus [{i}] ({c.name}): normalizing and concatenating '
                     f'{len(src_paths)} files into {norm_path} '
                     f'(workers={cfg.normalize_workers})')
            for sp in src_paths:
                log.info(f'    - {sp}')
            # Normalize each input to a per-file part, then merge. The parts
            # live next to the final norm_path so a crash leaves them on disk
            # for inspection; they're cleaned up only after a successful merge.
            part_paths = [f'{norm_path}.part{j}' for j in range(len(src_paths))]
            for sp, pp in zip(src_paths, part_paths):
                vocab = normalize_corpus(sp, pp, cfg.lm_type,
                                         workers=cfg.normalize_workers)
                all_vocab |= vocab
            _concat_normalized(part_paths, norm_path, cfg.lm_type)
            for pp in part_paths:
                os.remove(pp)
        normalized_paths.append(norm_path)

    # ---- Step 2: Train individual LMs ----
    log.info('Step 2: Training individual n-gram models')
    # For srilm, these are per-corpus ARPA files. For kenlm, these are
    # intermediate-file prefixes (lmplz writes <prefix>.1..N + sidecars).
    model_paths: list[str] = []
    use_intermediate = cfg.interpolator_backend == 'kenlm'
    if use_intermediate:
        model_dir = os.path.join(cfg.output_dir, 'intermediate')
    else:
        model_dir = os.path.join(cfg.output_dir, 'arpa')

    for i, c in enumerate(cfg.corpora):
        if use_intermediate:
            out_path = os.path.join(model_dir, c.name)  # prefix
        else:
            out_path = os.path.join(model_dir, f'{c.name}.arpa')

        reuse = OmegaConf.select(c, prebuilt_key)
        if reuse is not None:
            log.info(f'  Corpus [{i}] ({c.name}): using existing artifact from {reuse}')
            # For srilm we copy the ARPA into the run dir for portability.
            # For kenlm the intermediate is a prefix spanning several
            # files; copying them would double disk use unnecessarily, so
            # we reference the user-provided prefix directly.
            if use_intermediate:
                out_path = reuse
            else:
                shutil.copy2(reuse, out_path)
            model_paths.append(out_path)
            continue

        pruning = list(c.pruning) if OmegaConf.select(c, 'pruning') is not None else None

        train_single_lm(
            text_path=normalized_paths[i],
            output_path=out_path,
            order=c.order,
            pruning=pruning,
            memory=cfg.memory,
            lmplz_path=cfg.lmplz_path,
            intermediate=use_intermediate,
            discount_fallback=(cfg.lm_type == 'spelling'),
        )
        model_paths.append(out_path)

    # ---- Step 3: Interpolate (if multi-corpus) ----
    final_arpa = os.path.join(cfg.output_dir, 'lm.arpa')

    if len(model_paths) == 1:
        log.info('Step 3: Single corpus — skipping interpolation')
        if use_intermediate:
            # KenLM intermediate can't be consumed by build_binary; we
            # need an ARPA. Rebuild the single corpus as ARPA from its
            # normalized text. arpa_path reuse is already rejected in
            # kenlm mode, so normalized_paths[0] is guaranteed non-None.
            log.info('  (kenlm backend) re-running lmplz to produce ARPA for downstream binary compilation')
            c = cfg.corpora[0]
            pruning = list(c.pruning) if OmegaConf.select(c, 'pruning') is not None else None
            train_single_lm(
                text_path=normalized_paths[0],
                output_path=final_arpa,
                order=c.order,
                pruning=pruning,
                memory=cfg.memory,
                lmplz_path=cfg.lmplz_path,
                intermediate=False,
                discount_fallback=(cfg.lm_type == 'spelling'),
            )
        else:
            shutil.copy2(model_paths[0], final_arpa)
    else:
        backend_name = cfg.interpolator_backend.upper()
        log.info(f'Step 3: Interpolating models with {backend_name}')
        raw_weights = [c.weight for c in cfg.corpora]
        weight_sum = sum(raw_weights)
        weights = [w / weight_sum for w in raw_weights]
        log.info(f'  Raw weights: {raw_weights} -> Normalized: {[f"{w:.4f}" for w in weights]}')

        if cfg.interpolator_backend == 'srilm':
            max_order = max(c.order for c in cfg.corpora)
            interpolate_lms_srilm(
                arpa_paths=model_paths,
                weights=weights,
                output_arpa=final_arpa,
                max_order=max_order,
                srilm_path=cfg.srilm_ngram_path,
            )
        else:
            interpolate_lms_kenlm(
                intermediate_paths=model_paths,
                weights=weights,
                output_arpa=final_arpa,
                kenlm_interpolate_path=cfg.kenlm_interpolate_path,
                memory=cfg.memory,
                temp_prefix=os.path.join(cfg.output_dir, 'tmp_interpolate/'),
            )

    # ---- Step 4: Compile to binary ----
    log.info('Step 4: Compiling to KenLM binary (trie format)')
    bin_path = os.path.join(cfg.output_dir, 'lm_unpruned.bin')
    compile_to_binary(final_arpa, bin_path, cfg.build_binary_path)

    # ---- Step 5: Generate lexicon ----
    log.info('Step 5: Generating lexicon')
    cmu_dict = None
    english_words = None
    if cfg.lm_type == 'word':
        cmu_dict = load_cmu_dict(cfg.cmu_dict_path)
        if cfg.english_words_path:
            with open(cfg.english_words_path) as f:
                english_words = {line.strip().lower() for line in f if line.strip()}
            log.info(f'  Loaded {len(english_words)} words from {cfg.english_words_path}')

    lexicon_path = os.path.join(cfg.output_dir, 'lexicon.txt')
    generate_lexicon(cfg.lm_type, all_vocab, cmu_dict, lexicon_path, english_words)

    # ---- Step 6: Generate tokens ----
    log.info('Step 6: Generating tokens.txt')
    tokens_path = os.path.join(cfg.output_dir, 'tokens.txt')
    generate_tokens(tokens_path)

    # ---- Summary ----
    elapsed = time.time() - t_start
    log.info(f'Done in {elapsed:.1f}s')
    log.info(f'Output directory: {cfg.output_dir}')
    for fname in ['lm.arpa', 'lm_unpruned.bin', 'lexicon.txt', 'tokens.txt']:
        fpath = os.path.join(cfg.output_dir, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            log.info(f'  {fname}: {size / 1e6:.1f} MB')


def main():
    parser = argparse.ArgumentParser(
        description='Train KenLM n-gram language model for brain-to-text BCI decoding'
    )
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )

    cfg = load_and_validate_config(args.config)

    # Tee log output into output_dir/log.txt alongside stderr.
    log_path = os.path.join(cfg.output_dir, 'log.txt')
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logging.getLogger().addHandler(file_handler)
    log.info(f'Logging to {log_path}')

    # save cfg to output_dir for record-keeping
    cfg_out_path = os.path.join(cfg.output_dir, 'config.yaml')
    with open(cfg_out_path, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

    log.info(f'Training {cfg.lm_type} LM (interpolator_backend={cfg.interpolator_backend}, '
             f'normalize_workers={cfg.normalize_workers})')
    log.info(f'Output directory: {cfg.output_dir}')
    log.info(f'Corpora ({len(cfg.corpora)}):')
    for i, c in enumerate(cfg.corpora):
        source = ('arpa_path' if OmegaConf.select(c, 'arpa_path') else
                  'intermediate_path' if OmegaConf.select(c, 'intermediate_path') else
                  'normalized_path' if OmegaConf.select(c, 'normalized_path') else 'path')
        source_val = OmegaConf.select(c, source)
        weight_str = f', weight={c.weight}' if OmegaConf.select(c, 'weight') is not None else ''
        if isinstance(source_val, str):
            log.info(f'  [{i}] name={c.name}, {source}={source_val}, '
                     f'order={c.order}{weight_str}')
        else:
            srcs = list(source_val)
            log.info(f'  [{i}] name={c.name}, {source}=[{len(srcs)} files], '
                     f'order={c.order}{weight_str}')
            for sp in srcs:
                log.info(f'        - {sp}')

    train_ngram_lm(cfg)


if __name__ == '__main__':
    main()
