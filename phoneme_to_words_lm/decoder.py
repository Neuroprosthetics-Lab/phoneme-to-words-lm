
from __future__ import annotations
from flashlight.lib.text.decoder import CriterionType, LexiconDecoderOptions, KenLM, LM, Trie, SmearingMode, LexiconDecoder
from flashlight.lib.text.dictionary import Dictionary, load_words, create_word_dict
try:
    from flashlight.lib.text.decoder import HotwordLM
except ImportError:  # Older installed extensions still support the Python path.
    HotwordLM = None
import math
import time
import torch
import os
import yaml
import json
import warnings
from numbers import Real
from collections import Counter
from phoneme_to_words_lm.llm_scoring import (
    DEFAULT_PREFIX, SCORER_VERSION, positive_int, validate_contexts, score_sentences,
)
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer
from typing import Literal, Optional, List, Dict
from phoneme_to_words_lm.utils import remove_punctuation, replace_words, HF_CACHE_DIR


class BiasingLM(LM):
    """Compatibility/reference wrapper for extensions without native HotwordLM.

    Implements the flashlight ``LM`` interface by delegating ``start``,
    ``score``, and ``finish`` to an inner ``KenLM`` instance. The override
    on ``score`` adds ``hotword_bonus[label]`` (zero by default) to the
    KenLM log-prob, producing the final per-word language-model score
    seen by ``LexiconDecoder`` at word completion.

    The bonus is in log10 space (matching KenLM's score units). Magnitude
    can be mutated in place at any utterance boundary; the *set* of
    hotword indices, however, also seeds the beam-search trie via
    ``TRIE_HOTWORD_BIAS`` so partial-word paths survive beam pruning,
    which means changing the *set* requires a trie rebuild upstream.
    """

    def __init__(self, inner: KenLM):
        super().__init__()
        self.inner = inner
        # word_idx -> log10 bonus added to KenLM's score for that word.
        self.hotword_bonus: Dict[int, float] = {}

    def start(self, start_with_nothing: bool):
        return self.inner.start(start_with_nothing)

    def score(self, state, label: int):
        new_state, s = self.inner.score(state, label)
        bonus = self.hotword_bonus.get(label)
        if bonus is not None:
            s += bonus
        return new_state, s

    def finish(self, state):
        return self.inner.finish(state)


class KenLMFlashlightTextLM:
    """KenLM/flashlight-text n-gram lexicon decoder with optional LLM rescoring.

    Two-stage pipeline for brain-to-text (phoneme CTC logits -> word sequences):

    1. **N-gram beam search** (``ngram_decode``): Runs flashlight-text's
       ``LexiconDecoder`` with a KenLM language model to produce an n-best
       list of word sequences, each with separate acoustic and n-gram LM scores.

    2. **LLM rescoring** (``llm_rescore``, optional): Scores the n-best
       hypotheses with a causal language model and re-ranks via this equation:

           final_score = beam_score + llm_alpha * llm_score

    Input logits must be contiguous CPU float32 raw (pre-softmax) tensors of
    shape (B, T, N), with columns matching tokens.txt. Lengths, when supplied,
    are CPU integer tensors of shape (B,). Preprocessing applies temperature
    scaling, log-softmax, and blank penalty. Optional blank compression keeps
    the first original frame of each high-confidence blank run; full CTC
    search is the default.

    Each output hypothesis dict contains::

        {
            'word_seqs':       List[str],   # decoded word sequences, best first
            'ngram_scores':    List[float], # raw KenLM log-prob sums
            'acoustic_scores': List[float], # raw acoustic (emitting model) log-prob sums
            'beam_scores':    List[float], # native totals, including search penalties
            'final_scores':    List[float], # scores used for ranking (combined or LLM-rescored)
        }

    Native beam totals include log-add merging and search penalties; they
    cannot in general be reconstructed from the representative acoustic and
    n-gram scores. KenLM uses log10; acoustic and LLM scores use natural logs.
    Raw word IDs and spellings are retained beside normalized ``word_seqs``.

    ``raw_llm_scores`` and ``llm_token_counts`` retain target sums and exact
    counts under the sentence_v1 policy. ``llm_scores`` subtract the per-token
    length penalty. Unscored entries are None with zero counts. An empty
    native winner retains its finite scores and status empty. Missing input
    or paths use [''] and -inf scores, with status no_input or no_path.
    Exact score ties are broken by text.

    Initialization is split into three methods for efficient hyperparameter
    sweeps (all called by ``__init__``):

    - ``init_ngram_resources()``: Loads KenLM, dictionary, lexicon, and trie
      from disk. Only needs to be re-called when file paths change.
    - ``init_ngram_decoder()``: Builds the ``LexiconDecoder`` from already-loaded
      resources. Cheap — re-call when beam search params change (``lm_weight``,
      ``beam_size``, ``word_score``, etc.).
    - ``init_llm()``: Loads the LLM model and tokenizer. Only needs to be
      re-called when ``llm_model_name``, ``llm_device``, or ``llm_dtype`` change.

    Parameters like ``temperature``, ``blank_penalty``, ``n_best``,
    ``llm_alpha``, ``llm_length_penalty``, and ``llm_batch_size`` are read
    at decode/rescore time — updating ``self.<param>`` is sufficient, no
    re-initialization needed.

    **Online / streaming decoding** is supported via three methods that use
    flashlight's incremental beam search API:

    1. ``online_decode_begin()``: Initialize decoder state for a new utterance.
    2. ``online_decode_step(logits)``: Feed new frame(s) and get the current
       best sentence. Accepts one or more frames (useful for catch-up if the
       caller falls behind real-time).
    3. ``online_decode_end()``: Finalize decoding, retrieve the full n-best
       list, and optionally LLM-rescore. Returns the same format as the
       offline ``offline_decode()``.

    The online methods use the same ``ngram_decoder`` and beam parameters as
    offline decoding. One instance supports one operation/utterance at a time;
    offline use and resource changes are prohibited during an active stream.
    Call ``online_decode_abort()`` to abandon a stream. Streams retain history
    until the next begin/abort, so use finite utterances rather than unbounded
    audio streams.
    """

    def __init__(
        self,
        # --- paths ---
        lexicon_path: str,
        tokens_path: str,
        kenlm_model_path: str,
        # --- beam search ---
        beam_size: int = 1000,
        token_beam_size: int = 41,
        beam_threshold: float = 50.0,
        blank_skip_threshold: float = 1.0,
        lm_weight: float = 2.5,
        word_score: float = 0.0,
        unk_score: float = -math.inf,
        sil_score: float = 0.0,
        log_add: bool = True,
        n_best: int = 100,
        # --- logit preprocessing ---
        temperature: float = 1.0,
        blank_penalty: float = 9.0,
        # --- token names ---
        sil_token: str = "SIL",
        blank_token: str = "BLANK",
        unk_token: str = "<unk>",
        # --- LLM rescoring ---
        do_llm_rescoring: bool = True,
        llm_model_name: str = 'Qwen/Qwen3.5-4B',
        llm_cache_dir: str = HF_CACHE_DIR,
        llm_device: str = 'cuda:0',
        llm_dtype: str = 'bfloat16',
        llm_alpha: float = 0.55,
        llm_length_penalty: float = 0.0,
        llm_batch_size: int = 100,
        llm_lora_path: Optional[str] = None,
        # --- Hotword biasing ---
        hotwords: Optional[Dict[str, float]] = None,
        hotwords_path: Optional[str] = None,
        # sentence_v1 likelihood policy and allocation limits
        llm_prefix: str = DEFAULT_PREFIX,
        llm_score_eos: bool = False,
        llm_max_length: int = 512,
        llm_max_batch_tokens: int = 2048,
        llm_logprob_chunk_size: int = 16,
    ):
        """Initialize the decoder.

        Args:
            lexicon_path: Path to lexicon.txt (word -> phoneme mapping, tab-separated,
                each pronunciation ending with SIL).
            tokens_path: Path to tokens.txt (one token per line, order must match
                model output: BLANK, SIL, AA, AE, ...).
            kenlm_model_path: Path to KenLM binary (.bin) compiled from an ARPA file.
            beam_size: Maximum number of hypotheses kept at each decoding step.
            token_beam_size: Number of tokens considered per frame (limits search over
                large vocabularies). Set to vocab size (41) for phoneme models.
            beam_threshold: Prune hypotheses scoring more than this below the best.
            blank_skip_threshold: 1.0 disables compression. Below 1.0, retain
                only the first frame of each run whose temperature-scaled blank
                probability is >= this threshold, before the blank penalty.
                Retained frames keep all original token scores. Scores then
                describe the compressed sequence, not full-CTC likelihoods.
            lm_weight: Weight on n-gram scores during beam search. Its effect
                is already included in the native total used for final ranking.
            word_score: Per-word bonus/penalty during beam search.
            unk_score: Score for unknown words (default -inf to forbid them).
            sil_score: Score for silence tokens during beam search.
            log_add: If True, use log-add for merging hypothesis scores; if False, use max.
            n_best: Number of top hypotheses to keep for rescoring.
            temperature: Logit temperature before softmax. <1 sharpens, >1 flattens.
            blank_penalty: Multiplicative penalty on blank token probability.
                Applied as ``log_prob[blank] -= log(blank_penalty)``.
            sil_token: Name of the silence token in tokens.txt.
            blank_token: Name of the CTC blank token in tokens.txt.
            unk_token: Name of the unknown word token in the lexicon.
            do_llm_rescoring: Whether to load an LLM and rescore n-best hypotheses.
            llm_model_name: HuggingFace model name or path for the rescoring LLM.
            llm_cache_dir: HuggingFace cache directory for LLM weights.
            llm_device: Torch device string for the LLM (e.g. 'cuda:0').
            llm_dtype: Precision for LLM weights ('float16', 'bfloat16', 'float32').
            llm_alpha: Weight on LLM scores in the final rescoring formula:
                ``final = beam_score + llm_alpha * llm_score``.
            llm_length_penalty: Per-token penalty subtracted from LLM scores.
            llm_batch_size: Batch size for LLM inference during rescoring.
            llm_prefix: Textual conditioning prefix; sentence_v1 defaults to a newline.
            llm_score_eos: Whether to include EOS as a target (default False).
            llm_max_length: Maximum full context+candidate token count; error on overflow.
            llm_max_batch_tokens: Maximum padded B*T per model call (default 2048).
            llm_logprob_chunk_size: FP32 vocabulary rows materialized at once (default 16).
            llm_lora_path: Path to a LoRA adapter directory (from peft). If provided,
                the adapter is merged into the base LLM weights at load time via
                merge_and_unload(). None means use the base model as-is.
            hotwords: Optional initial mapping of word -> log10 bonus added to the
                language-model score for that word at decode time. Words must
                already be in the lexicon (no g2p fallback). The set of hotword
                indices also seeds a fixed bias into the beam-search trie so
                partial-word paths survive pruning. Use ``set_hotwords`` /
                ``clear_hotwords`` to mutate at runtime; mutations apply at the
                next utterance boundary.
            hotwords_path: Optional path to a YAML file containing a flat
                ``{word: bonus}`` mapping (same shape as ``hotwords``). When
                provided, the file is loaded and used as the initial hotword
                set. If ``hotwords`` is also non-empty, it is ignored and a
                warning is logged.
        """

        # store all params for use by init methods and decode/rescore
        self.lexicon_path = lexicon_path
        self.tokens_path = tokens_path
        self.kenlm_model_path = kenlm_model_path
        self.beam_size = beam_size
        self.token_beam_size = token_beam_size
        self.beam_threshold = beam_threshold
        self.blank_skip_threshold = blank_skip_threshold
        self.lm_weight = lm_weight
        self.word_score = word_score
        self.unk_score = unk_score
        self.sil_score = sil_score
        self.log_add = log_add
        self.n_best = n_best
        self.temperature = temperature
        self.blank_penalty = blank_penalty
        self.sil_token = sil_token
        self.blank_token = blank_token
        self.unk_token = unk_token
        self.do_llm_rescoring = do_llm_rescoring
        self.llm_model_name = llm_model_name
        self.llm_cache_dir = os.path.expanduser(llm_cache_dir)
        self.llm_device = llm_device
        self.llm_dtype = llm_dtype
        self.llm_alpha = llm_alpha
        self.llm_length_penalty = llm_length_penalty
        self.llm_batch_size = llm_batch_size
        self.llm_lora_path = llm_lora_path
        self.llm_prefix = llm_prefix
        self.llm_score_eos = llm_score_eos
        self.llm_max_length = llm_max_length
        self.llm_max_batch_tokens = llm_max_batch_tokens
        self.llm_logprob_chunk_size = llm_logprob_chunk_size
        self._validate_settings()

        # Hotword biasing state. ``_hotwords`` is the canonical (word -> bonus)
        # mapping currently applied to the decoder. ``_pending_hotwords`` is
        # set by ``set_hotwords`` and consumed at the next utterance boundary
        # so mid-decode swaps are impossible; ``None`` means nothing pending.
        # File path takes precedence: if ``hotwords_path`` is set, ``hotwords``
        # (inline) is ignored with a warning. Empty string is treated as unset
        # so callers can disable cleanly via YAML without a None literal.
        if hotwords_path and hotwords:
            warnings.warn('hotwords_path takes precedence over inline hotwords', UserWarning)
        if hotwords_path:
            self._hotwords: Dict[str, float] = _load_hotwords_from_file(hotwords_path)
        else:
            if hotwords is not None and not isinstance(hotwords, dict):
                raise TypeError('hotwords must be a mapping of word to finite bonus')
            self._hotwords: Dict[str, float] = dict(hotwords) if hotwords is not None else {}
        self._pending_hotwords: Optional[Dict[str, float]] = None

        # online streaming state
        self._reset_online_state()

        # initialize components
        self.init_ngram_resources()
        self.init_ngram_decoder()
        if self.do_llm_rescoring:
            self.init_llm()


    def _validate_settings(self):
        for name in ('beam_size', 'token_beam_size', 'n_best', 'llm_batch_size',
                     'llm_max_length', 'llm_max_batch_tokens', 'llm_logprob_chunk_size'):
            positive_int(name, getattr(self, name))
        for name in ('temperature', 'blank_penalty', 'beam_threshold', 'lm_weight',
                     'word_score', 'sil_score', 'llm_alpha', 'llm_length_penalty'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f'{name} must be finite, got {value!r}')
        if self.temperature <= 0 or self.blank_penalty <= 0:
            raise ValueError('temperature and blank_penalty must be positive')
        if self.beam_threshold < 0:
            raise ValueError('beam_threshold must be nonnegative')
        if (
            isinstance(self.unk_score, bool)
            or not isinstance(self.unk_score, Real)
            or math.isnan(self.unk_score)
            or self.unk_score == math.inf
        ):
            raise ValueError('unk_score must be finite or -inf')
        if not isinstance(self.log_add, bool):
            raise ValueError('log_add must be bool')
        if (
            isinstance(self.blank_skip_threshold, bool)
            or not isinstance(self.blank_skip_threshold, Real)
            or not 0 < self.blank_skip_threshold <= 1
        ):
            raise ValueError('blank_skip_threshold must be in (0, 1]')
        if self.llm_dtype not in ('float16', 'bfloat16', 'float32'):
            raise ValueError('llm_dtype must be float16, bfloat16, or float32')
        if not isinstance(self.llm_prefix, str) or not self.llm_prefix:
            raise ValueError('llm_prefix must be a nonempty textual prefix')
        if not isinstance(self.llm_score_eos, bool):
            raise ValueError('llm_score_eos must be bool')

    def _require_idle(self):
        if self._online_active:
            raise RuntimeError('An online utterance is active; call online_decode_end() or online_decode_abort() first')

    def _validate_logits(self, logits, lengths=None):
        if not isinstance(logits, torch.Tensor):
            raise TypeError('logits must be a torch.Tensor')
        if logits.dtype != torch.float32 or logits.device.type != 'cpu':
            raise ValueError('logits must be CPU float32')
        if logits.ndim != 3 or logits.shape[2] != self.token_dict.index_size():
            raise ValueError(f'logits must have shape (B, T, {self.token_dict.index_size()})')
        if not logits.is_contiguous():
            raise ValueError('logits must be contiguous')
        b, t, _ = logits.shape
        if lengths is None:
            lengths = torch.full((b,), t, dtype=torch.int64)
        if not isinstance(lengths, torch.Tensor) or lengths.device.type != 'cpu':
            raise ValueError('lengths must be a CPU integer tensor of shape (B,)')
        if (
            lengths.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
            or lengths.shape != (b,)
        ):
            raise ValueError('lengths must be an integer tensor of shape (B,)')
        if bool(((lengths < 0) | (lengths > t)).any()):
            raise ValueError(f'lengths must satisfy 0 <= length <= {t}')
        for row, length in zip(logits, lengths.tolist()):
            if not bool(torch.isfinite(row[:length]).all()):
                raise ValueError('valid logit frames must be finite (no NaN or infinity)')
        return lengths

    @staticmethod
    def _empty_result(status='no_path'):
        return dict(word_seqs=[''], raw_word_seqs=[''], word_ids=[[]],
                    ngram_scores=[-math.inf], acoustic_scores=[-math.inf],
                    beam_scores=[-math.inf], final_scores=[-math.inf],
                    raw_llm_scores=[None], llm_scores=[None], llm_token_counts=[0],
                    status=status)

    def init_ngram_resources(self):
        """
        Load KenLM, dictionary, lexicon, and trie from disk.

        Only needs to be re-called when file paths
        (``lexicon_path``, ``tokens_path``, ``kenlm_model_path``) or token
        names (``sil_token``, ``blank_token``, ``unk_token``) change.
        Must be followed by ``init_ngram_decoder()`` to rebuild the decoder.
        """
        self._require_idle()
        self._validate_settings()
        self._search_config = None  # A resource reload must be followed by decoder init.
        for name in ('lexicon_path', 'tokens_path', 'kenlm_model_path'):
            path = os.path.expanduser(os.fspath(getattr(self, name)))
            if not os.path.isfile(path):
                raise FileNotFoundError(f'{name} not found: {path}')
            setattr(self, name, path)
        with open(self.tokens_path) as stream:
            tokens = [line.strip() for line in stream]
        if not tokens or any(not t or len(t.split()) != 1 for t in tokens) or len(set(tokens)) != len(tokens):
            raise ValueError('tokens.txt must contain one distinct nonempty token per line')
        if (
            self.sil_token == self.blank_token
            or self.sil_token not in tokens
            or self.blank_token not in tokens
        ):
            raise ValueError('tokens must contain distinct silence and blank tokens')
        self.token_dict = Dictionary(self.tokens_path)
        self.lexicon = load_words(self.lexicon_path)
        # Repeated rows describe one pronunciation, not additional CTC paths.
        for word, spellings in self.lexicon.items():
            self.lexicon[word] = [list(phones) for phones in dict.fromkeys(map(tuple, spellings))]
        self.word_dict = create_word_dict(self.lexicon)
        self._hotword_lookup = {}
        counts = Counter()
        for word, spellings in self.lexicon.items():
            if not spellings:
                continue
            self._hotword_lookup.setdefault(word.strip().casefold(), []).append(word)
            for spelling in spellings:
                if any(token not in tokens or token == self.blank_token for token in spelling):
                    raise ValueError(f'Invalid pronunciation tokens for {word!r}: {spelling}')
                if spelling[-1] != self.sil_token:
                    raise ValueError(f'Pronunciation for {word!r} must end with {self.sil_token}')
                counts[tuple(spelling)] += 1
        if not counts:
            raise ValueError('lexicon has no usable pronunciations')
        # Check installed extension capacity, rather than assuming its patch.
        probe = Trie(len(tokens), tokens.index(self.sil_token))
        node = None
        for label in range(max(counts.values())):
            node = probe.insert([tokens.index(self.sil_token)], label, 0.)
        if len(node.labels) != max(counts.values()):
            raise ValueError('Installed Flashlight trie cannot retain every homophone; rebuild with sufficient label capacity')
        # Keep the large KenLM shared across hotword updates and trie rebuilds.
        self._kenlm = KenLM(self.kenlm_model_path, self.word_dict)

        self.sil_idx = self.token_dict.get_index(self.sil_token)
        self.blank_idx = self.token_dict.get_index(self.blank_token)
        self.unk_idx = self.word_dict.get_index(self.unk_token)

        # Resolve current hotwords into (word_idx -> bonus) and a set of
        # word_indices used to bias the trie's lookahead scores.
        hotword_idx_set, hotword_idx_bonus = self._resolve_hotwords(self._hotwords)
        self._hotwords = {k.strip().casefold(): float(v) for k, v in self._hotwords.items()}
        self.lm = self._make_hotword_lm(hotword_idx_bonus)
        self._hotword_idx_set = hotword_idx_set

        self.trie = _construct_trie(
            self.token_dict, self.word_dict, self.lexicon, self._kenlm,
            self.sil_idx, hotword_idx_set,
        )


    def _make_ngram_decoder(self, trie, lm=None):
        options = LexiconDecoderOptions(
            self.beam_size,
            self.token_beam_size,
            self.beam_threshold,
            self.lm_weight,
            self.word_score,
            self.unk_score,
            self.sil_score,
            self.log_add,
            CriterionType.CTC,
        )
        return LexiconDecoder(
            options,
            trie,
            self.lm if lm is None else lm,
            self.sil_idx,
            self.blank_idx,
            self.unk_idx,
            [],     # transitions (empty for CTC)
            False,  # is_token_lm
        )

    def init_ngram_decoder(self):
        """Build the LexiconDecoder from already-loaded resources.

        Re-call whenever beam search parameters change (``beam_size``,
        ``token_beam_size``, ``beam_threshold``, ``lm_weight``, ``word_score``,
        ``unk_score``, ``sil_score``, ``log_add``).
        Requires ``init_ngram_resources()`` to have been called first.
        Rebuilding during an active stream is prohibited.
        """
        self._require_idle()
        self._validate_settings()
        self.ngram_decoder = self._make_ngram_decoder(self.trie)
        self._search_config = self._current_search_config()

    def _current_search_config(self):
        names = ('beam_size', 'token_beam_size', 'beam_threshold', 'lm_weight',
                 'word_score', 'unk_score', 'sil_score', 'log_add')
        return {name: getattr(self, name) for name in names}

    def _check_search_config(self):
        if self._search_config != self._current_search_config():
            raise RuntimeError('Search options changed; call init_ngram_decoder() before decoding')

    def init_llm(self):
        """Load the LLM model and tokenizer for rescoring.

        Only needs to be re-called when
        ``llm_model_name``, ``llm_device``, or ``llm_dtype`` change.
        """
        self._require_idle()
        self._validate_settings()
        if self.llm_lora_path is not None:
            policy_path = os.path.join(os.path.expanduser(self.llm_lora_path), 'llm_scoring.json')
            if os.path.isfile(policy_path):
                with open(policy_path) as stream:
                    policy = json.load(stream)
                expected = dict(version=SCORER_VERSION, prefix=self.llm_prefix, score_eos=self.llm_score_eos)
                if policy != expected:
                    raise ValueError(f'Adapter scoring policy {policy!r} differs from decoder policy {expected!r}')
        self.llm_model, self.llm_tokenizer = _build_llm(
            self.llm_model_name,
            cache_dir=self.llm_cache_dir,
            device=self.llm_device,
            dtype=self.llm_dtype,
            lora_path=self.llm_lora_path,
        )


    # =========================================================================
    # Hotword biasing
    # =========================================================================

    @property
    def hotword_backend(self):
        """Active score-time backend: kenlm, native, or python compatibility."""
        if self.lm is self._kenlm:
            return 'kenlm'
        return 'native' if HotwordLM is not None and isinstance(self.lm, HotwordLM) else 'python'

    def _make_hotword_lm(self, bonuses):
        if not bonuses:
            return self._kenlm
        if HotwordLM is not None:
            return HotwordLM(self._kenlm, bonuses)
        warnings.warn(
            'Installed Flashlight has no native HotwordLM; using slower Python '
            'hotword scoring. Rebuild/install this repository\'s flashlight_text '
            'to enable native scoring.', RuntimeWarning, stacklevel=2)
        lm = BiasingLM(self._kenlm)
        lm.hotword_bonus = bonuses
        return lm

    def _resolve_hotwords(self, hotwords):
        if not isinstance(hotwords, dict):
            raise TypeError('hotwords must be a mapping of word to finite bonus')
        indices, bonuses, seen, missing = set(), {}, set(), []
        for word, bonus in hotwords.items():
            if not isinstance(word, str):
                raise TypeError('hotword keys must be strings')
            key = word.strip().casefold()
            if key in seen:
                raise ValueError(f'Duplicate normalized hotword: {key!r}')
            seen.add(key)
            if isinstance(bonus, bool) or not isinstance(bonus, Real) or not math.isfinite(bonus):
                raise ValueError(f'Hotword bonus for {word!r} must be finite')
            entries = self._hotword_lookup.get(key, [])
            if not entries:
                missing.append(word)
                continue
            if len(entries) > 1:
                raise ValueError(f'Ambiguous hotword {word!r}; lexicon entries: {entries}')
            idx = self.word_dict.get_index(entries[0])
            indices.add(idx)
            bonuses[idx] = float(bonus)
        if missing:
            raise KeyError(f'Hotwords not in lexicon: {missing!r}')
        return indices, bonuses


    def set_hotwords(self, hotwords: Dict[str, float]) -> None:
        """Replace the active hotword set with ``hotwords``.

        The application is always *deferred* to the next utterance boundary
        -- it runs at the top of ``online_decode_begin``, ``offline_decode``,
        or ``ngram_decode``. This avoids mutating decoder state mid-utterance,
        including the score-time bonuses read by the active LM.

        At apply time, if the *set* of hotword indices is unchanged from the
        currently-applied set, only the LM's bonus table is updated
        (no trie rebuild). Otherwise the trie and ``LexiconDecoder`` are
        rebuilt.

        Raises ``KeyError`` if any entry isn't already in the lexicon.
        """
        # Validate eagerly so caller errors surface at set time, not later.
        # The resolved result is discarded; ``_apply_pending_hotwords`` will
        # re-resolve at apply time against the (then-current) lexicon state.
        self._resolve_hotwords(hotwords)
        self._pending_hotwords = dict(hotwords)

    def get_hotwords(self) -> Dict[str, float]:
        """Return a copy of the currently-applied hotword mapping.

        Pending (not-yet-applied) updates from ``set_hotwords`` are NOT
        reflected here -- this is the ground-truth state of the decoder.
        """
        return dict(self._hotwords)

    def clear_hotwords(self) -> None:
        """Remove all hotwords. Equivalent to ``set_hotwords({})``."""
        self.set_hotwords({})

    def _apply_pending_hotwords(self):
        self._require_idle()
        if self._pending_hotwords is None:
            return
        pending = self._pending_hotwords
        indices, bonuses = self._resolve_hotwords(pending)
        # Build replacements before publishing any scoring-state mutation.
        if indices != self._hotword_idx_set:
            trie = _construct_trie(self.token_dict, self.word_dict, self.lexicon,
                                   self._kenlm, self.sil_idx, indices)
            lm = self._make_hotword_lm(bonuses)
            decoder = self._make_ngram_decoder(trie, lm)
            self.trie, self.lm, self.ngram_decoder = trie, lm, decoder
        elif indices:
            self.lm.hotword_bonus = bonuses
        self._hotword_idx_set = indices
        self._hotwords = {k.strip().casefold(): float(v) for k, v in pending.items()}
        self._pending_hotwords = None

    def offline_decode(self, logits, lengths=None, contexts=None):
        """Decode CPU raw logits and optionally rescore; one result per item.

        ngram_time is per-item preprocessing/search/extraction time. Packed LLM
        time is batch-scoped, explicitly named llm_batch_time; the legacy
        llm_rescore_time alias remains batch-scoped for compatibility.
        """
        validate_contexts(contexts, logits.shape[0] if isinstance(logits, torch.Tensor) and logits.ndim else 0)
        start = time.perf_counter()
        hypos = self.ngram_decode(logits, lengths)
        ngram_batch_time = time.perf_counter()-start
        llm_time = None
        if self.do_llm_rescoring:
            t = time.perf_counter()
            hypos = self.llm_rescore(hypos, contexts=contexts)
            llm_time = time.perf_counter()-t
        batch_total_time = time.perf_counter()-start
        for hypo in hypos:
            hypo.update(ngram_batch_time=ngram_batch_time, llm_batch_time=llm_time,
                        llm_rescore_time=llm_time, batch_size=len(hypos),
                        batch_total_time=batch_total_time)
        return hypos

    def process_logits(self, logits):
        """Temperature, log-softmax, then blank penalty; no frame deletion."""
        self._validate_settings()
        self._validate_logits(logits)
        processed = (logits / self.temperature).log_softmax(dim=-1)
        processed[:, :, self.blank_idx] -= math.log(self.blank_penalty)
        if not bool(torch.isfinite(processed).all()):
            raise ValueError('Logit preprocessing overflowed; adjust temperature or input scale')
        return processed

    def _select_blank_frames(self, log_probs, previous_high_blank=False):
        """Keep the first original frame of each high-confidence blank run.

        Accept processed (T,N) scores. The one-bit carry describes the last
        *input* frame, including discarded frames, so chunking cannot create
        extra separators. Empty chunks leave it unchanged. This is an
        approximation: omitted frames contribute neither scores nor paths.
        """
        if self.blank_skip_threshold == 1.0:
            return log_probs, False
        if not len(log_probs):
            return log_probs, previous_high_blank
        # Undo the blank penalty only for thresholding; retain scores as-is.
        high_blank = (log_probs[:, self.blank_idx] + math.log(self.blank_penalty)
                      >= math.log(self.blank_skip_threshold))
        keep = ~high_blank
        keep[0] |= not previous_high_blank
        keep[1:] |= ~high_blank[:-1]
        return log_probs[keep].contiguous(), bool(high_blank[-1])

    def _preprocessing_config(self):
        return dict(temperature=self.temperature, blank_penalty=self.blank_penalty,
                    blank_skip_threshold=self.blank_skip_threshold,
                    blank_skip_policy='keep_first' if self.blank_skip_threshold < 1.0 else 'disabled')

    def _extract_nbest(self, results, n_best, lm_weight=None):
        """Retain native totals, best representative per normalized string.

        lm_weight is accepted for source compatibility but never reconstructs
        native scores. Components are representative-path diagnostics.
        """
        positive_int('n_best', n_best)
        best, cutoff, top_score = {}, None, None
        for result in sorted(results, key=lambda r: r.score, reverse=True):
            if not math.isfinite(result.score):
                continue
            if top_score is None:
                top_score = result.score
            if cutoff is not None and result.score < cutoff:
                break
            ids = [x for x in result.words if x >= 0]
            # Abstain when no words were emitted by the best native path.
            # Include exact ties, irrespective of native result ordering.
            if not ids and result.score == top_score:
                empty = self._empty_result('empty')
                empty.update(beam_scores=[float(result.score)], final_scores=[float(result.score)],
                             ngram_scores=[float(result.lmScore)],
                             acoustic_scores=[float(result.emittingModelScore)])
                return empty
            raw = ' '.join(self.word_dict.get_entry(x) for x in ids)
            text = replace_words(remove_punctuation(raw))
            if not text:
                continue
            previous = best.get(text)
            if previous is None or (
                result.score == previous['beam_score'] and raw < previous['raw_text']
            ):
                best[text] = {
                    'text': text,
                    'raw_text': raw,
                    'word_ids': ids,
                    'ngram_score': float(result.lmScore),
                    'acoustic_score': float(result.emittingModelScore),
                    'beam_score': float(result.score),
                }
            if len(best) >= n_best and cutoff is None:
                cutoff = result.score
        # Native exact ties may arrive in allocator-dependent order. Process
        # the whole cutoff tie and use text as an explicit secondary key.
        rows = sorted(
            best.values(),
            key=lambda row: (-row['beam_score'], row['text'], row['raw_text']),
        )[:n_best]
        if not rows:
            return self._empty_result()
        return {
            'word_seqs': [row['text'] for row in rows],
            'raw_word_seqs': [row['raw_text'] for row in rows],
            'word_ids': [row['word_ids'] for row in rows],
            'ngram_scores': [row['ngram_score'] for row in rows],
            'acoustic_scores': [row['acoustic_score'] for row in rows],
            'beam_scores': [row['beam_score'] for row in rows],
            'final_scores': [row['beam_score'] for row in rows],
            'raw_llm_scores': [None] * len(rows),
            'llm_scores': [None] * len(rows),
            'llm_token_counts': [0] * len(rows),
            'status': 'ok',
        }

    def ngram_decode(self, logits, lengths=None):
        """Validate lengths, then optionally compress high-blank runs."""
        self._require_idle()
        self._validate_settings()
        self._check_search_config()
        lengths = self._validate_logits(logits, lengths)
        self._apply_pending_hotwords()
        hypos = []
        for row, length in zip(logits, lengths.tolist()):
            t = time.perf_counter()
            search_frames = 0
            if length == 0:
                hypo = self._empty_result('no_input')
            else:
                # Only valid frames are processed: padded NaNs do not matter.
                values = self.process_logits(row[:length].unsqueeze(0).contiguous())[0]
                values, _ = self._select_blank_frames(values)
                search_frames = len(values)
                results = self.ngram_decoder.decode(values.data_ptr(), search_frames, values.shape[1])
                hypo = self._extract_nbest(results, self.n_best)
            hypo.update(ngram_time=time.perf_counter()-t, total_frames=length,
                        search_frames=search_frames, search_config=dict(self._search_config),
                        preprocessing_config=self._preprocessing_config())
            hypos.append(hypo)
        return hypos

    def llm_rescore(self, hypotheses, contexts=None):
        """Add the LLM term to immutable beam totals; retain aligned metadata."""
        self._validate_settings()
        contexts = validate_contexts(contexts, len(hypotheses))
        sequences, seq_contexts, locations = [], [], []
        aligned = ('word_seqs', 'raw_word_seqs', 'word_ids', 'ngram_scores',
                   'acoustic_scores', 'beam_scores', 'raw_llm_scores',
                   'llm_scores', 'llm_token_counts', 'final_scores')
        for i, hypo in enumerate(hypotheses):
            if 'beam_scores' not in hypo:
                raise ValueError('llm_rescore requires beam_scores from the native decoder; legacy component-only results must be decoded again')
            size = len(hypo['word_seqs'])
            for key in aligned:
                if key in hypo and len(hypo[key]) != size:
                    raise ValueError(f'{key} is not aligned with word_seqs')
            for j, seq in enumerate(hypo['word_seqs']):
                if seq.strip():
                    if not math.isfinite(hypo['beam_scores'][j]):
                        raise ValueError('nonempty hypotheses require finite beam_scores')
                    locations.append((i, j))
                    sequences.append(seq)
                    seq_contexts.append(contexts[i])
        raw, counts = [], []
        if sequences:
            raw, counts = score_sentences(
                self.llm_model, self.llm_tokenizer, sequences, seq_contexts,
                prefix=self.llm_prefix, score_eos=self.llm_score_eos,
                max_length=self.llm_max_length, batch_size=self.llm_batch_size,
                max_batch_tokens=self.llm_max_batch_tokens,
                logprob_chunk_size=self.llm_logprob_chunk_size)
        output = []
        for hypo in hypotheses:
            copy = dict(hypo)
            n = len(hypo['word_seqs'])
            copy.update(raw_llm_scores=[None]*n, llm_scores=[None]*n,
                        llm_token_counts=[0]*n, final_scores=list(hypo['beam_scores']),
                        llm_scoring=dict(version=SCORER_VERSION, prefix=self.llm_prefix,
                                         score_eos=self.llm_score_eos,
                                         length_penalty=self.llm_length_penalty, alpha=self.llm_alpha))
            output.append(copy)
        for (i, j), score, count in zip(locations, raw, counts):
            hypo = output[i]
            adjusted_score = score - self.llm_length_penalty * count
            hypo['raw_llm_scores'][j] = score
            hypo['llm_scores'][j] = adjusted_score
            hypo['llm_token_counts'][j] = count
            hypo['final_scores'][j] = hypo['beam_scores'][j] + self.llm_alpha * adjusted_score
        for hypo in output:
            # Exact final ties must not depend on a previous rescoring order.
            order = sorted(range(len(hypo['word_seqs'])), key=lambda j: (
                -hypo['final_scores'][j], -hypo['beam_scores'][j], hypo['word_seqs'][j]))
            for key in aligned:
                if key in hypo:
                    hypo[key] = [hypo[key][j] for j in order]
        return output



    # =========================================================================
    # Online / streaming decoding
    # =========================================================================

    def _reset_online_state(self):
        """Clear Python stream state without touching the native decoder."""
        self._online_active = False
        self._online_total_frames = 0
        self._online_search_frames = 0
        self._online_prev_high_blank = False
        self._online_ngram_time = 0.0

    def online_decode_abort(self):
        """Explicitly abandon a stream and release its native history."""
        self._reset_online_state()
        # Reset the existing native instance with its installed options. Abort
        # must work even if the caller has since assigned an invalid setting.
        self.ngram_decoder.decode_begin()

    def online_decode_begin(self):
        """Start an utterance; abandon an active one only via explicit abort."""
        self._require_idle()
        self._validate_settings()
        self._check_search_config()
        self._apply_pending_hotwords()
        self.ngram_decoder.decode_begin()
        self._reset_online_state()
        self._online_stream_config = (self._preprocessing_config(), self.n_best)
        self._online_active = True

    def online_decode_step(self, logits):
        """Submit one complete contiguous chunk; total_frames includes blanks."""
        t = time.perf_counter()
        if not isinstance(logits, torch.Tensor):
            raise TypeError('logits must be a tensor')
        if logits.ndim == 2:
            logits = logits.unsqueeze(0)
        if logits.ndim != 3 or logits.shape[0] != 1:
            raise ValueError('online_decode_step expects (T,N) or (1,T,N)')
        logits = logits.contiguous()
        self._validate_logits(logits)
        self._validate_settings()
        self._check_search_config()
        if self._online_active and self._online_stream_config != (self._preprocessing_config(), self.n_best):
            raise RuntimeError('Preprocessing/n_best changed during an active stream; abort it before changing settings')
        values = self.process_logits(logits)[0]
        if not self._online_active:
            self.online_decode_begin()
        try:
            values, previous_high_blank = self._select_blank_frames(values, self._online_prev_high_blank)
            if len(values):
                self.ngram_decoder.decode_step(values.data_ptr(), len(values), values.shape[1])
            self._online_total_frames += logits.shape[1]
            self._online_search_frames += len(values)
            self._online_prev_high_blank = previous_high_blank
            if not self._online_search_frames:
                output = dict(word_seq='', score=0.0, ngram_score=0.0, acoustic_score=0.0)
            else:
                r = self.ngram_decoder.get_best_hypothesis()
                raw = ' '.join(self.word_dict.get_entry(x) for x in r.words if x >= 0)
                output = dict(word_seq=replace_words(remove_punctuation(raw)), score=r.score,
                              ngram_score=r.lmScore, acoustic_score=r.emittingModelScore)
            output.update(total_frames=self._online_total_frames, search_frames=self._online_search_frames)
        except Exception:
            self.online_decode_abort()
            raise
        self._online_ngram_time += time.perf_counter()-t
        return output

    def online_decode_end(self, context=None):
        """Finalize; always reset lifecycle state even if rescoring fails."""
        validate_contexts(None if context is None else [context], 1)
        try:
            self._validate_settings()
            self._check_search_config()
            if (
                self._online_active
                and self._online_stream_config != (self._preprocessing_config(), self.n_best)
            ):
                raise RuntimeError('Preprocessing/n_best changed during an active stream')
            if not self._online_active or not self._online_total_frames:
                hypo = self._empty_result('no_input')
                hypo.update(ngram_time=0.0, llm_rescore_time=None, total_frames=0, search_frames=0,
                            search_config=dict(self._search_config), preprocessing_config=self._preprocessing_config())
                return hypo
            t = time.perf_counter()
            self.ngram_decoder.decode_end()
            hypo = self._extract_nbest(self.ngram_decoder.get_all_final_hypothesis(), self.n_best)
            elapsed = self._online_ngram_time + time.perf_counter()-t
            llm_time = None
            if self.do_llm_rescoring:
                t = time.perf_counter()
                [hypo] = self.llm_rescore([hypo], contexts=None if context is None else [context])
                llm_time = time.perf_counter()-t
            hypo.update(ngram_time=elapsed, llm_rescore_time=llm_time,
                        total_frames=self._online_total_frames, search_frames=self._online_search_frames,
                        search_config=dict(self._search_config), preprocessing_config=self._preprocessing_config())
            return hypo
        finally:
            self._reset_online_state()



def _load_hotwords_from_file(path: str) -> Dict[str, float]:
    """Load a flat ``{word: bonus}`` mapping from a YAML file.

    The file must parse to a top-level mapping (or be empty/null, in which
    case an empty dict is returned). Duplicate keys are rejected before YAML
    can overwrite them; key and bonus types are validated by the caller. The
    path is run through ``os.path.expanduser`` so ``~`` is supported.
    """
    expanded = os.path.expanduser(path)
    if not os.path.exists(expanded):
        raise FileNotFoundError(f"hotwords_path does not exist: {expanded}")
    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def unique_mapping(loader, node):
        loader.flatten_mapping(node)
        data = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            if not isinstance(key, str):
                raise TypeError('hotword keys must be strings')
            if key in data:
                raise ValueError(f'Duplicate hotword YAML key: {key!r}')
            data[key] = loader.construct_object(value_node)
        return data

    UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)
    with open(expanded, 'r') as f:
        data = yaml.load(f, Loader=UniqueKeyLoader)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"hotwords file {expanded} must be a top-level mapping of "
            f"word -> bonus, got {type(data).__name__}."
        )
    return data


# Trie-side hotword bias (log10). Big enough to dominate lookahead-pruning
# decisions for hotword paths, but the value never reaches the final
# hypothesis score because the decoder subtracts ``lexMaxScore`` at word
# completion (see ``LexiconDecoder.cpp:127``). The actual ranking effect comes
# from ``BiasingLM.hotword_bonus`` at score-time; this constant only changes
# *which paths survive the beam* during partial-word traversal.
TRIE_HOTWORD_BIAS = 5.0


def _construct_trie(tokens_dict, word_dict, lexicon, lm, silence, hotword_idx_set=None):
    """Build a flashlight Trie from the lexicon, seeded with unigram LM scores.

    When ``hotword_idx_set`` is non-empty, the seed score for those word
    indices is bumped by ``TRIE_HOTWORD_BIAS`` so partial-word paths leading
    to them are kept alive by beam pruning even when the surrounding context
    doesn't favor them.
    """
    vocab_size = tokens_dict.index_size()
    trie = Trie(vocab_size, silence)
    start_state = lm.start(False)
    hotword_idx_set = hotword_idx_set or set()

    for word, spellings in lexicon.items():
        word_idx = word_dict.get_index(word)
        _, score = lm.score(start_state, word_idx)
        if word_idx in hotword_idx_set:
            score += TRIE_HOTWORD_BIAS
        for spelling in spellings:
            spelling_idx = [tokens_dict.get_index(token) for token in spelling]
            trie.insert(spelling_idx, word_idx, score)
    trie.smear(SmearingMode.MAX)
    return trie


def _build_llm(
    model_name: str,
    cache_dir: Optional[str] = None,
    device: str = 'cuda:0',
    dtype: Literal['float16', 'bfloat16', 'float32'] = 'bfloat16',
    lora_path: Optional[str] = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a causal language model and its tokenizer for rescoring.

    Args:
        model_name: HuggingFace model identifier or local path.
        cache_dir: Directory for cached model weights. Defaults to the
            HuggingFace default cache location when *None*.
        device: Torch device string to place the model on.
        dtype: Floating-point precision for model weights.
        lora_path: Optional path to a PEFT LoRA adapter directory. When
            provided, the adapter is loaded and merged into the base model
            weights so inference speed and memory are unchanged.

    Returns:
        A ``(model, tokenizer)`` tuple. The model is moved to *device*
        and set to eval mode. The tokenizer is configured with
        right-side padding and ``pad_token`` set to ``eos_token``.
    """

    if dtype not in ('float16', 'bfloat16', 'float32'):
        raise ValueError('dtype must be float16, bfloat16, or float32')
    torch_dtype = getattr(torch, dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=False)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        dtype=torch_dtype,
        attn_implementation='sdpa',
        local_files_only=False,
    )

    if lora_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()

    model.to(device).eval()

    return model, tokenizer


def _get_llm_scores(model, tokenizer, hypotheses, length_penalty=0.0, contexts=None, **kwargs):
    """sentence_v1 adjusted likelihood; raw sums/counts use score_sentences."""
    if not math.isfinite(length_penalty):
        raise ValueError('length_penalty must be finite')
    raw, counts = score_sentences(model, tokenizer, hypotheses, contexts, **kwargs)
    return [s-length_penalty*n for s, n in zip(raw, counts)]
