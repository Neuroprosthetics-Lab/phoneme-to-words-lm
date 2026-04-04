
from __future__ import annotations
from flashlight.lib.text.decoder import CriterionType, LexiconDecoderOptions, KenLM, Trie, SmearingMode, LexiconDecoder
from flashlight.lib.text.dictionary import Dictionary, load_words, create_word_dict
import math
import time
import numpy as np
import torch
import os
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer
from typing import Literal, Optional, List, Dict
from phoneme_to_words_lm.utils import remove_punctuation, replace_words, HF_CACHE_DIR


class KenLMFlashlightTextLM:
    """KenLM/flashlight-text n-gram lexicon decoder with optional LLM rescoring.

    Two-stage pipeline for brain-to-text (phoneme CTC logits -> word sequences):

    1. **N-gram beam search** (``ngram_decode``): Runs flashlight-text's
       ``LexiconDecoder`` with a KenLM language model to produce an n-best
       list of word sequences, each with separate acoustic and n-gram LM scores.

    2. **LLM rescoring** (``llm_rescore``, optional): Scores the n-best
       hypotheses with a causal language model and re-ranks via this equation:

           final_score = acoustic_score + lm_weight * ngram_score + llm_alpha * llm_score + word_score * num_words

    Input logits are expected to be raw (pre-softmax) and are automatically
    preprocessed with temperature scaling, log-softmax, and blank penalty.

    Each output hypothesis dict contains::

        {
            'word_seqs':       List[str],   # decoded word sequences, best first
            'ngram_scores':    List[float], # raw KenLM log-prob sums
            'acoustic_scores': List[float], # raw acoustic (emitting model) log-prob sums
            'final_scores':    List[float], # scores used for ranking (combined or LLM-rescored)
        }

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
    offline decoding.
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
            blank_skip_threshold: Skip frames where blank probability exceeds this
                value (0.0-1.0). Set to 1.0 to disable (default). Values like 0.95
                or 0.99 can speed up decoding with minimal accuracy impact.
            lm_weight: Weight on n-gram LM scores during beam search AND in the
                final rescoring formula. Controls the balance between acoustic and
                language scores.
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
            llm_dtype: Precision for LLM weights ('float16' or 'bfloat16').
            llm_alpha: Weight on LLM scores in the final rescoring formula:
                ``final = acoustic + lm_weight * ngram + llm_alpha * llm + word_score * num_words``.
            llm_length_penalty: Per-token penalty subtracted from LLM scores.
            llm_batch_size: Batch size for LLM inference during rescoring.
            llm_lora_path: Path to a LoRA adapter directory (from peft). If provided,
                the adapter is merged into the base LLM weights at load time via
                merge_and_unload(). None means use the base model as-is.
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

        # online streaming state
        self._online_active = False
        self._online_total_frames = 0

        # initialize components
        self.init_ngram_resources()
        self.init_ngram_decoder()
        if self.do_llm_rescoring:
            self.init_llm()


    def init_ngram_resources(self):
        """
        Load KenLM, dictionary, lexicon, and trie from disk.

        Only needs to be re-called when file paths
        (``lexicon_path``, ``tokens_path``, ``kenlm_model_path``) or token
        names (``sil_token``, ``blank_token``, ``unk_token``) change.
        Must be followed by ``init_ngram_decoder()`` to rebuild the decoder.
        """
        assert os.path.exists(self.lexicon_path), f"Lexicon file not found: {self.lexicon_path}"
        assert os.path.exists(self.tokens_path), f"Tokens file not found: {self.tokens_path}"
        assert os.path.exists(self.kenlm_model_path), f"KenLM model file not found: {self.kenlm_model_path}"

        self.token_dict = Dictionary(self.tokens_path)
        self.lexicon = load_words(self.lexicon_path)
        self.word_dict = create_word_dict(self.lexicon)
        self.lm = KenLM(self.kenlm_model_path, self.word_dict)

        self.sil_idx = self.token_dict.get_index(self.sil_token)
        self.blank_idx = self.token_dict.get_index(self.blank_token)
        self.unk_idx = self.word_dict.get_index(self.unk_token)

        self.trie = _construct_trie(self.token_dict, self.word_dict, self.lexicon, self.lm, self.sil_idx)


    def init_ngram_decoder(self):
        """Build the LexiconDecoder from already-loaded resources.

        Re-call whenever beam search parameters change (``beam_size``,
        ``token_beam_size``, ``beam_threshold``, ``lm_weight``, ``word_score``,
        ``unk_score``, ``sil_score``, ``log_add``).
        Requires ``init_ngram_resources()`` to have been called first.
        """
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
        self.ngram_decoder = LexiconDecoder(
            options,
            self.trie,
            self.lm,
            self.sil_idx,
            self.blank_idx,
            self.unk_idx,
            [],     # transitions (empty for CTC)
            False,  # is_token_lm
        )


    def init_llm(self):
        """Load the LLM model and tokenizer for rescoring.

        Only needs to be re-called when
        ``llm_model_name``, ``llm_device``, or ``llm_dtype`` change.
        """
        self.llm_model, self.llm_tokenizer = _build_llm(
            self.llm_model_name,
            cache_dir=self.llm_cache_dir,
            device=self.llm_device,
            dtype=self.llm_dtype,
            lora_path=self.llm_lora_path,
        )


    def offline_decode(
        self,
        logits: torch.FloatTensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> List[Dict]:
        """Run the full offline decode pipeline: n-gram beam search + optional LLM rescoring.

        Args:
            logits: CPU float32 tensor of shape ``(batch, time, num_tokens)``
                containing raw (pre-softmax) logit outputs from the acoustic model.
                Must be contiguous.
            lengths: Optional CPU int tensor of shape ``(batch,)`` with the valid
                number of time steps per sample. If None, all frames are used.

        Returns:
            List of hypothesis dicts (one per batch item), sorted best-first.
            See class docstring for the dict format.
        """

        t0 = time.perf_counter()
        hypos = self.ngram_decode(logits, lengths)
        ngram_time = time.perf_counter() - t0

        if self.do_llm_rescoring:
            t1 = time.perf_counter()
            hypos = self.llm_rescore(hypos)
            llm_time = time.perf_counter() - t1
        else:
            llm_time = None

        for hypo in hypos:
            hypo['ngram_time'] = ngram_time
            hypo['llm_rescore_time'] = llm_time

        return hypos


    def process_logits(
        self,
        logits: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Preprocess raw logits for the beam search decoder.

        Applies (in order): temperature scaling, log-softmax, and blank penalty.
        Does NOT apply blank frame skipping (that is handled per-sample in
        ``ngram_decode``).

        Args:
            logits: CPU float32 contiguous tensor of shape ``(batch, time, num_tokens)``.

        Returns:
            Log-probability tensor of the same shape, with blank penalty applied.
        """
        if logits.dtype != torch.float32:
            raise ValueError("logits must be float32.")

        if not logits.is_cpu:
            raise RuntimeError("logits must be a CPU tensor.")

        if not logits.is_contiguous():
            raise RuntimeError("logits must be contiguous.")

        if logits.ndim != 3:
            raise RuntimeError(f"logits must be 3D. Found {logits.shape}")

        if logits.size(2) != self.token_dict.index_size():
            raise RuntimeError(f"Expected logits with {self.token_dict.index_size()} tokens in dim 2, but found {logits.size(2)}.")

        # Temperature scaling
        logits = logits / self.temperature

        # Log softmax
        logits = logits.log_softmax(dim=-1)

        # Blank penalty (1.0 = disabled since log(1) = 0; values <= 0 would crash)
        if self.blank_penalty > 0 and self.blank_penalty != 1.0:
            logits[:, :, self.blank_idx] -= math.log(self.blank_penalty)

        return logits


    def _extract_nbest(self, results, n_best: int, lm_weight: float) -> Dict:
        """Extract, deduplicate, and sort n-best hypotheses from raw decoder results.

        Args:
            results: Raw flashlight decode results (list of DecodeResult objects).
            n_best: Maximum number of hypotheses to keep.
            lm_weight: Weight on n-gram LM scores for combined scoring.

        Returns:
            Hypothesis dict with 'word_seqs', 'ngram_scores', 'acoustic_scores',
            and 'final_scores', all sorted best-first.
        """
        results = results[:n_best]

        word_seqs = []
        seen = set()
        ngram_scores = []
        acoustic_scores = []
        for result in results:
            word_seq = replace_words(
                remove_punctuation(
                    ' '.join([self.word_dict.get_entry(x) for x in result.words if x >= 0]).lower()
                )
            )
            if not word_seq or word_seq in seen:
                continue
            seen.add(word_seq)
            word_seqs.append(word_seq)
            ngram_scores.append(result.lmScore)
            acoustic_scores.append(result.emittingModelScore)

        if not word_seqs:
            return {
                'word_seqs': [''],
                'ngram_scores': [-math.inf],
                'acoustic_scores': [-math.inf],
                'final_scores': [-math.inf],
            }

        word_counts = np.array([len(s.split()) for s in word_seqs])
        combined_scores = (
            np.array(acoustic_scores)
            + (lm_weight * np.array(ngram_scores))
            + (self.word_score * word_counts)
        )
        sorted_indices = np.argsort(combined_scores)[::-1]

        return {
            'word_seqs': [word_seqs[i] for i in sorted_indices],
            'ngram_scores': [ngram_scores[i] for i in sorted_indices],
            'acoustic_scores': [acoustic_scores[i] for i in sorted_indices],
            'final_scores': [combined_scores[i] for i in sorted_indices],
        }


    def ngram_decode(
        self,
        logits: torch.FloatTensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> List[Dict]:
        """Run n-gram beam search decoding on preprocessed logits.

        Applies logit preprocessing (temperature, log-softmax, blank penalty),
        optionally skips high-confidence blank frames per sample, then runs
        flashlight's ``LexiconDecoder`` to produce an n-best list ranked by
        ``acoustic + lm_weight * ngram + word_score * num_words``.

        Args:
            logits: CPU float32 contiguous tensor of shape ``(batch, time, num_tokens)``
                containing raw (pre-softmax) logit outputs.
            lengths: Optional CPU int tensor of shape ``(batch,)`` with valid
                frame counts. If None, all frames are used.

        Returns:
            List of hypothesis dicts (one per batch item). Each dict contains
            'word_seqs', 'ngram_scores', 'acoustic_scores', and 'final_scores',
            all sorted best-first by the combined score.
        """

        logits = self.process_logits(logits)

        if lengths is not None and not lengths.is_cpu:
            raise RuntimeError("lengths must be a CPU tensor.")

        B, T, N = logits.size()
        if lengths is None:
            lengths = torch.full((B,), T)

        hypos = []
        do_blank_skip = self.blank_skip_threshold < 1.0
        if do_blank_skip:
            log_blank_penalty = math.log(self.blank_penalty)
            log_blank_threshold = math.log(self.blank_skip_threshold)

        for b in range(B):

            T_b = int(lengths[b].item())
            log_probs_b = logits[b, :T_b]

            # Optional: skip frames where blank probability exceeds threshold.
            # We temporarily undo the blank penalty before checking, so the threshold is
            # compared against the true model confidence, not the penalized value.
            # Comparison is done in log space to avoid exp().
            if do_blank_skip:
                keep_mask = (log_probs_b[:, self.blank_idx] + log_blank_penalty) < log_blank_threshold
                log_probs_b = log_probs_b[keep_mask]
                T_b = log_probs_b.size(0)

            # Ensure contiguous for pointer arithmetic
            log_probs_b = log_probs_b.contiguous()

            # Decode
            if T_b == 0:
                hypos.append({
                    'word_seqs': [''],
                    'ngram_scores': [-math.inf],
                    'acoustic_scores': [-math.inf],
                    'final_scores': [-math.inf],
                })
                continue

            results = self.ngram_decoder.decode(log_probs_b.data_ptr(), T_b, N)
            hypos.append(self._extract_nbest(results, self.n_best, self.lm_weight))

        return hypos


    def llm_rescore(
        self,
        hypotheses: List[Dict],
    ) -> List[Dict]:
        """Rescore n-best hypotheses with the LLM and re-rank.

        Applies the scoring formula::

            final = acoustic + lm_weight * ngram + llm_alpha * llm + word_score * num_words

        Args:
            hypotheses: List of hypothesis dicts as returned by ``ngram_decode``.

        Returns:
            List of hypothesis dicts with updated 'final_scores' reflecting the
            LLM-interpolated ranking, re-sorted best-first.
        """

        # Pool all hypotheses across batch items for packed LLM batching
        all_seqs = []
        offsets = []  # (start, end) per hypo, or None if empty
        for hypo in hypotheses:
            if hypo['word_seqs']:
                start = len(all_seqs)
                all_seqs.extend(hypo['word_seqs'])
                offsets.append((start, len(all_seqs)))
            else:
                offsets.append(None)

        # Score all hypotheses in packed batches
        all_llm_scores = []
        for i in range(0, len(all_seqs), self.llm_batch_size):
            batch_scores = _get_llm_scores(
                self.llm_model,
                self.llm_tokenizer,
                all_seqs[i:i+self.llm_batch_size],
                length_penalty=self.llm_length_penalty,
            )
            all_llm_scores.extend(batch_scores)

        # Redistribute scores and rescore per batch item
        rescored_hypos = []
        for hypo, offset in zip(hypotheses, offsets):
            if offset is None:
                rescored_hypos.append(hypo)
                continue

            word_seqs = np.array(hypo['word_seqs'])
            word_counts = np.array([len(s.split()) for s in word_seqs])
            acoustic = np.array(hypo['acoustic_scores'])
            ngram = np.array(hypo['ngram_scores'])
            llm = np.array(all_llm_scores[offset[0]:offset[1]])

            # rescoring
            final = acoustic + (self.lm_weight * ngram) + (self.llm_alpha * llm) + (self.word_score * word_counts)

            # re-sort by final score (higher is better)
            order = np.argsort(final)[::-1]
            rescored_hypos.append({
                'word_seqs': word_seqs[order].tolist(),
                'ngram_scores': ngram[order].tolist(),
                'acoustic_scores': acoustic[order].tolist(),
                'llm_scores': llm[order].tolist(),
                'final_scores': final[order].tolist(),
            })

        return rescored_hypos


    # =========================================================================
    # Online / streaming decoding
    # =========================================================================

    def online_decode_begin(self):
        """Reset streaming state and start a new utterance.

        Initializes the flashlight decoder's internal beam state for
        incremental frame-by-frame decoding. Call this before the first
        ``online_decode_step()``.

        If called while a previous utterance is still active, the old
        state is abandoned and a fresh decode session begins.
        """
        self.ngram_decoder.decode_begin()
        self._online_total_frames = 0
        self._online_active = True


    def online_decode_step(
        self,
        logits: torch.FloatTensor,
    ) -> Dict:
        """Feed new logit frame(s) and return the current best decoded sentence.

        Uses flashlight's streaming ``decode_step`` + ``get_best_hypothesis``
        to incrementally extend the beam search with new frames.

        Args:
            logits: Raw (pre-softmax) logits for one or more new frames.
                Accepted shapes:

                - ``(T, num_tokens)``: one or more frames, unbatched
                - ``(1, T, num_tokens)``: single-item batch (batch dim squeezed)

                Must be CPU float32. ``T > 1`` is useful for catching up if
                the caller falls behind real-time.

        Returns:
            Dict with keys:

            - ``'word_seq'`` (str): current best decoded sentence (lowercased,
              space-separated words). Empty string if no words decoded yet.
            - ``'score'`` (float): combined hypothesis score.
            - ``'ngram_score'`` (float): raw KenLM log-prob sum.
            - ``'acoustic_score'`` (float): raw acoustic log-prob sum.
            - ``'total_frames'`` (int): total non-blank frames decoded so far.
        """
        # Auto-begin if needed
        if not self._online_active:
            self.online_decode_begin()

        # --- Input normalization ---
        if logits.ndim == 2:
            logits = logits.unsqueeze(0)  # (T, N) -> (1, T, N)
        if logits.ndim != 3 or logits.size(0) != 1:
            raise RuntimeError(
                f"online_decode_step expects shape (T, N) or (1, T, N), got {logits.shape}"
            )
        if not logits.is_contiguous():
            logits = logits.contiguous()

        # --- Preprocess: temperature, log-softmax, blank penalty ---
        log_probs = self.process_logits(logits)[0]  # (T, N)

        # --- Blank frame skipping ---
        if self.blank_skip_threshold < 1.0:
            log_blank_penalty = math.log(self.blank_penalty)
            log_blank_threshold = math.log(self.blank_skip_threshold)
            keep_mask = (log_probs[:, self.blank_idx] + log_blank_penalty) < log_blank_threshold
            log_probs = log_probs[keep_mask]

        # --- Feed surviving frames to the decoder ---
        N = log_probs.size(1)
        n_new = log_probs.size(0)
        for t in range(n_new):
            frame = log_probs[t].contiguous()
            self.ngram_decoder.decode_step(frame.data_ptr(), 1, N)
        self._online_total_frames += n_new

        # --- Get current best ---
        if self._online_total_frames == 0:
            return {
                'word_seq': '',
                'score': 0.0,
                'ngram_score': 0.0,
                'acoustic_score': 0.0,
                'total_frames': 0,
            }

        result = self.ngram_decoder.get_best_hypothesis()
        word_seq = replace_words(
            remove_punctuation(
                ' '.join(
                    [self.word_dict.get_entry(x) for x in result.words if x >= 0]
                ).lower()
            )
        )

        return {
            'word_seq': word_seq,
            'score': result.score,
            'ngram_score': result.lmScore,
            'acoustic_score': result.emittingModelScore,
            'total_frames': self._online_total_frames,
        }


    def online_decode_end(self) -> Dict:
        """Finalize online decoding and return the full n-best hypothesis list.

        Calls flashlight's ``decode_end`` to apply end-of-sentence LM scores,
        then retrieves the full n-best list and optionally rescores with the
        LLM. Resets streaming state so the decoder is ready for the next
        utterance (or offline use).

        Returns:
            Hypothesis dict (same format as offline ``offline_decode()`` output) with
            keys ``'word_seqs'``, ``'ngram_scores'``, ``'acoustic_scores'``,
            ``'final_scores'``, ``'ngram_time'``, and ``'llm_rescore_time'``.
            If no frames were decoded, all lists are empty.
        """
        if not self._online_active or self._online_total_frames == 0:
            self._online_active = False
            self._online_total_frames = 0
            return {
                'word_seqs': [''],
                'ngram_scores': [-math.inf],
                'acoustic_scores': [-math.inf],
                'final_scores': [-math.inf],
                'ngram_time': None,
                'llm_rescore_time': None,
            }

        # Finalize and get n-best
        t0 = time.perf_counter()
        self.ngram_decoder.decode_end()
        results = self.ngram_decoder.get_all_final_hypothesis()
        hypo = self._extract_nbest(results, self.n_best, self.lm_weight)
        ngram_time = time.perf_counter() - t0

        # Optional LLM rescoring
        if self.do_llm_rescoring and hypo['word_seqs']:
            t1 = time.perf_counter()
            [hypo] = self.llm_rescore([hypo])
            llm_time = time.perf_counter() - t1
        else:
            llm_time = None

        hypo['ngram_time'] = ngram_time
        hypo['llm_rescore_time'] = llm_time

        # Reset state
        self._online_active = False
        self._online_total_frames = 0

        return hypo


def _construct_trie(tokens_dict, word_dict, lexicon, lm, silence):
    """Build a flashlight Trie from the lexicon, seeded with unigram LM scores."""
    vocab_size = tokens_dict.index_size()
    trie = Trie(vocab_size, silence)
    start_state = lm.start(False)

    for word, spellings in lexicon.items():
        word_idx = word_dict.get_index(word)
        _, score = lm.score(start_state, word_idx)
        for spelling in spellings:
            spelling_idx = [tokens_dict.get_index(token) for token in spelling]
            trie.insert(spelling_idx, word_idx, score)
    trie.smear(SmearingMode.MAX)
    return trie


def _build_llm(
    model_name: str,
    cache_dir: Optional[str] = None,
    device: str = 'cuda:0',
    dtype: Literal['float16', 'bfloat16'] = 'bfloat16',
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

    torch_dtype = torch.float16 if dtype == 'float16' else torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        dtype=torch_dtype,
        attn_implementation='sdpa',
    )

    if lora_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()

    model.to(device).eval()

    return model, tokenizer


def _get_llm_scores(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    hypotheses: list[str],
    length_penalty: float = 0.0,
) -> list[float]:
    """Score candidate hypotheses by summing shifted per-token log-probs.

    Each hypothesis is tokenized, fed through the model, and scored as
    the sum of log-probabilities for each token given its left context.
    An attention mask ensures that padding tokens do not contribute to
    the score.

    Args:
        model: A causal LM returned by :func:`_build_llm`.
        tokenizer: The matching tokenizer returned by :func:`_build_llm`.
        hypotheses: Candidate sentence strings to score.
        length_penalty: If non-zero, subtract ``length_penalty * n_tokens``
            from each score to penalise longer sequences.

    Returns:
        A list of float log-probability scores, one per hypothesis.
    """
    if not hypotheses:
        return []

    inputs = tokenizer(hypotheses, return_tensors='pt', padding=True)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)

    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)

        # Shift: logits at position t predict token at position t+1
        token_log_probs = log_probs[:, :-1, :].gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

        # Masked sum over real (non-padding) tokens
        scores = (token_log_probs * attention_mask[:, 1:]).sum(dim=-1)

        if length_penalty != 0.0:
            scores = scores - length_penalty * attention_mask.sum(dim=-1)

    return scores.tolist()
