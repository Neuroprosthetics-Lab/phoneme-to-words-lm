"""Versioned causal-LM scoring shared by decoding and training utilities.

The exact text is tokenized jointly. A token crossing the context/candidate
character boundary belongs to the candidate. Padding and conditioning tokens
never contribute to the sum or length penalty. This module has no model loader.
"""
from dataclasses import dataclass
import math
from numbers import Integral

import torch

SCORER_VERSION = 'sentence_v1'
DEFAULT_PREFIX = '\n'


@dataclass(frozen=True)
class ScoringInput:
    input_ids: list[int]
    target_mask: list[bool]

    @property
    def token_count(self):
        return sum(self.target_mask[1:])


def positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f'{name} must be a positive integer, got {value!r}')


def validate_contexts(contexts, size):
    if contexts is None:
        return [''] * size
    if isinstance(contexts, str) or len(contexts) != size:
        raise ValueError(f'contexts must contain exactly {size} strings')
    if any(not isinstance(c, str) for c in contexts):
        raise TypeError('each context must be a string')
    return list(contexts)


def prepare_scoring_inputs(tokenizer, hypotheses, contexts=None, *,
                           prefix=DEFAULT_PREFIX, score_eos=False,
                           max_length=512):
    """Tokenize without truncating targets; fail explicitly on overlength input.

    Text is ``prefix + context + optional newline + hypothesis``. The context
    is preserved verbatim and gets a trailing newline if it has none. No chat
    template or invented BOS ID is used. EOS scoring is an explicit option.
    """
    positive_int('max_length', max_length)
    if not isinstance(prefix, str) or not prefix:
        raise ValueError('prefix must be a nonempty textual conditioning prefix')
    if not isinstance(score_eos, bool):
        raise ValueError('score_eos must be bool')
    if isinstance(hypotheses, str) or any(not isinstance(h, str) for h in hypotheses):
        raise TypeError('hypotheses must be a sequence of strings')
    contexts = validate_contexts(contexts, len(hypotheses))
    if score_eos and tokenizer.eos_token_id is None:
        raise ValueError('score_eos requires tokenizer.eos_token_id')
    prepared = []
    for i, (hypothesis, context) in enumerate(zip(hypotheses, contexts)):
        conditioning = prefix + context
        if context and not context.endswith('\n'):
            conditioning += '\n'
        text = conditioning + hypothesis
        try:
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        except (NotImplementedError, TypeError) as exc:
            raise ValueError('sentence_v1 scoring requires a tokenizer with offset mappings (use a fast tokenizer)') from exc
        ids = list(encoded['input_ids'])
        mask = [end > len(conditioning) and end > start
                for start, end in encoded['offset_mapping']]
        if score_eos and hypothesis:
            ids.append(tokenizer.eos_token_id)
            mask.append(True)
        if not ids or mask[0]:
            raise ValueError('prefix must tokenize to at least one conditioning token before the first target; choose a longer prefix')
        if len(ids) > max_length:
            raise ValueError(f'LLM input {i} has {len(ids)} tokens, exceeding max_length={max_length}; shorten context or raise the limit (hypotheses are never silently truncated)')
        prepared.append(ScoringInput(ids, mask))
    return prepared


def padded_inputs(requests, tokenizer, device):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError('tokenizer needs a pad_token_id or eos_token_id')
    width = max(len(r.input_ids) for r in requests)
    ids = torch.tensor([r.input_ids + [pad_id] * (width-len(r.input_ids)) for r in requests], device=device)
    attention = torch.tensor([[1] * len(r.input_ids) + [0] * (width-len(r.input_ids)) for r in requests], device=device)
    targets = torch.tensor([r.target_mask + [False] * (width-len(r.input_ids)) for r in requests], dtype=torch.bool, device=device)
    return ids, attention, targets


def _score_batch(model, tokenizer, requests, logprob_chunk_size):
    """Keep BF16/model logits; materialize only a bounded set of FP32 rows."""
    with torch.inference_mode():
        ids, attention, targets = padded_inputs(requests, tokenizer, model.device)
        output = model(input_ids=ids, attention_mask=attention, use_cache=False)
        logits = output.logits
        rows, positions = (targets[:, 1:] & attention[:, 1:].bool()).nonzero(as_tuple=True)
        token_scores = torch.zeros((len(requests), ids.shape[1]-1), dtype=torch.float32, device=model.device)
        for start in range(0, rows.numel(), logprob_chunk_size):
            batch_rows = rows[start:start + logprob_chunk_size]
            token_positions = positions[start:start + logprob_chunk_size]
            chunk_logits = logits[batch_rows, token_positions].float()
            # Each position predicts the following token.
            target_ids = ids[batch_rows, token_positions + 1, None]
            target_logits = chunk_logits.gather(-1, target_ids).squeeze(-1)
            values = target_logits - torch.logsumexp(chunk_logits, dim=-1)
            token_scores[batch_rows, token_positions] = values
        values = token_scores.sum(dim=-1).tolist()
    if any(not math.isfinite(v) for v in values):
        raise RuntimeError('LLM returned a nonfinite target log-probability')
    return values


def score_prepared(model, tokenizer, requests, *, batch_size=100,
                   max_batch_tokens=2048, logprob_chunk_size=16):
    """Length-bucket requests, cap padded token volume, retry OOM by splitting.

    Returns raw sums in original request order. A singleton OOM is actionable;
    no score is substituted and retry depth is bounded by batch size.
    """
    for name, value in [('batch_size', batch_size), ('max_batch_tokens', max_batch_tokens),
                        ('logprob_chunk_size', logprob_chunk_size)]:
        positive_int(name, value)
    scores = [0.0] * len(requests)
    # No-target examples need no model pass.
    order = sorted((i for i, r in enumerate(requests) if r.token_count),
                   key=lambda i: len(requests[i].input_ids))
    for i in order:
        if len(requests[i].input_ids) > max_batch_tokens:
            raise ValueError(f'LLM input {i} exceeds max_batch_tokens={max_batch_tokens} even at batch size 1')

    def infer(indices):
        # Leave the except block before retrying, releasing failed tensors and
        # traceback references before empty_cache and recursive allocation.
        failed = False
        try:
            values = _score_batch(model, tokenizer, [requests[i] for i in indices], logprob_chunk_size)
        except torch.OutOfMemoryError:
            if len(indices) == 1:
                raise RuntimeError('A single LLM scoring request does not fit in memory; shorten context, lower max length, or use a smaller model') from None
            failed = True
        if failed:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            split = len(indices) // 2
            infer(indices[:split])
            infer(indices[split:])
        else:
            for i, value in zip(indices, values):
                scores[i] = value

    batch = []
    for i in order:
        if batch and (len(batch) >= batch_size or (len(batch)+1)*len(requests[i].input_ids) > max_batch_tokens):
            infer(batch)
            batch = []
        batch.append(i)
    if batch:
        infer(batch)
    return scores


def score_sentences(model, tokenizer, hypotheses, contexts=None, *, prefix=DEFAULT_PREFIX,
                    score_eos=False, max_length=512, batch_size=100,
                    max_batch_tokens=2048, logprob_chunk_size=16):
    requests = prepare_scoring_inputs(tokenizer, hypotheses, contexts,
                                      prefix=prefix, score_eos=score_eos, max_length=max_length)
    sums = score_prepared(model, tokenizer, requests, batch_size=batch_size,
                          max_batch_tokens=max_batch_tokens, logprob_chunk_size=logprob_chunk_size)
    return sums, [r.token_count for r in requests]


def training_example(request):
    """Pretokenized trainer example with exactly the scorer's target labels."""
    return dict(input_ids=request.input_ids, attention_mask=[1]*len(request.input_ids),
                labels=[token if target else -100 for token, target in zip(request.input_ids, request.target_mask)])


@dataclass
class ScoringDataCollator:
    """Right-pad prepared examples without masking real EOS tokens by identity."""
    pad_token_id: int

    def __call__(self, examples):
        width = max(len(e['input_ids']) for e in examples)
        return {
            key: torch.tensor([list(e[key])+[fill]*(width-len(e[key])) for e in examples])
            for key, fill in [('input_ids', self.pad_token_id), ('attention_mask', 0), ('labels', -100)]
        }


def sentence_perplexity(model, tokenizer, sentences, *, batch_size=16,
                        max_length=512, prefix=DEFAULT_PREFIX, score_eos=False):
    """Aggregate NLL over exact target counts; preserve the model's train mode."""
    was_training = model.training
    model.eval()
    try:
        raw, counts = score_sentences(model, tokenizer, sentences, prefix=prefix,
                                      score_eos=score_eos, max_length=max_length,
                                      batch_size=batch_size)
    finally:
        model.train(was_training)
    count = sum(counts)
    if not count:
        return math.inf
    try:
        return math.exp(-math.fsum(raw)/count)
    except OverflowError:
        return math.inf
