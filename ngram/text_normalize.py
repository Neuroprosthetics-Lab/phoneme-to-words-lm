"""Text normalization pipeline for n-gram LM corpus preprocessing.

Produces clean lowercase text with no punctuation, no special characters,
and all numbers converted to words. Handles sentence splitting so each
output line is a single sentence.

This is an offline preprocessing pipeline for training corpora — it is
intentionally more aggressive than the runtime normalization in
general_utils.py (which is used by real-time BCI nodes).
"""

from __future__ import annotations

import functools
import html
import logging
import multiprocessing
import os
import re
import string
import sys
import unicodedata

from num2words import num2words
from tqdm import tqdm

from phoneme_to_words_lm.utils import replace_words

log = logging.getLogger(__name__)

# Try to load nltk sentence tokenizer at import time.
try:
    from nltk.tokenize import sent_tokenize as _nltk_sent_tokenize
    # Verify the punkt_tab data is available.
    _nltk_sent_tokenize("Test sentence.")
    sent_tokenize = _nltk_sent_tokenize
except LookupError:
    import nltk
    nltk.download('punkt_tab', quiet=True)
    from nltk.tokenize import sent_tokenize


# ---------------------------------------------------------------------------
# Unicode punctuation normalization
# ---------------------------------------------------------------------------

# Mapping of Unicode punctuation to ASCII equivalents.
_UNICODE_PUNCT_MAP = str.maketrans({
    # Smart/curly quotes -> straight
    '\u2018': "'",   # left single
    '\u2019': "'",   # right single
    '\u201A': "'",   # single low-9
    '\u201B': "'",   # single high-reversed-9
    '\u201C': '"',   # left double
    '\u201D': '"',   # right double
    '\u201E': '"',   # double low-9
    '\u201F': '"',   # double high-reversed-9
    '\u2039': "'",   # single left-pointing angle
    '\u203A': "'",   # single right-pointing angle
    '\u00AB': '"',   # left-pointing double angle (guillemet)
    '\u00BB': '"',   # right-pointing double angle (guillemet)

    # Dashes -> space (word boundaries)
    '\u2013': ' ',   # en dash
    '\u2014': ' ',   # em dash
    '\u2015': ' ',   # horizontal bar

    # Ellipsis -> period (for sentence splitting)
    '\u2026': '.',

    # Other common Unicode punctuation
    '\u2032': "'",   # prime
    '\u2033': '"',   # double prime
    '\u00B7': ' ',   # middle dot
    '\u2022': ' ',   # bullet
    '\u2010': '-',   # hyphen
    '\u2011': '-',   # non-breaking hyphen
    '\u2012': '-',   # figure dash
})


def normalize_unicode_punctuation(text: str) -> str:
    """Replace Unicode punctuation with ASCII equivalents."""
    return text.translate(_UNICODE_PUNCT_MAP)


# ---------------------------------------------------------------------------
# URL and email removal
# ---------------------------------------------------------------------------

_URL_RE = re.compile(
    r'https?://\S+'        # http:// or https://
    r'|www\.\S+'           # www.something
    r'|ftp://\S+',        # ftp://
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r'\S+@\S+\.\S+')


def remove_urls_and_emails(text: str) -> str:
    """Strip URLs and email addresses from text."""
    text = _URL_RE.sub('', text)
    text = _EMAIL_RE.sub('', text)
    return text


# ---------------------------------------------------------------------------
# Ordinal and decade expansion
# ---------------------------------------------------------------------------

_ORDINAL_RE = re.compile(r'(\d+)(st|nd|rd|th)\b', re.IGNORECASE)

_DECADE_RE = re.compile(r'\b(\d{4})s\b')


# num2words is expensive; cache results per-process keyed on the raw matched
# substring. A few hundred distinct numeric tokens cover the working set of
# any realistic corpus, so a 200k cap is effectively unbounded in practice.
@functools.lru_cache(maxsize=200_000)
def _ordinal_words_cached(digits: str) -> str | None:
    try:
        return num2words(int(digits), to='ordinal')
    except Exception:
        return None


@functools.lru_cache(maxsize=200_000)
def _decade_words_cached(digits: str) -> str | None:
    try:
        parts = num2words(int(digits), to='year').split()
        last = parts[-1]
        parts[-1] = last[:-1] + 'ies' if last.endswith('y') else last + 's'
        return ' '.join(parts)
    except Exception:
        return None


def expand_ordinals(text: str) -> str:
    """Expand ordinal numbers: '1st' -> 'first', '23rd' -> 'twenty-third'."""
    def _replace(m):
        words = _ordinal_words_cached(m.group(1))
        return words if words is not None else m.group()
    return _ORDINAL_RE.sub(_replace, text)


def expand_decades(text: str) -> str:
    """Expand decades: '1990s' -> 'nineteen nineties', '2020s' -> 'twenty twenties'."""
    def _replace(m):
        words = _decade_words_cached(m.group(1))
        return words if words is not None else m.group()
    return _DECADE_RE.sub(_replace, text)


# ---------------------------------------------------------------------------
# Number-to-words conversion
# ---------------------------------------------------------------------------

# Standalone numbers: integers, decimals, with optional commas.
# Negative lookbehind/ahead prevents matching digits embedded in words.
_NUMBER_RE = re.compile(r'(?<![a-zA-Z\d])-?\d[\d,]*\.?\d*(?![a-zA-Z\d])')


@functools.lru_cache(maxsize=200_000)
def _number_words_cached(raw: str) -> str | None:
    s = raw.replace(',', '')
    try:
        if '.' in s:
            words = num2words(float(s))
        else:
            n = int(s)
            if 1000 <= n <= 2099 and len(s) == 4:
                words = num2words(n, to='year')
            else:
                words = num2words(n)
        # num2words produces hyphens ("forty-two") and commas
        # ("one thousand, two hundred") — clean them out.
        return words.replace('-', ' ').replace(',', '')
    except Exception:
        return None


def numbers_to_words(text: str) -> str:
    """Replace standalone numeric sequences with their word equivalents.

    Four-digit integers in 1000-2099 are read as years ("1992" -> "nineteen
    ninety two") rather than cardinal numbers. Post-processes num2words
    output to remove hyphens and commas so that e.g. "forty-two" becomes
    "forty two".
    """
    def _replace(match):
        words = _number_words_cached(match.group())
        return words if words is not None else match.group()
    return _NUMBER_RE.sub(_replace, text)


# ---------------------------------------------------------------------------
# Unicode normalization (strip accents)
# ---------------------------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """NFKD-normalize text and drop anything outside ASCII.

    'cafe\\u0301' -> 'cafe', 'nai\\u0308ve' -> 'naive'. Combining marks
    split off by NFKD (accents/diacritics) are non-ASCII and get dropped.
    Non-Latin scripts (Greek, CJK, ...) and atomic non-decomposable chars
    like 'ß' are also dropped here — they would be stripped one step later
    by `remove_punctuation`'s `[a-zA-Z '\\s]` filter anyway, so removing
    them up front is equivalent and ~14× faster than the pure-Python
    `unicodedata.category`-per-char scan this replaced.
    """
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')


# ---------------------------------------------------------------------------
# Punctuation removal
# ---------------------------------------------------------------------------

# Single-pass translate table that replaces the previous three-step
# "punct-to-space + strip-non-allowed + lowercase" pipeline. Input at this
# stage is ASCII-only (normalize_unicode has already stripped everything
# else), so we can enumerate all 128 code points:
#   - A-Z         -> lowercase
#   - -/.|[]      -> space (so 'well-known' -> 'well known', '[[link|thumb]]'
#                   -> 'link thumb', 'Dr.Sara' -> 'dr sara', '1/2' -> '1 2')
#   - a-z, ', whitespace -> keep
#   - everything else -> delete
def _build_remove_punct_table() -> dict[int, str | None]:
    keep_whitespace = set(' \t\n\r\f\v')
    keep_letters = set(string.ascii_lowercase)
    punct_to_space = set('-/.|[]')
    table: dict[int, str | None] = {}
    for i in range(128):
        c = chr(i)
        if 'A' <= c <= 'Z':
            table[i] = c.lower()
        elif c in punct_to_space:
            table[i] = ' '
        elif c in keep_letters or c == "'" or c in keep_whitespace:
            continue
        else:
            table[i] = None
    return table


_REMOVE_PUNCT_TABLE = _build_remove_punct_table()

# An apostrophe run without a letter on at least one side is a quote-like
# artifact ('Beatles', ''economic, india'', ' m); one with letters on both
# sides is a contraction (don't, o'clock) and must survive. Folds the
# previous "'{2,} -> '" + boundary-strip regexes into one pass.
_APOSTROPHE_RE = re.compile(r"(?<![a-zA-Z])'+|'+(?![a-zA-Z])")
_WHITESPACE_RE = re.compile(r'\s+')


def remove_punctuation(text: str) -> str:
    """Remove punctuation, lowercase, and collapse whitespace.

    Replaces hyphens, slashes, periods, pipes, and brackets with spaces
    (so 'well-known' -> 'well known', 'Dr.Sara' -> 'dr sara', '1/2'
    -> '1 2', '[[link|thumb]]' -> 'link thumb'); keeps only lowercase
    letters, apostrophes, and spaces.

    Apostrophes are kept inside words to preserve contractions ("don't",
    "it's", "o'clock"), but word-boundary apostrophes -- which come from
    quote marks wrapping words, e.g. "'Beatles'" or 'He said 'don\\'t\\''
    -- are stripped. Runs of adjacent apostrophes are also collapsed so
    stray doubled-quote artifacts ("''economic", "india''") don't survive.
    """
    text = text.translate(_REMOVE_PUNCT_TABLE)
    text = _APOSTROPHE_RE.sub('', text)
    text = _WHITESPACE_RE.sub(' ', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

_DIGIT_LETTER_BOUNDARY_RE = re.compile(r'(?<=\d)(?=[a-zA-Z])|(?<=[a-zA-Z])(?=\d)')
_DIGIT_RE = re.compile(r'\d')


def normalize_line(sentence: str) -> str:
    """Apply per-sentence normalization (steps 5-11 of the pipeline).

    Expects a single sentence (already split). Returns cleaned text
    or empty string if nothing remains.
    """
    # Skip digit-oriented stages entirely when the sentence has no digits.
    # Each of those stages short-circuits on `re.sub` with no match, but
    # the regex scan still costs O(len) per stage — a single `\d` search
    # up front lets us skip four of them on digit-free lines.
    if _DIGIT_RE.search(sentence):
        sentence = expand_ordinals(sentence)
        sentence = expand_decades(sentence)
        # Split digits from adjacent letters so num2words can match dosages
        # and units ("100mg" -> "100 mg" -> "one hundred mg", "L4" -> "L 4").
        sentence = _DIGIT_LETTER_BOUNDARY_RE.sub(' ', sentence)
        sentence = numbers_to_words(sentence)
    sentence = normalize_unicode(sentence)
    sentence = remove_punctuation(sentence)
    sentence = replace_words(sentence)
    return sentence.strip()


_HONORIFIC_RE = re.compile(
    r'\b(Mr|Mrs|Ms|Dr|Jr|Sr|Lt|St|Capt|Sgt)\s+\.',
    flags=re.IGNORECASE,
)
# Orphan apostrophe left by spaced smart-quote contractions ("I ' m" -> "I'm").
# Only collapse when the right side is an actual English contraction suffix
# (s, t, m, d, ll, re, ve, em, n) so we do NOT merge actual single-quoted
# speech like "He said ' hello ' to me".
_ORPHAN_APOSTROPHE_RE = re.compile(
    r"(?<=[a-zA-Z])\s+'\s+(s|t|m|d|ll|re|ve|em|n)\b",
    flags=re.IGNORECASE,
)


def normalize_lines(text: str) -> list[str]:
    """Apply the full normalization pipeline to a raw text line.

    Steps 1-4 (pre-processing and sentence splitting) produce multiple
    sentences, then steps 5-11 are applied to each. Returns a list of
    normalized sentences (empty sentences are dropped).
    """
    text = html.unescape(text)
    text = normalize_unicode_punctuation(text)
    # Ampersand -> " and " so it doesn't silently merge surrounding tokens
    # ("B&C" -> "b and c", "T&Cs" -> "t and cs").
    text = text.replace('&', ' and ')
    # Collapse spaced honorifics so punkt treats "Mr ." as "Mr." (an
    # abbreviation), not as a sentence boundary.
    text = _HONORIFIC_RE.sub(r'\1.', text)
    # Re-join orphan apostrophes from spaced smart-quote contractions.
    text = _ORPHAN_APOSTROPHE_RE.sub(r"'\1", text)
    text = remove_urls_and_emails(text)

    # Collapse ellipsis-like sequences before sentence splitting.
    text = re.sub(r'\.{2,}', '.', text)

    sentences = sent_tokenize(text)

    results = []
    for sentence in sentences:
        normalized = normalize_line(sentence)
        if normalized:
            results.append(normalized)
    return results


# ---------------------------------------------------------------------------
# Parallel workers
# ---------------------------------------------------------------------------

# Chunks are lists of raw bytes (one entry per input line) so the parent can
# ship them to workers without decoding, and the parent's hot loop is just
# "read bytes, hand to pool". Workers handle decode + normalize_lines + local
# accumulation, then return per-chunk results. We also return the chunk's
# total byte length so the parent can advance the tqdm pbar.

def _pool_init() -> None:
    """Pre-warm nltk's Punkt tokenizer in each worker.

    sent_tokenize lazily loads a few-MB pickle on first call; doing it in the
    initializer avoids paying that cost on the first real chunk.
    """
    try:
        sent_tokenize('Warmup sentence.')
    except Exception:
        pass


def _normalize_chunk_word(chunk: list[bytes]) -> tuple[list[str], set[str], int]:
    sentences_out: list[str] = []
    vocab: set[str] = set()
    nbytes = 0
    for raw in chunk:
        nbytes += len(raw)
        line = raw.decode('utf-8', errors='replace')
        for sentence in normalize_lines(line):
            sentences_out.append(sentence)
            vocab.update(sentence.split())
    return sentences_out, vocab, nbytes


def _normalize_chunk_spelling(chunk: list[bytes]) -> tuple[set[str], int]:
    seen: set[str] = set()
    nbytes = 0
    for raw in chunk:
        nbytes += len(raw)
        line = raw.decode('utf-8', errors='replace')
        for sentence in normalize_lines(line):
            for word in sentence.split():
                seen.add(' '.join(word))
    return seen, nbytes


def _iter_raw_chunks(fin, chunk_lines: int):
    chunk: list[bytes] = []
    for raw in fin:
        chunk.append(raw)
        if len(chunk) >= chunk_lines:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def normalize_corpus(
    input_path: str,
    output_path: str,
    lm_type: str,
    workers: int = 1,
    chunk_lines: int = 1000,
) -> set[str]:
    """Normalize a raw text corpus and write it to *output_path*.

    Args:
        input_path: Path to the raw text corpus (one sentence per line).
        output_path: Where to write the normalized text.
        lm_type: "word" or "spelling".
        workers: Number of worker processes. ``1`` (or less) runs serially
            with deterministic output order — useful for debugging and
            regression diffs. ``>1`` uses ``multiprocessing.Pool.imap_unordered``
            across chunks of ``chunk_lines`` input lines; output order is
            not preserved (KenLM's ``lmplz`` doesn't care about line order
            since it counts n-grams).
        chunk_lines: Number of raw input lines per parallel chunk. Default
            1000 balances IPC overhead vs. work granularity for both short
            dialogue corpora and long wiki-style articles.

    Returns:
        For word mode: set of all unique words found in the normalized corpus.
        For spelling mode: empty set (vocab comes from the letter lexicon).
    """
    vocab: set[str] = set()
    n_out = 0

    total_bytes = os.path.getsize(input_path)
    tqdm_kwargs = dict(
        total=total_bytes,
        unit='B',
        unit_scale=True,
        desc=f'  Normalizing {os.path.basename(input_path)}',
        disable=not sys.stderr.isatty(),
    )

    parallel = workers > 1

    if lm_type == 'spelling':
        seen: set[str] = set()
        with open(input_path, 'rb') as fin, tqdm(**tqdm_kwargs) as pbar:
            chunks = _iter_raw_chunks(fin, chunk_lines)
            if parallel:
                with multiprocessing.Pool(workers, initializer=_pool_init) as pool:
                    for local_seen, nbytes in pool.imap_unordered(
                            _normalize_chunk_spelling, chunks, chunksize=1):
                        pbar.update(nbytes)
                        seen.update(local_seen)
            else:
                for chunk in chunks:
                    local_seen, nbytes = _normalize_chunk_spelling(chunk)
                    pbar.update(nbytes)
                    seen.update(local_seen)

        sorted_entries = sorted(seen)
        with open(output_path, 'w') as fout:
            for entry in sorted_entries:
                fout.write(entry + '\n')
        n_out = len(sorted_entries)

    else:
        with open(input_path, 'rb') as fin, open(output_path, 'w') as fout, \
                tqdm(**tqdm_kwargs) as pbar:
            chunks = _iter_raw_chunks(fin, chunk_lines)
            if parallel:
                with multiprocessing.Pool(workers, initializer=_pool_init) as pool:
                    for sentences, local_vocab, nbytes in pool.imap_unordered(
                            _normalize_chunk_word, chunks, chunksize=1):
                        pbar.update(nbytes)
                        fout.write('\n'.join(sentences))
                        if sentences:
                            fout.write('\n')
                        n_out += len(sentences)
                        vocab.update(local_vocab)
            else:
                for chunk in chunks:
                    sentences, local_vocab, nbytes = _normalize_chunk_word(chunk)
                    pbar.update(nbytes)
                    for sentence in sentences:
                        fout.write(sentence + '\n')
                    n_out += len(sentences)
                    vocab.update(local_vocab)

    log.info(f'  Normalized {input_path}: {n_out} lines out'
             + (f', {len(vocab)} unique words' if lm_type == 'word' else
                f', {n_out} unique character-expanded words')
             + (f' (workers={workers})' if parallel else ''))
    return vocab


def collect_vocab_from_normalized(normalized_path: str) -> set[str]:
    """Scan a pre-normalized text file and return the set of unique words."""
    vocab: set[str] = set()
    total_bytes = os.path.getsize(normalized_path)
    with open(normalized_path, 'rb') as f, tqdm(
        total=total_bytes,
        unit='B',
        unit_scale=True,
        desc=f'  Scanning vocab from {os.path.basename(normalized_path)}',
        disable=not sys.stderr.isatty(),
    ) as pbar:
        for raw in f:
            pbar.update(len(raw))
            vocab.update(raw.decode('utf-8', errors='replace').strip().split())
    return vocab
