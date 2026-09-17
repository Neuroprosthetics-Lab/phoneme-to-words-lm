"""Lightweight public API; load the native/model stack only when requested."""
from phoneme_to_words_lm.utils import (
    remove_punctuation,
    replace_words,
    phonemize_sentence,
    PHONE_DEF_SIL,
    PHONE_DEF_SIL_BLANK,
    LOGIT_PHONE_DEF,
    SIL_DEF,
    HF_CACHE_DIR,
)


def __getattr__(name):
    if name == 'KenLMFlashlightTextLM':
        from .decoder import KenLMFlashlightTextLM
        globals()[name] = KenLMFlashlightTextLM
        return KenLMFlashlightTextLM
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


__all__ = ['KenLMFlashlightTextLM', 'remove_punctuation', 'replace_words',
           'phonemize_sentence', 'PHONE_DEF_SIL', 'PHONE_DEF_SIL_BLANK',
           'LOGIT_PHONE_DEF', 'SIL_DEF', 'HF_CACHE_DIR']
