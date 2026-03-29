#!/usr/bin/env python3
"""Demo: create fake phoneme logits from text and decode them with KenLMFlashlightTextLM.

This script phonemizes a sentence, creates synthetic logit frames with the
correct phoneme at each time step (plus Gaussian noise), and runs them through
the n-gram beam search decoder. Useful for testing the decoder pipeline without
a trained neural model.

Usage:
    python examples/fake_logits_demo.py \
        --lexicon_path /path/to/lexicon.txt \
        --tokens_path /path/to/tokens.txt \
        --kenlm_model_path /path/to/lm.bin \
        --text "hello how are you" \
        --noise_std 3.0
"""

import argparse

import numpy as np
import torch
from g2p_en import G2p

from phoneme_to_words_lm import KenLMFlashlightTextLM
from phoneme_to_words_lm.utils import phonemize_sentence, LOGIT_PHONE_DEF


def create_fake_logits(text, g2p, noise_std):
    """Create synthetic phoneme logits for a given sentence.

    Phonemizes the sentence, then builds a logit matrix where each time step
    has a high value (100) at the correct phoneme index and zeros elsewhere.
    Blank frames are inserted between phonemes and at the start/end.
    Gaussian noise is added to simulate imperfect neural decoding.

    Args:
        text: Sentence to phonemize and encode.
        g2p: A g2p_en.G2p instance for grapheme-to-phoneme conversion.
        noise_std: Standard deviation of Gaussian noise added to logits.

    Returns:
        numpy array of shape (T, num_tokens) where num_tokens = len(LOGIT_PHONE_DEF).
    """
    phonemes = phonemize_sentence(text, g2p=g2p)

    # create fake logits that correspond to the phonemes
    logits = []
    for i in range(3):
        # add 3 blanks to the start
        logits.append(np.zeros(len(LOGIT_PHONE_DEF)))
        logits[-1][LOGIT_PHONE_DEF.index('BLANK')] = 100
    for i, p in enumerate(phonemes):
        # loop through phonemes
        logits.append(np.zeros(len(LOGIT_PHONE_DEF)))
        logits[-1][LOGIT_PHONE_DEF.index(p)] = 100
        for i in range(np.random.randint(0, 2)):
            # add up to 1 blank between phonemes
            logits.append(np.zeros(len(LOGIT_PHONE_DEF)))
            logits[-1][LOGIT_PHONE_DEF.index('BLANK')] = 100
    # add one more blank at the end
    logits.append(np.zeros(len(LOGIT_PHONE_DEF)))
    logits[-1][LOGIT_PHONE_DEF.index('BLANK')] = 100

    # convert to numpy array
    logits = np.array(logits)

    # add white noise
    logits += np.random.normal(0, noise_std, logits.shape)

    return logits


def main():
    parser = argparse.ArgumentParser(
        description="Create fake phoneme logits and decode with KenLMFlashlightTextLM")
    parser.add_argument("--lexicon_path", type=str, required=True,
                        help="Path to lexicon.txt (word -> phoneme mapping)")
    parser.add_argument("--tokens_path", type=str, required=True,
                        help="Path to tokens.txt (one token per line)")
    parser.add_argument("--kenlm_model_path", type=str, required=True,
                        help="Path to KenLM binary (.bin)")
    parser.add_argument("--text", type=str, default="hello how are you",
                        help="Sentence to encode as fake logits (default: 'hello how are you')")
    parser.add_argument("--noise_std", type=float, default=3.0,
                        help="Std dev of Gaussian noise added to logits (default: 3.0)")
    parser.add_argument("--n_best", type=int, default=10,
                        help="Number of top hypotheses to display (default: 10)")
    args = parser.parse_args()

    print(f"Input text: \"{args.text}\"")
    print(f"Noise std: {args.noise_std}")
    print()

    # Phonemize
    g2p = G2p()
    phonemes = phonemize_sentence(args.text, g2p=g2p)
    print(f"Phonemes: {' '.join(phonemes)}")

    # Create fake logits
    logits_np = create_fake_logits(args.text, g2p, args.noise_std)
    print(f"Logit shape: {logits_np.shape} (T={logits_np.shape[0]}, tokens={logits_np.shape[1]})")
    print()

    # Initialize decoder (no LLM rescoring for this demo)
    print("Initializing decoder...")
    decoder = KenLMFlashlightTextLM(
        lexicon_path=args.lexicon_path,
        tokens_path=args.tokens_path,
        kenlm_model_path=args.kenlm_model_path,
        do_llm_rescoring=False,
    )

    # --- Offline decoding ---
    print("\n=== Offline Decoding ===")
    logits_tensor = torch.from_numpy(logits_np).to(torch.float32).unsqueeze(0)  # (1, T, N)
    results = decoder.offline_decode(logits_tensor)
    hypo = results[0]

    n_show = min(args.n_best, len(hypo['word_seqs']))
    print(f"Top {n_show} hypotheses:")
    for i in range(n_show):
        print(f"  {i+1}. \"{hypo['word_seqs'][i]}\"  "
              f"(score={hypo['final_scores'][i]:.2f}, "
              f"acoustic={hypo['acoustic_scores'][i]:.2f}, "
              f"ngram={hypo['ngram_scores'][i]:.2f})")

    # --- Online (streaming) decoding ---
    print("\n=== Online (Streaming) Decoding ===")
    decoder.online_decode_begin()

    # Feed frames one at a time
    for t in range(logits_np.shape[0]):
        frame = torch.from_numpy(logits_np[t:t+1]).to(torch.float32)  # (1, N)
        step_result = decoder.online_decode_step(frame)
        if (t + 1) % 10 == 0 or t == logits_np.shape[0] - 1:
            print(f"  Frame {t+1}/{logits_np.shape[0]}: \"{step_result['word_seq']}\"")

    # Finalize
    final = decoder.online_decode_end()
    if final['word_seqs']:
        print(f"\nFinal result: \"{final['word_seqs'][0]}\"")
    else:
        print("\nFinal result: (empty)")


if __name__ == "__main__":
    main()
