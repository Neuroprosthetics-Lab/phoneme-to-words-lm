"""
LM decoder hyperparameter sweep worker.

Standalone script executed as a subprocess by lm_sweep.py. Loads precomputed
phoneme logits from a pkl file, initializes a KenLMFlashlightTextLM decoder
with trial hyperparameters, decodes all (or subsampled) sentences, sweeps
alpha post-hoc, and writes results to a JSON file.

Usage (standalone testing):
    python lm_sweep_worker.py --config worker_config.json
"""

import argparse
import json
import math
import os
import pickle
import time

import editdistance
import numpy as np
import torch

from phoneme_to_words_lm import KenLMFlashlightTextLM
from phoneme_to_words_lm.utils import remove_punctuation, replace_words, HF_CACHE_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="LM sweep worker")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to worker config JSON")
    return parser.parse_args()


def load_and_filter_logits(cfg):
    """Load logits pkl and return filtered, subsampled list of sentence dicts."""
    with open(cfg["logits_pkl_path"], "rb") as f:
        phoneme_results = pickle.load(f)

    # Normalize transcriptions
    for r in phoneme_results:
        r["transcription"] = replace_words(remove_punctuation(r["transcription"]))

    filtered = []
    for r in phoneme_results:
        t = r["transcription"]
        words = t.split()
        if not words or max(len(w) for w in words) == 1:
            continue
        filtered.append(r)

    # Subsample
    eval_every_nth = cfg.get("eval_every_nth", 1)
    if eval_every_nth > 1:
        filtered = filtered[::eval_every_nth]

    return filtered


def decode_sentences(decoder, sentences, reorder_logit_columns):
    """Decode all sentences and collect per-sentence results and timing."""
    all_results = []

    for i, sent in enumerate(sentences):
        logits = torch.from_numpy(sent["logits"]).to(torch.float32).unsqueeze(0)

        if reorder_logit_columns:
            # [BLANK, ph1..phN, SIL] -> [BLANK, SIL, ph1..phN]
            logits = torch.cat(
                (logits[:, :, 0:1], logits[:, :, -1:], logits[:, :, 1:-1]),
                dim=-1,
            )

        lengths = torch.tensor([sent["adjusted_lens"]], dtype=torch.int32)

        decoder_results = decoder.offline_decode(logits=logits, lengths=lengths)
        hypo = decoder_results[0]

        decode_time = hypo.get("ngram_time", 0.0) or 0.0
        llm_time = hypo.get("llm_rescore_time", 0.0) or 0.0
        total_time = decode_time + llm_time

        all_results.append({
            "word_seqs": hypo["word_seqs"],
            "acoustic_scores": hypo["acoustic_scores"],
            "ngram_scores": hypo["ngram_scores"],
            "llm_scores": hypo.get("llm_scores", []),
            "transcription": sent["transcription"],
            "decode_time": total_time,
        })

        if (i + 1) % 50 == 0 or (i + 1) == len(sentences):
            avg_t = np.mean([r['decode_time'] for r in all_results])
            print(f"  Decoded {i + 1}/{len(sentences)} sentences "
                  f"(avg {avg_t:.3f}s/sent)")

    return all_results


def alpha_posthoc_sweep(all_results, lm_weight, alpha_low, alpha_high, alpha_step):
    """Sweep alpha values post-hoc and return (best_alpha, best_wer, alpha_wer_curve)."""
    if not all_results:
        raise ValueError("No sentences to evaluate — all were filtered out. "
                         "Check logits pkl and filtering settings.")

    alphas = np.round(np.arange(alpha_low, alpha_high + alpha_step / 2, alpha_step),
                      decimals=10)

    # Pre-convert scores to numpy arrays once (avoids repeated conversion per alpha)
    precomputed = []
    for result in all_results:
        label_words = result["transcription"].split()
        if len(result["word_seqs"]) == 0:
            precomputed.append({
                "empty": True,
                "n_ref_words": len(label_words),
            })
        else:
            acoustic = np.array(result["acoustic_scores"])
            ngram = np.array(result["ngram_scores"])
            llm = np.array(result["llm_scores"]) if result["llm_scores"] else np.zeros_like(acoustic)
            base_scores = acoustic + (lm_weight * ngram)
            precomputed.append({
                "empty": False,
                "base_scores": base_scores,
                "llm": llm,
                "word_seqs": result["word_seqs"],
                "label_words": label_words,
            })

    alpha_wer_curve = []
    best_wer = float("inf")
    best_n_edits = 0
    best_n_words = 0

    for alpha in alphas:
        n_words_total = 0
        n_edits_total = 0

        for p in precomputed:
            if p["empty"]:
                n_words_total += p["n_ref_words"]
                n_edits_total += p["n_ref_words"]
                continue

            final = p["base_scores"] + (alpha * p["llm"])
            best_idx = np.argmax(final)
            pred_words = p["word_seqs"][best_idx].split()

            n_words_total += len(p["label_words"])
            n_edits_total += editdistance.eval(p["label_words"], pred_words)

        wer = n_edits_total / n_words_total if n_words_total > 0 else 0.0
        alpha_wer_curve.append([float(alpha), float(wer)])

        if wer < best_wer:
            best_wer = wer
            best_n_edits = n_edits_total
            best_n_words = n_words_total

    # Among alphas tied at best WER, pick the middle one for stability.
    # Two WERs are equal iff their edit counts are equal (n_words is constant),
    # so float equality is exact here.
    tied = [a for a, w in alpha_wer_curve if w == best_wer]
    best_alpha = tied[len(tied) // 2]

    return best_alpha, best_wer, best_n_edits, best_n_words, alpha_wer_curve


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    trial_params = cfg["trial_params"]
    lm_weight = float(trial_params["lm_weight"])

    # Cap VRAM so multiple workers can share a GPU without OOM
    max_vram_fraction = cfg.get("max_vram_fraction")
    if max_vram_fraction is not None:
        gpu_idx = int(cfg["llm_device"].split(":")[-1])
        torch.cuda.set_per_process_memory_fraction(max_vram_fraction, gpu_idx)

    print(f"LM Sweep Worker")
    print(f"  Logits: {cfg['logits_pkl_path']}")
    print(f"  LLM: {trial_params.get('llm_model_name', 'N/A')}")
    print(f"  GPU: {cfg['llm_device']}")
    print(f"  Params: {trial_params}")
    print()

    print("Loading logits...")
    t0 = time.time()
    sentences = load_and_filter_logits(cfg)
    print(f"  Loaded {len(sentences)} sentences in {time.time() - t0:.1f}s")
    print()

    print("Initializing decoder...")
    t0 = time.time()
    decoder = KenLMFlashlightTextLM(
        lexicon_path=cfg["lexicon_path"],
        tokens_path=cfg["tokens_path"],
        kenlm_model_path=cfg["kenlm_model_path"],
        beam_size=int(trial_params["beam_size"]),
        token_beam_size=int(trial_params["token_beam_size"]),
        beam_threshold=float(trial_params["beam_threshold"]),
        blank_skip_threshold=float(trial_params["blank_skip_threshold"]),
        lm_weight=lm_weight,
        word_score=0.0,
        unk_score=-math.inf,
        sil_score=0.0,
        log_add=True,
        n_best=int(trial_params["n_best"]),
        temperature=float(trial_params["temperature"]),
        blank_penalty=float(trial_params["blank_penalty"]),
        do_llm_rescoring=True,
        llm_model_name=trial_params["llm_model_name"],
        llm_cache_dir=cfg.get("llm_cache_dir") or HF_CACHE_DIR,
        llm_device=cfg["llm_device"],
        llm_dtype=cfg.get("llm_dtype", "bfloat16"),
        llm_alpha=0.0,  # placeholder — alpha is swept post-hoc
        llm_length_penalty=float(trial_params["llm_length_penalty"]),
        llm_batch_size=int(cfg.get("llm_batch_size", 100)),
        llm_lora_path=cfg.get("llm_lora_path", None),
        hotwords=cfg.get("hotwords"),
        hotwords_path=cfg.get("hotwords_path"),
    )
    print(f"  Decoder initialized in {time.time() - t0:.1f}s")
    print()

    print("Decoding...")
    reorder = cfg.get("reorder_logit_columns", True)
    all_results = decode_sentences(decoder, sentences, reorder)

    decode_times = [r["decode_time"] for r in all_results]
    avg_decode_time = float(np.mean(decode_times)) if decode_times else 0.0
    print(f"\n  Avg decode time: {avg_decode_time:.4f}s/sentence")
    print()

    print("Running alpha post-hoc sweep...")
    alpha_low = cfg.get("alpha_low", 0.0)
    alpha_high = cfg.get("alpha_high", 3.0)
    alpha_step = cfg.get("alpha_step", 0.05)

    best_alpha, best_wer, n_edits, n_words, alpha_wer_curve = alpha_posthoc_sweep(
        all_results, lm_weight, alpha_low, alpha_high, alpha_step,
    )
    print(f"  Best alpha: {best_alpha:.3f}")
    print(f"  Best WER: {best_wer:.5f} ({n_edits} edits / {n_words} words)")

    results = {
        "wer": float(best_wer),
        "avg_decode_time": avg_decode_time,
        "best_alpha": best_alpha,
        "n_sentences_evaluated": len(sentences),
        "n_words_total": n_words,
        "n_edits_total": n_edits,
        "alpha_wer_curve": alpha_wer_curve,
    }

    output_path = cfg["output_path"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
