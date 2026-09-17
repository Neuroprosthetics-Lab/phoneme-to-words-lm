import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import math
import json
import tempfile
from types import SimpleNamespace

import torch
from test_llm_scoring import TinyModel, CharacterTokenizer
from phoneme_to_words_lm.llm_scoring import (prepare_scoring_inputs, training_example,
    ScoringDataCollator, sentence_perplexity, score_sentences)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sweep'))
from lm_sweep_worker import alpha_posthoc_sweep
from lm_sweep import load_lm_sweep_config, make_lm_objective

class IntegrationTests(unittest.TestCase):
    def test_training_labels_match_scoring_targets(self):
        tok=CharacterTokenizer();model=TinyModel()
        requests=prepare_scoring_inputs(tok,['a','abc'],score_eos=True)
        batch=ScoringDataCollator(0)([training_example(r) for r in requests])
        self.assertEqual(batch['labels'][0].tolist(),[-100,35,0,-100,-100])
        logits=model(batch['input_ids'],batch['attention_mask'],False).logits
        nll=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,64),batch['labels'][:,1:].reshape(-1),reduction='sum')
        raw,counts=score_sentences(model,tok,['a','abc'],score_eos=True)
        self.assertAlmostEqual(float(nll),-sum(raw),places=4)
        self.assertEqual(int((batch['labels'][:,1:]!=-100).sum()),sum(counts))

    def test_perplexity_batch_invariance_and_training_mode(self):
        tok=CharacterTokenizer();model=TinyModel();model.train()
        a=sentence_perplexity(model,tok,['a','abc'],batch_size=1)
        b=sentence_perplexity(model,tok,['a','abc'],batch_size=2)
        self.assertAlmostEqual(a,b,places=4);self.assertTrue(model.training)
        self.assertTrue(math.isfinite(sentence_perplexity(model,tok,['a'])))
        self.assertEqual(sentence_perplexity(model,tok,['']),math.inf)
        with self.assertRaises(ValueError):sentence_perplexity(model,tok,['too long'],max_length=2)
        self.assertTrue(model.training)

    def test_sweep_uses_beam_and_same_alpha_formula(self):
        row=dict(word_seqs=['correct','wrong'],acoustic_scores=[-10.,-1.],ngram_scores=[0.,0.],beam_scores=[-1.,-2.],llm_scores=[-2.,0.],transcription='correct')
        result=alpha_posthoc_sweep([row],1.,0.,1.,.5)
        self.assertEqual(result[-1],[[0.,0.],[.5,0.],[1.,1.]])
        self.assertEqual(result[1],0.)
        # Reranking a dump previously ordered by another alpha must preserve
        # the direct scorer's native-beam secondary choice at the .5 tie.
        reversed_row = dict(row)
        for key in ['word_seqs', 'acoustic_scores', 'ngram_scores', 'beam_scores', 'llm_scores']:
            reversed_row[key] = list(reversed(row[key]))
        self.assertEqual(alpha_posthoc_sweep([reversed_row],1.,0.,1.,.5)[-1],result[-1])
        empty=dict(word_seqs=[''],beam_scores=[-math.inf],llm_scores=[None],transcription='a word')
        self.assertEqual(alpha_posthoc_sweep([empty],1.,0.,1.,.5)[1],1.)
        legacy=dict(row);del legacy['beam_scores']
        with self.assertRaisesRegex(ValueError,'beam_scores'):alpha_posthoc_sweep([legacy],1.,0.,1.,.5)

    def test_launcher_serializes_scoring_policy_and_allocation_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            resource = root/'resource'
            resource.write_text('fixture')
            policy = dict(llm_prefix='Sentence:\n', llm_score_eos=True,
                          llm_max_length=127, llm_max_batch_tokens=511,
                          llm_logprob_chunk_size=3)
            config = dict.fromkeys(['lexicon_path', 'tokens_path', 'kenlm_model_path',
                                   'logits_pkl_path'], str(resource))
            config.update(policy, output_dir=str(root/'output'),
                          parameters={'beam_size': {'type': 'fixed', 'value': 5}})
            path = root/'config.yaml'
            path.write_text(json.dumps(config))
            cfg = load_lm_sweep_config(str(path))
            seen = []
            def worker(command, **kwargs):
                payload = json.loads(Path(command[-1]).read_text())
                seen.append(payload)
                Path(payload['output_path']).write_text(json.dumps(dict(
                    wer=.1, avg_decode_time=.2, best_alpha=.3,
                    n_sentences_evaluated=1, n_words_total=10, n_edits_total=1)))
                return SimpleNamespace(returncode=0, stdout='', stderr='')
            pool = SimpleNamespace(acquire=lambda: 0, release=lambda _: None)
            trial = SimpleNamespace(number=0, set_user_attr=lambda *_: None)
            with patch('lm_sweep.subprocess.run', side_effect=worker):
                make_lm_objective(cfg, pool, 'unused_worker.py', {0: 1})(trial)
            self.assertEqual({k: seen[0][k] for k in policy}, policy)
            self.assertIsNone(seen[0]['llm_cache_dir'])

if __name__=='__main__':unittest.main()
