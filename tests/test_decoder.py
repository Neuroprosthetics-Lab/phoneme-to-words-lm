import math
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from flashlight.lib.text.decoder import ZeroLM
from phoneme_to_words_lm.decoder import KenLMFlashlightTextLM
from phoneme_to_words_lm.utils import phonemize_sentence, cmu_dict


class DecoderFixture(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        (root/'tokens').write_text('BLANK\nSIL\nA\nB\n')
        (root/'lexicon').write_text('SINGLE A SIL\nDOUBLE A A SIL\nBEE B SIL\n')
        (root/'lm').write_text('ZeroLM fixture')
        self.paths = dict(lexicon_path=str(root/'lexicon'), tokens_path=str(root/'tokens'), kenlm_model_path=str(root/'lm'))
        self.patch = patch('phoneme_to_words_lm.decoder.KenLM', side_effect=lambda *_: ZeroLM())
        self.patch.start(); self.addCleanup(self.patch.stop)
        self.d = self.make()

    def make(self, **kw):
        cfg = dict(beam_size=100, token_beam_size=4, lm_weight=0., blank_penalty=1., do_llm_rescoring=False)
        cfg.update(kw)
        return KenLMFlashlightTextLM(**self.paths, **cfg)

    @staticmethod
    def logits(ids):
        x = torch.full((1,len(ids),4),-20.)
        x[0,range(len(ids)),ids] = 0.
        return x

class DecoderTests(DecoderFixture):
    def test_native_logadd_total_and_order_survive(self):
        x = torch.randn(1,12,4,generator=torch.Generator().manual_seed(0))
        values = self.d.process_logits(x)
        raw = self.d.ngram_decoder.decode(values.data_ptr(),12,4)
        h = self.d._extract_nbest(raw,100)
        winner = next(r for r in raw if any(w >= 0 for w in r.words))
        text = ' '.join(self.d.word_dict.get_entry(w).lower() for w in winner.words if w>=0)
        self.assertEqual(h['word_seqs'][0],text)
        self.assertEqual(h['beam_scores'][0],winner.score)
        self.assertEqual(h['final_scores'],h['beam_scores'])
        self.assertNotAlmostEqual(h['beam_scores'][0],h['acoustic_scores'][0])

    def test_nonzero_native_penalties_are_preserved(self):
        d = self.make(sil_score=-2.,word_score=1.7,unk_score=-10.,log_add=False)
        h = d.offline_decode(self.logits([2,1,0,1]))[0]
        self.assertEqual(h['final_scores'],h['beam_scores'])
        self.assertNotEqual(h['beam_scores'][0],h['acoustic_scores'][0]+1.7)

    def test_invalid_lengths_never_reach_native(self):
        x = self.logits([2,0,1])
        for lengths in [torch.tensor([4]),torch.tensor([-1]),torch.tensor([1.2]),torch.tensor([[2]]),torch.tensor([True]),torch.tensor([]),torch.tensor([2],device='meta')]:
            with self.subTest(lengths=lengths):
                with patch.object(self.d,'ngram_decoder') as native:
                    with self.assertRaises(ValueError): self.d.ngram_decode(x,lengths)
                    native.decode.assert_not_called()

    def test_invalid_logits(self):
        for x in [torch.zeros(1,3,4,dtype=torch.float64),torch.zeros(1,4,4).transpose(1,2),torch.zeros(3,4),torch.zeros(1,3,5),torch.full((1,3,4),float('nan')),torch.full((1,3,4),float('inf'))]:
            with self.subTest(shape=x.shape), self.assertRaises(ValueError): self.d.ngram_decode(x)

    def test_padding_and_batch_invariance(self):
        x = self.logits([2,1]); y = self.logits([3,0,3,1])
        batch = torch.full((2,4,4),float('nan')); batch[0,:2]=x[0]; batch[1]=y[0]
        both = self.d.offline_decode(batch,torch.tensor([2,4]))
        for result, alone in zip(both,[self.d.offline_decode(x)[0],self.d.offline_decode(y)[0]]):
            self.assertEqual(result['word_seqs'],alone['word_seqs'])
            self.assertEqual(result['beam_scores'],alone['beam_scores'])

    def test_empty_schema_and_no_llm_call(self):
        self.d.do_llm_rescoring = True
        with patch('phoneme_to_words_lm.decoder.score_sentences') as scorer:
            h = self.d.offline_decode(torch.zeros(1,0,4))[0]
            scorer.assert_not_called()
        self.assertEqual(h['status'],'no_input')
        self.assertEqual(h['word_seqs'],[''])
        self.assertEqual(h['raw_llm_scores'],[None])
        self.assertEqual(h['final_scores'],[-math.inf])
        self.assertEqual(self.d.ngram_decode(torch.zeros(0,3,4)),[])

    def test_repeated_phone_and_chunks(self):
        x = self.logits([0,2,0,0,2,1,0])
        expected = self.d.offline_decode(x)[0]
        self.assertEqual(expected['word_seqs'][0],'double')
        for size in [1,2,3,7]:
            self.d.online_decode_begin()
            for i in range(0,7,size): self.d.online_decode_step(x[:,i:i+size])
            got = self.d.online_decode_end()
            self.assertEqual(got['word_seqs'],expected['word_seqs'])
            self.assertEqual(got['beam_scores'],expected['beam_scores'])
            self.assertEqual(got['total_frames'],7)
            self.assertGreater(got['ngram_time'],0)

    def test_stream_interleaving_requires_explicit_abort(self):
        self.d.online_decode_step(self.logits([2]))
        for call in [lambda:self.d.offline_decode(self.logits([3,1])),self.d.online_decode_begin,self.d.init_ngram_decoder,self.d.init_ngram_resources]:
            with self.assertRaises(RuntimeError):call()
        self.d.online_decode_abort()
        self.assertEqual(self.d.offline_decode(self.logits([3,1]))[0]['word_seqs'][0],'bee')

    def test_rescore_failure_resets_stream(self):
        self.d.online_decode_step(self.logits([2,1]))
        self.d.do_llm_rescoring=True
        with patch.object(self.d,'llm_rescore',side_effect=RuntimeError('inference failed')):
            with self.assertRaisesRegex(RuntimeError,'inference failed'):self.d.online_decode_end()
        self.assertFalse(self.d._online_active)
        self.d.do_llm_rescoring=False
        self.assertEqual(self.d.offline_decode(self.logits([3,1]))[0]['word_seqs'][0],'bee')

    def test_invalid_step_does_not_start_stream(self):
        with self.assertRaises(ValueError): self.d.online_decode_step(torch.zeros(2,3,4))
        self.assertFalse(self.d._online_active)

    def test_stream_settings_cannot_change_mid_utterance(self):
        self.d.online_decode_step(self.logits([2]))
        self.d.temperature=2
        with self.assertRaises(RuntimeError):self.d.online_decode_step(self.logits([1]))
        self.d.online_decode_abort()

    def test_abort_works_after_invalid_setting_without_reconfiguring_search(self):
        self.d.online_decode_step(self.logits([2]))
        installed = dict(self.d._search_config)
        self.d.temperature = 0
        self.d.lm_weight = 1.
        self.d.online_decode_abort()
        self.assertFalse(self.d._online_active)
        self.assertEqual(self.d._search_config, installed)
        self.d.temperature = 1.
        with self.assertRaisesRegex(RuntimeError, 'init_ngram_decoder'):
            self.d.offline_decode(self.logits([2, 1]))
        self.d.init_ngram_decoder()
        self.assertEqual(self.d.offline_decode(self.logits([2, 1]))[0]['word_seqs'][0], 'single')

    def test_end_rejects_changed_preprocessing_and_resets(self):
        self.d.online_decode_step(self.logits([2, 1]))
        self.d.n_best = 2
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            self.d.online_decode_end()
        self.assertFalse(self.d._online_active)
        self.assertEqual(self.d.offline_decode(self.logits([2, 1]))[0]['word_seqs'][0], 'single')

    def test_search_settings_require_reinitialization(self):
        self.d.lm_weight=1.
        with self.assertRaisesRegex(RuntimeError,'init_ngram_decoder'):self.d.ngram_decode(self.logits([2,1]))
        self.d.init_ngram_decoder()
        self.d.ngram_decode(self.logits([2,1]))

    def test_hotwords_case_and_deferred_updates(self):
        self.d.set_hotwords({'single':2.})
        self.assertEqual(self.d.get_hotwords(),{})
        self.d.online_decode_begin()
        self.assertEqual(self.d.get_hotwords(),{'single':2.})
        trie=self.d.trie
        self.d.set_hotwords({'SINGLE':3.})
        self.assertEqual(self.d.get_hotwords(),{'single':2.})
        self.d.online_decode_end(); self.d.online_decode_begin()
        self.assertIs(trie,self.d.trie)
        self.assertEqual(self.d.get_hotwords(),{'single':3.})
        self.d.online_decode_end(); self.d.clear_hotwords(); self.d.online_decode_begin()
        self.assertEqual(self.d.get_hotwords(),{})

    def test_hotword_invalid_collisions_and_atomic_rebuild(self):
        for value in [{'SINGLE':1.,'single':2.},{'single':float('nan')},{'missing':1.}]:
            with self.assertRaises((ValueError,KeyError)):self.d.set_hotwords(value)
        self.d.set_hotwords({'single':1.})
        with patch('phoneme_to_words_lm.decoder._construct_trie',side_effect=RuntimeError('build failed')):
            with self.assertRaises(RuntimeError):self.d.online_decode_begin()
        self.assertIs(self.d.lm, self.d._kenlm)
        self.assertEqual(self.d.get_hotwords(),{})

    def test_initial_and_yaml_hotword_validation(self):
        self.assertEqual(self.make(hotwords={' SINGLE ': -2.}).get_hotwords(), {'single': -2.})
        for value in [[], '', {'single': True}, {'single': '2'}]:
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                self.make(hotwords=value)
        path = Path(self.tmp.name)/'hotwords.yaml'
        for value in ['single: 1\nsingle: 2\n', 'single: 1\nSINGLE: 2\n',
                      'single: true\n', '1: 2\n', 'single: .inf\n']:
            path.write_text(value)
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                self.make(hotwords_path=str(path))
        path.write_text('SINGLE: 0\n')
        with self.assertWarnsRegex(UserWarning, 'precedence'):
            decoder = self.make(hotwords={'bee': 2}, hotwords_path=str(path))
        self.assertEqual(decoder.get_hotwords(), {'single': 0.})

    def test_adapter_policy_checked_before_model_load(self):
        path = Path(self.tmp.name)/'adapter'
        path.mkdir()
        self.d.llm_lora_path = str(path)
        policy = dict(version='sentence_v1', prefix='\n', score_eos=False)
        (path/'llm_scoring.json').write_text(json.dumps(policy))
        with patch('phoneme_to_words_lm.decoder._build_llm', return_value=(object(), object())) as load:
            self.d.init_llm()
            load.assert_called_once()
        self.d.llm_score_eos = True
        with patch('phoneme_to_words_lm.decoder._build_llm') as load:
            with self.assertRaisesRegex(ValueError, 'differs'):
                self.d.init_llm()
            load.assert_not_called()

    def test_invalid_settings(self):
        for kw in [dict(temperature=0),dict(blank_penalty=-1),dict(n_best=0),dict(beam_size=-1),dict(token_beam_size=1.2),dict(llm_batch_size=0),dict(lm_weight=float('nan')),dict(llm_dtype='typo'),dict(llm_prefix=''),dict(blank_skip_threshold=0),dict(blank_skip_threshold=-.1),dict(blank_skip_threshold=1.01),dict(blank_skip_threshold=float('nan')),dict(blank_skip_threshold=True)]:
            with self.subTest(kw=kw),self.assertRaises(ValueError):self.make(**kw)

    def test_context_validation(self):
        for contexts in [['a','b'],[None],'abc']:
            with self.assertRaises((ValueError,TypeError)):self.d.offline_decode(self.logits([2,1]),contexts=contexts)
        with self.assertRaises(ValueError):self.d.llm_rescore([dict(word_seqs=['single'])])

    def test_resource_validation(self):
        p=Path(self.paths['lexicon_path'])
        for text in ['SINGLE A\n','SINGLE BAD SIL\n','SINGLE BLANK SIL\n','']:
            p.write_text(text)
            with self.assertRaises(ValueError):self.make()

    def test_phonemizer_preserves_repeats_and_corrects_right_word(self):
        p=cmu_dict['adventurer'][0]
        self.assertEqual(phonemize_sentence('adventurer',g2p=lambda _:p,verbosity=False),p+['SIL'])
        actual=phonemize_sentence('cat dog',g2p=lambda _:['K','AE','T',' ','K','AE','T'],verbosity=False)
        self.assertEqual(actual,['K','AE','T','SIL','D','AO','G','SIL'])
        with self.assertRaises(ValueError):phonemize_sentence('cat',g2p=lambda _:['K','AE','T'],return_seq=True,maxSeqLen=2)


class RankingTests(DecoderFixture):
    def test_final_ties_ignore_previous_rescoring_order(self):
        h = dict(word_seqs=['single', 'bee'], beam_scores=[-1., -2.])
        self.d.llm_alpha = 1.
        self.d.llm_model = object(); self.d.llm_tokenizer = object()
        with patch('phoneme_to_words_lm.decoder.score_sentences', return_value=([-2., 0.], [1, 1])):
            [first] = self.d.llm_rescore([h])
        self.assertEqual(first['word_seqs'], ['bee', 'single'])
        self.d.llm_alpha = .5
        with patch('phoneme_to_words_lm.decoder.score_sentences', return_value=([0., -2.], [1, 1])):
            [second] = self.d.llm_rescore([first])
        self.assertEqual(second['final_scores'], [-2., -2.])
        self.assertEqual(second['word_seqs'], ['single', 'bee'])

    def test_dedup_budget_ties_and_metadata(self):
        from types import SimpleNamespace
        ids=[self.d.word_dict.get_index(w) for w in ['SINGLE','DOUBLE','BEE']]
        def result(words,score):return SimpleNamespace(words=words,score=score,lmScore=-1.,emittingModelScore=-2.)
        rows=[result([],0),result([ids[0]],5),result([ids[0]],4),result([ids[1]],3),result([ids[2]],3)]
        h=self.d._extract_nbest(rows,2)
        self.assertEqual(h['word_seqs'],['single','bee'])
        self.assertEqual(h['beam_scores'],[5,3])
        self.assertEqual(h['raw_word_seqs'],['SINGLE','BEE'])
        self.assertEqual(self.d._extract_nbest(list(reversed(rows)),2),h)
        h['ngram_time']=.123;h['source_metadata']={'trial':42}
        self.d.llm_alpha=1.;self.d.llm_length_penalty=.5
        self.d.llm_model=object();self.d.llm_tokenizer=object()
        with patch('phoneme_to_words_lm.decoder.score_sentences',return_value=([-10.,-1.],[1,2])):
            [out]=self.d.llm_rescore([h])
        self.assertEqual(out['word_seqs'],['bee','single'])
        self.assertEqual(out['beam_scores'],[3,5]);self.assertEqual(out['final_scores'],[1.,-5.5])
        self.assertEqual(out['llm_token_counts'],[2,1]);self.assertEqual(out['raw_word_seqs'],['BEE','SINGLE'])
        self.assertEqual(out['ngram_time'],.123);self.assertEqual(out['source_metadata'],{'trial':42})
        self.assertEqual(h['final_scores'],[5,3])
        # Idempotent base: applying the same raw scores again does not add twice.
        with patch('phoneme_to_words_lm.decoder.score_sentences',return_value=([-1.,-10.],[2,1])):
            [twice]=self.d.llm_rescore([out])
        self.assertEqual(twice['final_scores'],out['final_scores'])

    def test_ambiguous_case_collision_rejected(self):
        Path(self.paths['lexicon_path']).write_text('Word A SIL\nWORD B SIL\n')
        d=self.make()
        with self.assertRaisesRegex(ValueError,'Ambiguous'):d.set_hotwords({'word':2.})

    def test_capacity_and_duplicate_tokens_rejected(self):
        Path(self.paths['lexicon_path']).write_text(''.join(f'W{i} A SIL\n' for i in range(61)))
        with self.assertRaisesRegex(ValueError,'homophone'):self.make()
        Path(self.paths['tokens_path']).write_text('BLANK\nSIL\nA\nA\n')
        with self.assertRaisesRegex(ValueError,'distinct'):self.make()

    def test_native_step_failure_aborts(self):
        with patch.object(self.d,'ngram_decoder') as native:
            native.decode_step.side_effect=RuntimeError('native failed')
            with self.assertRaisesRegex(RuntimeError,'native failed'):self.d.online_decode_step(self.logits([2]))
        self.assertFalse(self.d._online_active)

if __name__=='__main__':unittest.main()
