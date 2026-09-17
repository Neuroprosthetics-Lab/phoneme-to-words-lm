"""An empty native winner must not be replaced by a lower-scoring word."""
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from test_decoder import DecoderFixture
from phoneme_to_words_lm.evaluation import prepare_ranking


class EmptyWinnerTests(DecoderFixture):
    @staticmethod
    def native(words, score):
        return SimpleNamespace(words=words, score=score, lmScore=-2., emittingModelScore=-3.)

    def test_top_empty_and_ties_preserve_native_scores(self):
        word = self.d.word_dict.get_index('SINGLE')
        for nonempty_score in (-6., -5.):
            rows = [self.native([-1, -1], -5.), self.native([word], nonempty_score)]
            for order in (rows, list(reversed(rows))):
                result = self.d._extract_nbest(order, 1)
                self.assertEqual(result['status'], 'empty')
                self.assertEqual(result['word_seqs'], [''])
                self.assertEqual(result['word_ids'], [[]])
                self.assertEqual(result['beam_scores'], [-5.])
                self.assertEqual(result['final_scores'], [-5.])
                self.assertEqual(result['ngram_scores'], [-2.])
                self.assertEqual(result['acoustic_scores'], [-3.])

    def test_lower_empty_and_nonfinite_paths_do_not_displace_words(self):
        word = self.d.word_dict.get_index('SINGLE')
        rows = [self.native([], math.inf), self.native([], -6.), self.native([word], -5.)]
        result = self.d._extract_nbest(rows, 2)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['word_seqs'], ['single'])
        self.assertEqual(self.d._extract_nbest([], 1)['status'], 'no_path')
        self.assertEqual(self.d._extract_nbest([self.native([], -math.inf)], 1)['status'], 'no_path')

    def test_punctuation_only_raw_word_is_not_an_empty_native_path(self):
        Path(self.paths['lexicon_path']).write_text('!!! A SIL\nSINGLE A SIL\n')
        decoder = self.make()
        punctuation = decoder.word_dict.get_index('!!!')
        word = decoder.word_dict.get_index('SINGLE')
        result = decoder._extract_nbest([self.native([punctuation], -1.), self.native([word], -2.)], 1)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['word_seqs'], ['single'])

    def test_synthetic_blank_runs_full_compressed_and_streamed(self):
        x = self.logits([0]*8)
        for threshold in (1., .98):
            decoder = self.make(blank_skip_threshold=threshold)
            decoder.do_llm_rescoring = True
            with patch('phoneme_to_words_lm.decoder.score_sentences') as scorer:
                expected = decoder.offline_decode(x)[0]
                self.assertEqual(expected['status'], 'empty')
                self.assertTrue(math.isfinite(expected['beam_scores'][0]))
                for size in (1, 3, 8):
                    decoder.online_decode_begin()
                    for start in range(0, 8, size):
                        decoder.online_decode_step(x[:, start:start+size])
                    actual = decoder.online_decode_end()
                    for field in ('word_seqs', 'status', 'beam_scores', 'final_scores', 'search_frames'):
                        self.assertEqual(actual[field], expected[field])
                scorer.assert_not_called()
            self.assertEqual(decoder.offline_decode(x[:, :0])[0]['status'], 'no_input')

    def test_mixed_batch_scores_only_speech_and_counts_empty_as_deletions(self):
        decoder = self.make(n_best=1)
        decoder.do_llm_rescoring = True
        decoder.llm_model = decoder.llm_tokenizer = object()
        x = torch.cat([self.logits([0, 0]), self.logits([2, 1])])
        with patch('phoneme_to_words_lm.decoder.score_sentences', return_value=([-1.], [1])) as scorer:
            empty, speech = decoder.offline_decode(x)
        self.assertEqual(scorer.call_args.args[2], ['single'])
        self.assertEqual(empty['raw_llm_scores'], [None])
        self.assertEqual(empty['llm_token_counts'], [0])
        self.assertEqual(empty['final_scores'], empty['beam_scores'])
        self.assertEqual(speech['status'], 'ok')
        empty['transcription'] = 'two words'
        arrays = prepare_ranking([empty])
        self.assertEqual(arrays[0]['edits'].tolist(), [2])


if __name__ == '__main__':
    unittest.main()
