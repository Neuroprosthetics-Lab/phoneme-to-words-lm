"""Regression cases for keep-first blank compression, including chunk edges."""
import unittest
from unittest.mock import patch

import torch
from test_decoder import DecoderFixture


class BlankCompressionTests(DecoderFixture):
    def setUp(self):
        super().setUp()
        self.d.blank_skip_threshold = .98

    def test_keeps_first_original_frame_and_thresholds_before_penalty(self):
        self.d.blank_penalty = 9.
        # Leading, internal and trailing runs, separated by low-confidence
        # blanks as well as phones. No selected row may be modified.
        probabilities = torch.tensor([
            [.99, .002, .005, .003], [.999, .0002, .0005, .0003],
            [.97, .01, .01, .01], [.99, .003, .002, .005],
            [.99, .003, .005, .002], [.01, .01, .97, .01],
            [.99, .004, .003, .003], [.999, .0003, .0003, .0004]])
        processed = self.d.process_logits(probabilities.log().unsqueeze(0))[0]
        selected, carry = self.d._select_blank_frames(processed)
        torch.testing.assert_close(selected, processed[[0, 2, 3, 5, 6]], rtol=0, atol=0)
        self.assertTrue(carry)
        self.assertTrue(selected.is_contiguous())
        # Carry from a previous chunk discards only the leading continuation.
        continued, _ = self.d._select_blank_frames(processed, True)
        torch.testing.assert_close(continued, processed[[2, 3, 5, 6]], rtol=0, atol=0)

    def test_temperature_controls_blank_confidence(self):
        x = torch.tensor([[[5., 0., 0., 0.], [5., 0., 0., 0.]]])
        self.d.temperature = .5
        self.assertEqual(len(self.d._select_blank_frames(self.d.process_logits(x)[0])[0]), 1)
        self.d.temperature = 2.
        self.assertEqual(len(self.d._select_blank_frames(self.d.process_logits(x)[0])[0]), 2)

    def test_repeated_phone_and_scores_match_manually_compressed_input(self):
        x = self.logits([0, 0, 2, 0, 0, 0, 2, 1, 0, 0])
        compressed = self.d.offline_decode(x)[0]
        full = self.make().offline_decode(x[:, [0, 2, 3, 6, 7, 8]].contiguous())[0]
        self.assertEqual(compressed['word_seqs'][0], 'double')
        for key in ['word_seqs', 'word_ids', 'raw_word_seqs', 'beam_scores',
                    'acoustic_scores', 'ngram_scores', 'final_scores']:
            self.assertEqual(compressed[key], full[key])
        self.assertEqual((compressed['total_frames'], compressed['search_frames']), (10, 6))
        self.assertEqual(compressed['preprocessing_config']['blank_skip_policy'], 'keep_first')
        # The old deletion policy loses both separators and predicts SINGLE.
        deleted = self.make().offline_decode(x[:, [2, 6, 7]].contiguous())[0]
        self.assertEqual(deleted['word_seqs'][0], 'single')

    def test_all_two_chunk_splits_and_empty_chunks_match_offline(self):
        x = self.logits([0, 0, 2, 0, 0, 0, 2, 1, 0, 0])
        expected = self.d.offline_decode(x)[0]
        for split in range(x.shape[1]+1):
            with self.subTest(split=split):
                self.d.online_decode_begin()
                self.d.online_decode_step(x[:, :split])
                self.d.online_decode_step(x[:, :0])
                self.d.online_decode_step(x[:, split:])
                result = self.d.online_decode_end()
                for key in ['word_seqs', 'beam_scores', 'acoustic_scores',
                            'ngram_scores', 'total_frames', 'search_frames', 'preprocessing_config']:
                    self.assertEqual(result[key], expected[key])

    def test_one_frame_chunks_preserve_a_blank_run_across_calls(self):
        x = self.logits([2, 0, 0, 0, 2, 1])
        self.d.online_decode_begin()
        search_counts = []
        for i in range(x.shape[1]):
            step = self.d.online_decode_step(x[:, i:i+1])
            self.assertEqual(step['total_frames'], i+1)
            search_counts.append(step['search_frames'])
        self.assertEqual(search_counts, [1, 2, 2, 2, 3, 4])
        self.assertEqual(self.d.online_decode_end()['word_seqs'][0], 'double')

    def test_all_blank_empty_input_and_utterance_reset(self):
        x = self.logits([0, 0, 0, 0])
        first = self.d.offline_decode(x)[0]
        self.assertEqual((first['total_frames'], first['search_frames']), (4, 1))
        self.assertEqual(first['status'], 'empty')
        for abort in [True, False]:
            self.d.online_decode_step(x)
            if abort:
                self.d.online_decode_abort()
            else:
                self.d.online_decode_end()
            self.assertEqual(self.d.online_decode_step(x)['search_frames'], 1)
            self.d.online_decode_end()
        empty = self.d.offline_decode(x[:, :0])[0]
        self.assertEqual((empty['total_frames'], empty['search_frames']), (0, 0))
        self.assertEqual(empty['status'], 'no_input')

    def test_batch_padding_does_not_extend_blank_runs(self):
        x = torch.full((2, 7, 4), float('nan'))
        x[0, :3] = self.logits([2, 0, 0])[0]
        x[1] = self.logits([0, 0, 3, 0, 0, 0, 1])[0]
        results = self.d.offline_decode(x, torch.tensor([3, 7]))
        self.assertEqual([r['search_frames'] for r in results], [2, 4])
        for row, length, result in zip(x, [3, 7], results):
            separate = self.d.offline_decode(row[:length].unsqueeze(0).contiguous())[0]
            self.assertEqual(result['beam_scores'], separate['beam_scores'])
            self.assertEqual(result['word_seqs'], separate['word_seqs'])

    def test_threshold_one_disables_compression_even_with_certain_blanks(self):
        self.d.blank_skip_threshold = 1.
        x = self.logits([0, 0, 2, 0, 0, 2, 1])
        processed = self.d.process_logits(x)[0]
        selected, carry = self.d._select_blank_frames(processed, True)
        self.assertIs(selected, processed)
        self.assertFalse(carry)
        result = self.d.offline_decode(x)[0]
        self.assertEqual(result['search_frames'], 7)

    def test_mid_stream_threshold_change_rejected(self):
        self.d.online_decode_step(self.logits([0, 0]))
        self.d.blank_skip_threshold = .99
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            self.d.online_decode_step(self.logits([2]))
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            self.d.online_decode_end()
        self.assertFalse(self.d._online_active)

    def test_discarded_only_chunk_does_not_call_native_step(self):
        self.d.online_decode_step(self.logits([0]))
        with patch.object(self.d, 'ngram_decoder', wraps=self.d.ngram_decoder) as native:
            step = self.d.online_decode_step(self.logits([0, 0, 0]))
            native.decode_step.assert_not_called()
        self.assertEqual((step['total_frames'], step['search_frames']), (4, 1))
        self.d.online_decode_abort()


if __name__ == '__main__':
    unittest.main()
