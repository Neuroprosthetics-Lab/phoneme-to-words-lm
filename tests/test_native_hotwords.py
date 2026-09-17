"""Native bonus arithmetic/ownership and decoder hotword transitions.

Run with the rebuilt flashlight-text wheel to exercise all tests. The old
extension remains supported through an explicitly tested Python fallback.
"""
import gc
import math
import unittest
import warnings
import weakref
from unittest.mock import patch

import torch
from flashlight.lib.text.decoder import LM, LMState, ZeroLM
from phoneme_to_words_lm.decoder import BiasingLM, HotwordLM
from test_decoder import DecoderFixture


class RecordingLM(LM):
    def __init__(self):
        super().__init__()
        self.initial = LMState()
        self.started = None

    def start(self, start_with_nothing):
        self.started = start_with_nothing
        return self.initial

    def score(self, state, label):
        # Model a native float32 return even when invoked directly in Python.
        return state.child(label), -0.10000000149011612

    def finish(self, state):
        return state.child(1000), -1.7


@unittest.skipIf(HotwordLM is None, 'rebuild flashlight_text for native API tests')
class NativeHotwordTests(unittest.TestCase):
    def test_double_addition_then_float_rounding(self):
        inner = RecordingLM()
        # 0.1 nearly cancels the inner float32 -0.1; prematurely converting
        # the bonus to float32 would incorrectly return exactly zero.
        bonuses = {0: 0.1, 1: -0.123456789123, 2: 0., 3: 1.23456789123}
        native = HotwordLM(inner, bonuses)
        reference = BiasingLM(inner)
        reference.hotword_bonus = bonuses
        state = native.start(False)
        for label in range(5):
            got_state, got = native.score(state, label)
            ref_state, expected = LM.score(reference, state, label)
            self.assertEqual(got_state.compare(ref_state), 0)
            self.assertEqual(got, expected)
        self.assertNotEqual(native.score(state, 0)[1], 0.)

    def test_boundaries_and_state_identity(self):
        inner = RecordingLM()
        native = HotwordLM(inner, {0: 2., 1000: 99.})
        for start_with_nothing in (False, True):
            state = native.start(start_with_nothing)
            self.assertIs(inner.started, start_with_nothing)
            self.assertEqual(state.compare(inner.initial), 0)
            next_state, _ = native.score(state, 0)
            self.assertEqual(next_state.compare(inner.score(state, 0)[0]), 0)
            finished, score = native.finish(next_state)
            expected_state, expected_score = LM.finish(inner, next_state)
            self.assertEqual(finished.compare(expected_state), 0)
            self.assertEqual(score, expected_score)  # No bonus on finish/EOS.

    def test_replacement_validation_and_copy(self):
        native = HotwordLM(ZeroLM(), {2: 1.})
        copy = native.hotword_bonus
        copy[2] = 4.
        self.assertEqual(native.hotword_bonus, {2: 1.})
        for bonuses in ({-1: 1.}, {1: math.nan}, {1: math.inf}, {1: -math.inf}):
            with self.assertRaises(ValueError):
                native.hotword_bonus = bonuses
            self.assertEqual(native.hotword_bonus, {2: 1.})
        with self.assertRaises(ValueError):
            HotwordLM(None)
        state = native.start(False)
        for bonuses in ({2: 0.}, {2: -1.}, {3: 2.}, {}):
            native.hotword_bonus = bonuses
            for label in (2, 3):
                self.assertEqual(native.score(state, label)[1], bonuses.get(label, 0.))

    def test_owns_native_inner_and_keeps_python_inner_alive(self):
        for factory in (ZeroLM, RecordingLM):
            inner = factory()
            reference = weakref.ref(inner)
            native = HotwordLM(inner, {2: 2.})
            del inner
            gc.collect()
            self.assertIsNotNone(reference())
            state = native.start(False)
            self.assertTrue(math.isfinite(native.score(state, 2)[1]))
            self.assertTrue(math.isfinite(native.finish(state)[1]))
            del native
            gc.collect()
            self.assertIsNone(reference())


class HotwordBackendTests(DecoderFixture):
    def test_empty_path_uses_native_lm_even_on_old_extension(self):
        with patch('phoneme_to_words_lm.decoder.HotwordLM', None), warnings.catch_warnings(record=True) as caught:
            decoder = self.make()
            decoder.clear_hotwords()
            decoder.offline_decode(self.logits([2, 1]))
        self.assertEqual(caught, [])
        self.assertIs(decoder.lm, decoder._kenlm)
        self.assertEqual(decoder.hotword_backend, 'kenlm')

    def test_old_extension_warns_and_remains_usable(self):
        with patch('phoneme_to_words_lm.decoder.HotwordLM', None):
            with self.assertWarnsRegex(RuntimeWarning, 'Rebuild/install'):
                decoder = self.make(hotwords={'single': .1}, lm_weight=1.)
            self.assertEqual(decoder.hotword_backend, 'python')
            decoder.offline_decode(self.logits([2, 1]))
            decoder.set_hotwords({'single': -.2})
            decoder.offline_decode(self.logits([2, 1]))
            self.assertEqual(decoder.get_hotwords(), {'single': -.2})
            decoder.clear_hotwords()
            decoder.offline_decode(self.logits([2, 1]))
            self.assertIs(decoder.lm, decoder._kenlm)

    @unittest.skipIf(HotwordLM is None, 'rebuild flashlight_text for native transitions')
    def test_backend_transitions_and_magnitude_do_not_reload_resources(self):
        decoder = self.make(lm_weight=1.)
        kenlm = decoder._kenlm
        original = decoder.offline_decode(self.logits([2, 1]))[0]
        decoder.set_hotwords({'single': .1})
        self.assertEqual(decoder.hotword_backend, 'kenlm')
        decoder.online_decode_begin()
        self.assertEqual(decoder.hotword_backend, 'native')
        trie, lm, search = decoder.trie, decoder.lm, decoder.ngram_decoder
        decoder.set_hotwords({'SINGLE': -.2})
        self.assertEqual(decoder.lm.hotword_bonus, {decoder.word_dict.get_index('SINGLE'): .1})
        decoder.online_decode_end()
        decoder.online_decode_begin()
        self.assertIs(decoder.trie, trie)
        self.assertIs(decoder.lm, lm)
        self.assertIs(decoder.ngram_decoder, search)
        self.assertEqual(decoder.get_hotwords(), {'single': -.2})
        decoder.online_decode_abort()
        decoder.clear_hotwords()
        restored = decoder.offline_decode(self.logits([2, 1]))[0]
        self.assertIs(decoder._kenlm, kenlm)
        self.assertIs(decoder.lm, kenlm)
        self.assertNotEqual(decoder.trie, trie)
        for field in ('word_ids', 'word_seqs', 'beam_scores', 'ngram_scores', 'acoustic_scores'):
            self.assertEqual(original[field], restored[field])

    @unittest.skipIf(HotwordLM is None, 'rebuild flashlight_text for native transitions')
    def test_native_decoder_owns_lm_after_python_references_are_dropped(self):
        decoder = self.make(hotwords={'single': 1.}, lm_weight=1.)
        search = decoder.ngram_decoder
        scores = decoder.process_logits(self.logits([2, 1]))
        del decoder
        gc.collect()
        result = search.decode(scores.data_ptr(), 2, 4)
        self.assertTrue(result)
        self.assertTrue(math.isfinite(result[0].score))

    @unittest.skipIf(HotwordLM is None, 'rebuild flashlight_text for native transitions')
    def test_native_constructor_or_decoder_failure_is_atomic(self):
        self.d.set_hotwords({'single': 1.})
        for target in ('phoneme_to_words_lm.decoder.HotwordLM',
                       'phoneme_to_words_lm.decoder.LexiconDecoder'):
            previous = self.d.trie, self.d.lm, self.d.ngram_decoder
            with patch(target, side_effect=RuntimeError('build failed')):
                with self.assertRaisesRegex(RuntimeError, 'build failed'):
                    self.d.online_decode_begin()
            self.assertEqual(previous, (self.d.trie, self.d.lm, self.d.ngram_decoder))
            self.assertEqual(self.d.get_hotwords(), {})
            self.assertEqual(self.d._pending_hotwords, {'single': 1.})
        self.d.online_decode_begin()
        self.assertEqual(self.d.get_hotwords(), {'single': 1.})

    @unittest.skipIf(HotwordLM is None, 'rebuild flashlight_text for native comparisons')
    def test_ctc_candidates_scores_and_stream_updates_match_python(self):
        for log_add in (False, True):
            decoder = self.make(lm_weight=1.5, log_add=log_add, blank_skip_threshold=.98)
            with patch('phoneme_to_words_lm.decoder.HotwordLM', None), warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                reference = self.make(lm_weight=1.5, log_add=log_add, blank_skip_threshold=.98)
                updates = ({}, {'single': .1}, {'single': 0.}, {'single': -.1},
                           {'bee': 1.23456789123, 'single': -.25}, {})
                for i, hotwords in enumerate(updates):
                    # Use the legacy wrapper for every update in the reference,
                    # including empty sets, as in the frozen pre-D decoder.
                    reference.set_hotwords(hotwords)
                    reference._apply_pending_hotwords()
                    ref_lm = BiasingLM(reference._kenlm)
                    ref_lm.hotword_bonus = reference._resolve_hotwords(hotwords)[1]
                    reference.lm = ref_lm
                    reference.init_ngram_decoder()
                    # Restore availability for the real decoder's update.
                    with patch('phoneme_to_words_lm.decoder.HotwordLM', HotwordLM):
                        decoder.set_hotwords(hotwords)
                        x = torch.randn(1, 18, 4, generator=torch.Generator().manual_seed(i))
                        x[:, 4:8, 0] = 25.  # A compressed run across chunk edges.
                        expected = reference.offline_decode(x)[0]
                        got = decoder.offline_decode(x)[0]
                        decoder.online_decode_begin()
                        decoder.online_decode_step(x[:, :6])
                        decoder.set_hotwords({'bee': 3.})  # Deferred through end.
                        decoder.online_decode_step(x[:, 6:])
                        streamed = decoder.online_decode_end()
                    for field in ('word_ids', 'raw_word_seqs', 'word_seqs', 'beam_scores',
                                  'ngram_scores', 'acoustic_scores', 'final_scores', 'search_frames'):
                        self.assertEqual(got[field], expected[field], (log_add, hotwords, field))
                        self.assertEqual(streamed[field], expected[field], (log_add, hotwords, field))


if __name__ == '__main__':
    unittest.main()
