"""Runtime/reference cleanup must preserve words and Unicode contractions."""
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from ngram.text_normalize import normalize_lines
from phoneme_to_words_lm.finetune_llm import preprocess_sentences
from phoneme_to_words_lm.sweep_contract import runtime_identity
from phoneme_to_words_lm.utils import remove_punctuation, replace_words
from sweep.lm_sweep_worker import load_and_filter_logits
from test_decoder import DecoderFixture


def normalize(text):
    return replace_words(remove_punctuation(text))


class TextNormalizationTests(unittest.TestCase):
    def test_experiment_identity_records_normalization_policy(self):
        self.assertEqual(runtime_identity()['text_normalization'], 'english_v3')

    def test_separator_punctuation_never_joins_words(self):
        for separator in [',', '.', '/', '\\', ':', ';', '!', '?', '(', ')',
                          '[', ']', '{', '}', '|', '&', '+', '_', '"', '--', '---']:
            with self.subTest(separator=separator):
                self.assertEqual(normalize('left'+separator+'right'), 'left right')
        for raw, expected in [('left- right', 'left right'), ('left -right', 'left right'),
                              ('-hello-', 'hello'), ('well-known', 'well-known'),
                              ('mother-in-law', 'mother-in-law')]:
            with self.subTest(raw=raw):
                self.assertEqual(normalize(raw), expected)

    def test_quotes_contractions_elisions_and_possessives(self):
        cases = [("he said 'hello' to me", 'he said hello to me'),
                 ('he said ‘hello’ to me', 'he said hello to me'),
                 ("he said 'don't go' to me", "he said don't go to me"),
                 ("the letter 's' is here", 'the letter s is here'),
                 ("the letter ' s ' is here", 'the letter s is here'),
                 ("'hello world' and 'goodbye now'", 'hello world and goodbye now'),
                 ("take a sip of the potion 'cause it's overdue", "take a sip of the potion 'cause it's overdue"),
                 ("give 'em the masters' work", "give 'em the masters' work"),
                 ("we 're ready and I ' m here", "we're ready and i'm here"),
                 ("O'Brien's well-known work", "o'brien's well-known work"),
                 (" ' -- / \" ", '')]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(normalize(raw), expected)
                self.assertEqual(normalize(expected), expected)
                self.assertEqual(preprocess_sentences([raw]), [expected] if expected else [])

    def test_ambiguous_valid_words_remain_distinct(self):
        for left, right in [('he lets me go', "he let's me go"),
                            ('she masters the task', "she master's the task"),
                            ('cant', "can't"), ('wont', "won't"),
                            ("masters'", "master's"), ('were', "we're")]:
            with self.subTest(left=left):
                self.assertEqual(normalize(left), left)
                self.assertNotEqual(normalize(left), normalize(right))

    def test_unicode_contractions_names_boundaries_and_idempotence(self):
        for raw, expected in [("We’re ready", "we're ready"), ('José is here', 'jose is here'),
                              ('Jose\u0301 is here', 'jose is here'),
                              ('left—right – next', 'left right next'),
                              ('first\tsecond\nthird', 'first second third'),
                              ('colour theatre', 'color theater'), ('dont', "don't")]:
            with self.subTest(raw=raw):
                actual = normalize(raw)
                self.assertEqual(actual, expected)
                self.assertEqual(normalize(actual), actual)
                self.assertEqual(preprocess_sentences([raw]), [expected])
                with patch('ngram.text_normalize.sent_tokenize', side_effect=lambda text: [text]):
                    self.assertEqual(normalize_lines(raw), [expected])

    def test_references_from_strings_and_character_ids_match_training(self):
        raw = 'José lets us know we’re ready'
        expected = "jose lets us know we're ready"
        rows = [dict(transcription=ref, logits=np.zeros((1,41)), adjusted_len=1)
                for ref in (raw, np.array([ord(c) for c in raw]+[0]))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'logits.pkl'
            path.write_bytes(pickle.dumps(rows))
            loaded = load_and_filter_logits(dict(logits_pkl_path=str(path)))
        self.assertEqual([row['transcription'] for row in loaded], [expected, expected])

    def test_punctuation_references_match_training(self):
        texts = ["he said 'hello' to me", "potion 'cause it's overdue", 'left/right--next']
        expected = ['he said hello to me', "potion 'cause it's overdue", 'left right next']
        rows = [dict(transcription=ref, logits=np.zeros((1,41)), adjusted_len=1)
                for raw in texts for ref in (raw, np.array([ord(c) for c in raw]+[0]))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'logits.pkl'
            path.write_bytes(pickle.dumps(rows))
            loaded = load_and_filter_logits(dict(logits_pkl_path=str(path)))
        self.assertEqual([row['transcription'] for row in loaded],
                         [text for text in expected for _ in range(2)])
        self.assertEqual(preprocess_sentences(texts), expected)


class NormalizedCandidateTests(DecoderFixture):
    def test_separator_cleanup_keeps_deduplication_and_scores_aligned(self):
        words = ['LEFT/RIGHT', 'LEFT--RIGHT', 'LEFTRIGHT', 'T_A_C']
        Path(self.paths['lexicon_path']).write_text(''.join(word+' A SIL\n' for word in words))
        decoder = self.make()
        ids = [decoder.word_dict.get_index(word) for word in words]
        rows = [SimpleNamespace(words=[word], score=-i, lmScore=0., emittingModelScore=-i)
                for i, word in enumerate(ids)]
        result = decoder._extract_nbest(rows, 3)
        self.assertEqual(result['word_seqs'], ['left right', 'leftright', 't a c'])
        self.assertEqual(result['raw_word_seqs'], [words[i] for i in (0,2,3)])
        self.assertEqual(result['word_ids'], [[ids[i]] for i in (0,2,3)])
        self.assertEqual(result['beam_scores'], [0, -2, -3])

    def test_native_words_stay_separate_in_offline_and_streaming_output(self):
        Path(self.paths['lexicon_path']).write_text("POTION A SIL\n'CAUSE B SIL\n")
        decoder = self.make(log_add=False)
        logits = self.logits([2,1,3,1,0])  # One frame after word completion exposes the partial.
        offline = decoder.offline_decode(logits)[0]
        self.assertEqual(offline['word_seqs'][0], "potion 'cause")
        self.assertEqual(offline['raw_word_seqs'][0], "POTION 'CAUSE")
        decoder.online_decode_begin()
        for frame in logits.unbind(dim=1):
            partial = decoder.online_decode_step(frame.unsqueeze(1))
        self.assertEqual(partial['word_seq'], "potion 'cause")
        online = decoder.online_decode_end()
        for field in ('word_seqs', 'raw_word_seqs', 'word_ids', 'beam_scores'):
            self.assertEqual(online[field], offline[field])

    def test_raw_word_ids_survive_and_distinct_words_use_separate_slots(self):
        words = ['LETS', "LET'S", 'JOSÉ']
        Path(self.paths['lexicon_path']).write_text(''.join(word+' A SIL\n' for word in words))
        decoder = self.make()
        ids = [decoder.word_dict.get_index(word) for word in words]
        rows = [SimpleNamespace(words=[word], score=-i, lmScore=0., emittingModelScore=-i)
                for i, word in enumerate(ids)]
        result = decoder._extract_nbest(rows, 3)
        self.assertEqual(result['word_seqs'], ['lets', "let's", 'jose'])
        self.assertEqual(result['raw_word_seqs'], words)
        self.assertEqual(result['word_ids'], [[word] for word in ids])


if __name__ == '__main__':
    unittest.main()
