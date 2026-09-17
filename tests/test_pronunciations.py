"""Repeated resource rows must not create extra decoding paths."""
from pathlib import Path
import unittest

from test_decoder import DecoderFixture
from ngram.train_ngram_lm import generate_lexicon


class PronunciationTests(DecoderFixture):
    def test_duplicate_rows_preserve_scores_and_hotword_rebuilds(self):
        path = Path(self.paths['lexicon_path'])
        original = path.read_text()
        for log_add in (False, True):
            for hotwords in ({}, {'single': 2.}):
                with self.subTest(log_add=log_add, hotwords=hotwords):
                    outputs = []
                    for copies in (1, 2, 100):
                        path.write_text(original + 'SINGLE A SIL\n' * (copies-1))
                        decoder = self.make(log_add=log_add, lm_weight=1.)
                        decoder.set_hotwords(hotwords)
                        result = decoder.offline_decode(self.logits([2, 1]))[0]
                        outputs.append(result)
                        self.assertEqual(decoder.lexicon['SINGLE'], [['A', 'SIL']])
                    for result in outputs[1:]:
                        for field in ('word_seqs', 'beam_scores', 'acoustic_scores', 'ngram_scores'):
                            self.assertEqual(result[field], outputs[0][field])

    def test_distinct_pronunciations_homophones_and_repeated_phones_survive(self):
        Path(self.paths['lexicon_path']).write_text(
            'SINGLE A SIL\nSINGLE B SIL\nSINGLE A SIL\n'
            'TWIN A SIL\nDOUBLE A A SIL\n')
        decoder = self.make()
        self.assertEqual(decoder.lexicon['SINGLE'], [['A', 'SIL'], ['B', 'SIL']])
        self.assertIn('single', decoder.offline_decode(self.logits([3, 1]))[0]['word_seqs'])
        result = decoder.offline_decode(self.logits([2, 1]))[0]
        self.assertTrue({'single', 'twin'}.issubset(result['word_seqs']))
        self.assertEqual(decoder.offline_decode(self.logits([2, 0, 2, 1]))[0]['word_seqs'][0], 'double')

    def test_generated_lexicon_deduplicates_complete_rows(self):
        path = Path(self.tmp.name)/'generated'
        phones = {'word': [['AA', 'AA'], ['AA', 'AA'], ['B']], 'twin': [['AA', 'AA']]}
        generate_lexicon('word', {'WORD', 'TWIN'}, phones, str(path))
        self.assertEqual(path.read_text().splitlines(),
                         ['TWIN\tAA AA SIL', 'WORD\tAA AA SIL', 'WORD\tB SIL'])


if __name__ == '__main__':
    unittest.main()
