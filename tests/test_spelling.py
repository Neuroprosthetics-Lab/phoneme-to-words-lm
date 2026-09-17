"""Spelling corpora and reused models must match the spoken a-z lexicon."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ngram.text_normalize import _normalize_chunk_spelling
from ngram.train_ngram_lm import _concat_normalized, load_and_validate_config, train_ngram_lm


class SpellingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def config(self, corpora, backend='srilm'):
        cfg = dict(output_dir=str(self.root/'output'), lm_type='spelling',
                   interpolator_backend=backend, normalize_workers=1, corpora=corpora,
                   lmplz_path='/bin/true', build_binary_path='/bin/true',
                   srilm_ngram_path='/bin/true', kenlm_interpolate_path='/bin/true')
        path = self.root/'config.json'
        path.write_text(json.dumps(cfg))
        return load_and_validate_config(str(path))

    def arpa(self, name, tokens):
        path = self.root/name
        path.write_text('\\data\\\nngram 1=5\nngram 2=1\n\\1-grams:\n'
                        '-1 <s>\n-1 </s>\n-1 <unk>\n'
                        + ''.join('-1 '+token+'\n' for token in tokens)
                        + '\\2-grams:\n-1 <s> a\n\\end\\\n')
        return str(path)

    def test_raw_apostrophes_accents_repetitions_and_dedup(self):
        text = "Can't cant can’t master's masters' masters José book."
        with patch('ngram.text_normalize.sent_tokenize', side_effect=lambda text: [text]):
            words, _ = _normalize_chunk_spelling([text.encode()])
        self.assertEqual(words, {'c a n t', 'm a s t e r s', 'j o s e', 'b o o k'})

    def test_multiple_normalized_files_canonicalize_spacing_and_dedup(self):
        a, b, output = [self.root/name for name in ('a', 'b', 'merged')]
        a.write_text('c a n t\nb o o k')
        b.write_text('c  a\tn t\nb o o k\n')
        _concat_normalized([str(a), str(b)], str(output), 'spelling')
        self.assertEqual(output.read_text(), 'b o o k\nc a n t\n')

    def test_invalid_single_and_multiple_normalized_files_fail_before_estimation(self):
        good, bad = self.root/'good', self.root/'bad'
        good.write_text('c a n t\n')
        for invalid in ("c a n ' t", 'word', 'c A t', 'a 1'):
            bad.write_text(invalid+'\n')
            for sources in (str(bad), [str(good), str(bad)]):
                with self.subTest(sources=sources, invalid=invalid):
                    with patch('ngram.train_ngram_lm.train_single_lm') as estimate:
                        with self.assertRaisesRegex(ValueError, 'spelling.*a-z'):
                            train_ngram_lm(self.config([dict(name='letters', normalized_path=sources, order=2)]))
                        estimate.assert_not_called()
                    self.assertFalse((self.root/'output/lm_unpruned.bin').exists())

    def test_prebuilt_arpa_rejects_unsupported_vocabulary(self):
        for token in ("'", 'cat', 'A'):
            arpa = self.arpa('bad.arpa', ['a', token])
            with self.subTest(token=token), self.assertRaisesRegex(ValueError, 'spelling.*a-z'):
                self.config([dict(arpa_path=arpa, order=2)])
        arpa = self.arpa('good.arpa', ['a', 'b'])
        self.config([dict(arpa_path=arpa, order=2)])

    def test_prebuilt_intermediate_rejects_unsupported_vocabulary(self):
        prefix = self.root/'intermediate'
        Path(str(prefix)+'.kenlm_intermediate').write_text('Counts 5 1\n')
        for order in (1, 2):
            Path(str(prefix)+f'.{order}').write_bytes(b'fixture')
        vocab = Path(str(prefix)+'.vocab')
        text = self.root/'good'
        text.write_text('a b\n')
        corpora = [dict(intermediate_path=str(prefix), order=2, weight=1),
                   dict(normalized_path=str(text), order=2, weight=1)]
        vocab.write_bytes(b'<unk>\x00<s>\x00</s>\x00a\x00\x27\x00')
        with self.assertRaisesRegex(ValueError, 'spelling.*a-z'):
            self.config(corpora, 'kenlm')
        vocab.write_bytes(b'<unk>\x00<s>\x00</s>\x00a\x00b\x00')
        self.config(corpora, 'kenlm')

    def test_estimated_vocabulary_is_checked_before_binary_compilation(self):
        text = self.root/'letters'
        text.write_text('c a n t\n')
        bad_arpa = self.arpa('estimated.arpa', ['a', "'"])
        def estimate(**kwargs):
            Path(kwargs['output_path']).write_text(Path(bad_arpa).read_text())
        with patch('ngram.train_ngram_lm.train_single_lm', side_effect=estimate):
            with patch('ngram.train_ngram_lm.compile_to_binary') as compile_binary:
                with self.assertRaisesRegex(ValueError, 'spelling.*a-z'):
                    train_ngram_lm(self.config([dict(normalized_path=str(text), order=2)]))
                compile_binary.assert_not_called()


if __name__ == '__main__':
    unittest.main()
