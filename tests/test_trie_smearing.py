"""MAX lookahead must not depend on the number of homophones."""
import math
import unittest

from flashlight.lib.text.decoder import SmearingMode, Trie


class TrieSmearingTests(unittest.TestCase):
    def test_max_terminal_scores_ignore_multiplicity(self):
        for count in (1, 2, 60):
            with self.subTest(count=count):
                trie = Trie(4, 0)
                for label in range(count):
                    trie.insert([1, 2], label, -3.)
                trie.smear(SmearingMode.MAX)
                self.assertEqual(trie.search([1, 2]).max_score, -3.)
                self.assertEqual(trie.search([1]).max_score, -3.)

    def test_max_combines_terminals_and_descendants_in_any_order(self):
        entries = [([1], -4.), ([1], -3.), ([1, 2], -2.), ([1, 3], -5.)]
        for values in (entries, list(reversed(entries))):
            trie = Trie(4, 0)
            for label, (tokens, score) in enumerate(values):
                trie.insert(tokens, label, score)
            trie.smear(SmearingMode.MAX)
            self.assertEqual(trie.search([1]).max_score, -2.)
            self.assertEqual(trie.search([1, 3]).max_score, -5.)

    def test_max_handles_impossible_and_positive_infinite_labels(self):
        for scores, expected in [([-math.inf, -math.inf], -math.inf),
                                 ([-math.inf, -3.], -3.), ([math.inf, math.inf], math.inf)]:
            trie = Trie(3, 0)
            for label, score in enumerate(scores):
                trie.insert([1], label, score)
            trie.smear(SmearingMode.MAX)
            self.assertEqual(trie.search([1]).max_score, expected)

    def test_logadd_and_none_modes_keep_their_contracts(self):
        trie = Trie(4, 0)
        for label, tokens in enumerate(([1], [1], [1, 2])):
            trie.insert(tokens, label, -3.)
        trie.smear(SmearingMode.LOGADD)
        expected = -3. + math.log(3)
        self.assertAlmostEqual(trie.search([1]).max_score, expected, places=6)
        trie.smear(SmearingMode.NONE)
        self.assertAlmostEqual(trie.search([1]).max_score, expected, places=6)


if __name__ == '__main__':
    unittest.main()
