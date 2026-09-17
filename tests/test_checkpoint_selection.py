"""Exercise final checkpoint selection through the actual training entry point."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from phoneme_to_words_lm import finetune_llm as training
from test_llm_scoring import CharacterTokenizer


class Dataset:
    @classmethod
    def from_list(cls, values):
        return cls()

    def shuffle(self, **kwargs):
        return self


class Trainer:
    def __init__(self, model, callbacks, **kwargs):
        self.model, self.callbacks = model, callbacks
        self.state = SimpleNamespace(global_step=0)

    def train(self):
        for step in (1, 2, 3):
            self.model.step = self.state.global_step = step
            for callback in self.callbacks:
                callback.on_step_end(None, self.state, None, self.model)
        return SimpleNamespace(metrics={})


class CheckpointSelectionTests(unittest.TestCase):
    def run_training(self, scheduled_ppl, final_ppl, *, validation=True, eval_steps=2):
        model = SimpleNamespace(step=0, config=SimpleNamespace(use_cache=True),
                                named_modules=lambda: [('q_proj', None)])
        tokenizer = CharacterTokenizer()
        tokenizer.pad_token = 'pad'
        evaluations = []

        def report(model, *args, **kwargs):
            evaluations.append(model.step)
            ppl = scheduled_ppl if model.step == 2 else final_ppl
            return dict(overall=dict(nll=1., target_tokens=1, perplexity=ppl), by_source={})

        def save(model, tokenizer, directory, **kwargs):
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            (directory/'saved_step.json').write_text(json.dumps(model.step))

        modules = {
            'datasets': SimpleNamespace(Dataset=Dataset),
            'peft': SimpleNamespace(LoraConfig=lambda **kw: kw, TaskType=SimpleNamespace(CAUSAL_LM='causal')),
            'trl': SimpleNamespace(SFTTrainer=Trainer, SFTConfig=lambda **kw: kw),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'sentences.txt'
            source.write_text('hello world\nhello there\ncat sat\ndog runs\n')
            output = root/'output'
            argv = ['finetune', '--source-files', 'text:'+str(source), '--output-dir', str(output),
                    '--dtype', 'float32', '--max-steps', '3', '--eval-steps', str(eval_steps),
                    '--val-fraction', '.25' if validation else '0']
            with contextlib.ExitStack() as stack:
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                stack.enter_context(patch.object(sys, 'argv', argv))
                stack.enter_context(patch.dict(sys.modules, modules))
                stack.enter_context(patch.object(training.AutoTokenizer, 'from_pretrained', return_value=tokenizer))
                stack.enter_context(patch.object(training.AutoModelForCausalLM, 'from_pretrained', return_value=model))
                stack.enter_context(patch.object(training, 'validation_report', side_effect=report))
                stack.enter_context(patch.object(training, 'save_adapter', side_effect=save))
                stack.enter_context(patch.object(training, 'verify_adapter_reload', return_value={}))
                stack.enter_context(patch('torch.cuda.is_available', return_value=False))
                stack.enter_context(patch('importlib.metadata.version', return_value='fixture'))
                training.main()
            results = json.loads((output/'finetuning_results.json').read_text())
            self.assertEqual(results['text_normalization'], 'english_v3')
            selected_step = json.loads((output/'saved_step.json').read_text())
            self.assertEqual(json.loads((output/'final/saved_step.json').read_text()), 3)
            return results, selected_step, evaluations

    def test_final_improvement_regression_tie_and_missing_validation(self):
        for previous, final, expected in [(10., 5., 3), (5., 10., 2), (5., 5., 2), (None, 5., 3), (None, None, 3)]:
            with self.subTest(previous=previous, final=final):
                result, step, calls = self.run_training(previous, final)
                self.assertEqual(step, expected)
                self.assertEqual(calls.count(3), 1)
                if previous is not None or final is not None:
                    self.assertEqual(result['best_validation']['overall']['perplexity'], min(p for p in (previous, final) if p is not None))
                else:
                    self.assertEqual(result['selected_adapter'], 'final')

    def test_final_step_already_evaluated_is_not_scored_twice(self):
        result, step, calls = self.run_training(10., 5., eval_steps=3)
        self.assertEqual(step, 3)
        self.assertEqual(calls, [0, 3])
        self.assertEqual(result['best_validation'], result['final_validation'])

    def test_training_without_validation_still_publishes_final(self):
        result, step, _ = self.run_training(10., 5., validation=False)
        self.assertEqual(step, 3)
        self.assertEqual(result['selected_adapter'], 'final')
        self.assertIsNone(result['best_validation'])


if __name__ == '__main__':
    unittest.main()
