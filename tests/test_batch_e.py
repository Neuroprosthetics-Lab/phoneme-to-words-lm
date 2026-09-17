"""Batch E: configuration, dataset isolation, metrics and artifact regressions."""
import importlib
import json
import math
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import optuna
from omegaconf import OmegaConf

from phoneme_to_words_lm.sweep_contract import alpha_values, resolve_decoder_config, verify_resume, file_identity, runtime_identity
from phoneme_to_words_lm.finetune_llm import (prepare_dataset, optimizer_schedule,
    validation_report, PerplexityCallback, save_adapter)
from test_llm_scoring import TinyModel, CharacterTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'sweep'))
from lm_sweep import load_lm_sweep_config, make_lm_objective, WorkerExecutionError, parse_device_specs
from lm_sweep_worker import load_and_filter_logits, alpha_posthoc_sweep, decode_sentences
from ngram.train_ngram_lm import (_concat_normalized, arpa_vocabulary, generate_lexicon,
                                load_and_validate_config, train_ngram_lm)


class FileFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, name, text):
        path = self.root/name
        path.write_text(text)
        return str(path)


class SweepTests(FileFixture):
    def config(self, **extra):
        file = self.write('resource', 'fixture')
        result = dict.fromkeys(['lexicon_path','tokens_path','kenlm_model_path','logits_pkl_path'], file)
        result.update(output_dir=str(self.root/'out'), do_llm_rescoring=False,
                      parameters={'beam_size': {'type':'fixed','value':5}})
        result.update(extra)
        path = self.write('config.yaml', json.dumps(result))
        return load_lm_sweep_config(path)

    def test_full_effective_config_and_paths(self):
        adapter = self.root/'adapter'; adapter.mkdir()
        cfg = self.config(llm_lora_path=str(adapter), hotwords={'the': .2}, sil_score=-2.,
                          word_score=1.2, unk_score=-10., log_add=False)
        effective = resolve_decoder_config(OmegaConf.to_container(cfg, resolve=True))
        for key in ('llm_lora_path','hotwords','sil_score','word_score','unk_score','log_add'):
            self.assertEqual(effective[key], OmegaConf.to_container(cfg, resolve=True)[key])
        self.assertEqual(effective['beam_size'], 5)
        self.assertTrue(Path(effective['llm_cache_dir']).is_absolute())
        with patch.dict('os.environ', {'HOME': str(self.root)}):
            self.assertEqual(resolve_decoder_config(dict(cfg, lexicon_path='~/resource'))['lexicon_path'], str(self.root/'resource'))

    def test_config_rejects_invalid_settings_before_worker(self):
        cases = [dict(eval_every_nth=0),dict(max_sentences=True),dict(alpha_range={'step':0}),
                 dict(alpha_range={'low':-1}),dict(objective_weights={'wer':0,'time':0}),
                 dict(objective_weights={'wer':math.nan}),dict(typo=1),dict(log_add='false'),
                 dict(llm_alpha=.5),dict(alpha_range={'steps':.1}),
                 dict(parameters={'typo': {'type':'fixed','value':1}}),
                 dict(parameters={'beam_size': {'type':'float_range','low':1.5,'high':5}}),
                 dict(parameters={'temperature': {'type':'float_range','low':0,'high':1}}),
                 dict(parameters={'temperature': {'type':'float_range','low':1,'high':2,'step':0}})]
        for case in cases:
            with self.subTest(case=case), self.assertRaises((ValueError, TypeError)):
                self.config(**case)

    def test_device_validation_and_alpha_grid(self):
        for value in ('cuda:-1','cuda:0:1:2','cuda:0,cuda:0','cuda:0:0'):
            with self.assertRaises(ValueError): parse_device_specs(value)
        self.assertEqual(alpha_values(0, 1, .6), [0., .6])

    def test_encoded_transcriptions_lengths_filtering_and_context(self):
        row=dict(logits=np.zeros((4,4),np.float32),adjusted_len=3,
                 transcription=np.array([ord(c) for c in 'hello']+[0,0]),context='previous sentence')
        rows=[row,dict(row,transcription='a b'),dict(row,adjusted_lens=3,transcription='hello there')]
        path=self.root/'logits.pkl';path.write_bytes(pickle.dumps(rows))
        selected=load_and_filter_logits(dict(logits_pkl_path=str(path)))
        self.assertEqual([s['source_index'] for s in selected],[0,2])
        self.assertEqual(selected[0]['transcription'],'hello')
        class Decoder:
            def offline_decode(self, logits, lengths, contexts):
                self.contexts=contexts
                return [dict(word_seqs=['hello'],beam_scores=[-1.],raw_llm_scores=[None],
                             llm_token_counts=[0],llm_scores=[None])]
        decoder=Decoder()
        result=decode_sentences(decoder,selected[:1],True)
        self.assertEqual(decoder.contexts,['previous sentence'])
        self.assertEqual(result[0]['edit_counts_by_text'],{'hello':0})
        rows[0]['adjusted_lens']=2;path.write_bytes(pickle.dumps(rows))
        with self.assertRaisesRegex(ValueError,'conflicting'):load_and_filter_logits(dict(logits_pkl_path=str(path)))

    def test_edit_counts_reused_and_raw_length_penalty(self):
        row=dict(transcription='right',word_seqs=['right','wrong'],beam_scores=[-1.,-2.],
                 llm_scores=[-2.,0.],raw_llm_scores=[-2.,0.],llm_token_counts=[1,2])
        import editdistance
        with patch('lm_sweep_worker.editdistance.eval',wraps=editdistance.eval) as distance:
            alpha_posthoc_sweep([row],1,0,3,.01)
            self.assertEqual(distance.call_count,2)
            alpha_posthoc_sweep([row],1,0,3,.01,length_penalty=.5)
            self.assertEqual(distance.call_count,2)
        self.assertEqual(alpha_posthoc_sweep([row],1,1,1,.1,length_penalty=2)[1],0.)

    def test_failed_worker_is_failed_not_pruned_and_releases_device(self):
        cfg=self.config(); released=[]
        pool=SimpleNamespace(acquire=lambda:0,release=released.append)
        objective=make_lm_objective(cfg,pool,'worker.py',{0:1})
        study=optuna.create_study()
        with patch('lm_sweep.subprocess.run',return_value=SimpleNamespace(returncode=1,stdout='',stderr='fixture failure')):
            study.optimize(objective,n_trials=1,catch=(WorkerExecutionError,))
        self.assertEqual(study.trials[0].state,optuna.trial.TrialState.FAIL)
        self.assertEqual(released,[0])

    def test_resume_identity_and_content_hash(self):
        study=optuna.create_study()
        verify_resume(study,dict(sha256='a'),resume=False)
        verify_resume(study,dict(sha256='a'),resume=True)
        with self.assertRaisesRegex(ValueError,'Cannot resume'):
            verify_resume(study,dict(sha256='b'),resume=True)
        path=self.write('data','same size')
        before=file_identity(path)
        Path(path).write_text('new bytes')
        self.assertNotEqual(before['sha256'],file_identity(path)['sha256'])

    def test_runtime_identity_includes_optional_adapter_dependency(self):
        def version(name):
            if name == 'peft':
                raise importlib.metadata.PackageNotFoundError(name)
            return '1'
        with patch('phoneme_to_words_lm.sweep_contract.importlib.metadata.version', side_effect=version):
            missing = runtime_identity()
        with patch('phoneme_to_words_lm.sweep_contract.importlib.metadata.version', return_value='1'):
            installed = runtime_identity()
        self.assertIsNone(missing['peft'])
        self.assertEqual(installed['peft'], '1')
        self.assertNotEqual(missing, installed)

    def test_timeout_and_malformed_results_release_device(self):
        for timeout in (True, False):
            with self.subTest(timeout=timeout):
                cfg = self.config(worker_timeout_seconds=1)
                released = []
                pool = SimpleNamespace(acquire=lambda: 0, release=released.append)
                objective = make_lm_objective(cfg, pool, 'worker.py', {0:1})
                def worker(*args, **kwargs):
                    if timeout:
                        raise subprocess.TimeoutExpired('worker', 1, output=b'partial log')
                    Path(cfg.output_dir, 'trial_0', 'results.json').write_text('{bad json')
                    return SimpleNamespace(returncode=0, stdout='', stderr='')
                study = optuna.create_study()
                with patch('lm_sweep.subprocess.run', side_effect=worker):
                    study.optimize(objective, n_trials=1, catch=(WorkerExecutionError,))
                self.assertEqual(study.trials[0].state, optuna.trial.TrialState.FAIL)
                self.assertEqual(released, [0])


class TrainingTests(FileFixture):
    def test_global_split_before_upsampling_is_disjoint(self):
        text='\n'.join('a sentence '+chr(97+i//26)+chr(97+i%26) for i in range(100))
        a=self.write('a',text);b=self.write('b',text.upper())
        train,val,stats,by_source=prepare_dataset({'a':a,'b':b},{'a':2},.2)
        self.assertFalse(set(train)&set(val))
        self.assertEqual(len(val),20)
        self.assertEqual(len(train),240)
        self.assertEqual(by_source['a'],by_source['b'])
        again=prepare_dataset({'b':b,'a':a},{'a':2},.2)
        self.assertEqual((train,val),(again[0],again[1]))

    def test_tiny_and_empty_datasets_and_factors(self):
        path=self.write('one','hello')
        with self.assertRaisesRegex(ValueError,'two distinct'):
            prepare_dataset({'a':path},{},.1)
        train,val,*_=prepare_dataset({'a':path},{},0)
        self.assertEqual((train,val),(['hello'],[]))
        for factor in (0,-1,1.5,True):
            with self.assertRaises(ValueError):prepare_dataset({'a':path},{'a':factor},0)
        for fraction in (-.1,1.,math.nan):
            with self.assertRaises(ValueError):prepare_dataset({'a':path},{},fraction)
        with self.assertRaises(ValueError):prepare_dataset({'a':path},{'unknown':1},0)
        Path(path).write_text('')
        with self.assertRaisesRegex(ValueError,'No usable'):prepare_dataset({'a':path},{},0)

    def test_partial_optimizer_batches_and_schedule(self):
        self.assertEqual(optimizer_schedule(3,8,4,2,.25),
                         dict(steps_per_epoch=1,total_steps=2,eval_steps=1))

    def test_validation_scores_unique_sentences_only_once(self):
        model=TinyModel();model.train();tok=CharacterTokenizer()
        from phoneme_to_words_lm.llm_scoring import score_sentences
        with patch('phoneme_to_words_lm.finetune_llm.score_sentences',wraps=score_sentences) as score:
            result=validation_report(model,tok,['a','abc','a'],{'x':['a'],'y':['a','abc']})
            self.assertEqual(score.call_count,1)
            self.assertEqual(score.call_args.args[2],['a','abc'])
        self.assertEqual(result['overall'],result['by_source']['y'])
        self.assertTrue(model.training)

    def test_failed_adapter_save_keeps_previous_files(self):
        target=self.root/'adapter';target.mkdir();(target/'adapter_model.safetensors').write_text('old')
        model=SimpleNamespace(save_pretrained=lambda path: (_ for _ in ()).throw(RuntimeError('save failed')))
        with self.assertRaisesRegex(RuntimeError,'save failed'):
            save_adapter(model,None,target,prefix='\n',score_eos=False)
        self.assertEqual((target/'adapter_model.safetensors').read_text(),'old')


class NgramTests(FileFixture):
    def test_missing_normalization_dependencies_fail_without_touching_output(self):
        from ngram import text_normalize as normalize
        text = self.write('raw', 'There are 12 cats.')
        output = self.write('normalized', 'previous corpus')
        with patch.object(normalize, 'num2words', side_effect=ImportError('missing num2words')):
            for function in (normalize._ordinal_words_cached, normalize._decade_words_cached,
                             normalize._number_words_cached):
                function.cache_clear()
                with self.assertRaisesRegex(ImportError, 'num2words'):
                    function('1990')
            with self.assertRaises(ImportError):
                normalize.normalize_corpus(text, output, 'word', workers=2)
        self.assertEqual(Path(output).read_text(), 'previous corpus')

    def test_concatenation_preserves_sentence_boundaries_and_empty_files(self):
        inputs=[self.write('a','hello world'),self.write('empty',''),self.write('b','good morning\n')]
        out=str(self.root/'joined')
        _concat_normalized(inputs,out,'word')
        self.assertEqual(Path(out).read_text(),'hello world\ngood morning\n')
        _concat_normalized([self.write('c','a b\na b'),self.write('d','c d')],out,'spelling')
        self.assertEqual(Path(out).read_text(),'a b\nc d\n')

    def test_arpa_vocabulary_preserves_native_case_and_lexicon_lookup(self):
        arpa=self.write('model.arpa','\\data\\\nngram 1=4\nngram 2=1\n\\1-grams:\n-1 <s>\n-1 </s>\n-1 <unk>\n-1 HELLO\n\\2-grams:\n')
        self.assertEqual(arpa_vocabulary(arpa,2),{'HELLO'})
        with self.assertRaisesRegex(ValueError,'differs'):arpa_vocabulary(arpa,3)
        out=self.root/'lexicon'
        generate_lexicon('word',{'HELLO'},{'hello':[['HH','AH','L','OW']]},str(out))
        self.assertEqual(out.read_text(),'HELLO\tHH AH L OW SIL\n')
        with self.assertRaisesRegex(ValueError,'No usable'):generate_lexicon('word',{'x'},{},str(out))

    def test_invalid_pruning_and_intermediate_fail_early(self):
        text=self.write('text','hello world\n')
        cfg=dict(output_dir=str(self.root/'out'),lm_type='word',interpolator_backend='kenlm',
                 corpora=[dict(path=text,order=3,pruning=[0,1,0])])
        path=self.write('cfg.yaml',json.dumps(cfg))
        with self.assertRaisesRegex(ValueError,'nondecreasing'):load_and_validate_config(path)
        cfg['corpora']=[dict(intermediate_path='unused',order=3)]
        Path(path).write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError,'Single prebuilt intermediate'):load_and_validate_config(path)
        cfg['corpora']=[dict(path=text,order=2,weight=1),dict(path=text,order=3,weight=1)]
        Path(path).write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError,'equal corpus orders'):load_and_validate_config(path)

    def test_failed_ngram_build_keeps_previous_complete_artifacts(self):
        out=self.root/'out';out.mkdir();(out/'lm_unpruned.bin').write_text('old')
        (out/'build_manifest.json').write_text('old manifest')
        text=self.write('text','hello')
        cfg=OmegaConf.create(dict(output_dir=str(out),interpolator_backend='srilm',
            corpora=[dict(normalized_path=text,order=2)], lmplz_path='unused',
            build_binary_path='unused',srilm_ngram_path='unused',kenlm_interpolate_path='unused'))
        with patch('ngram.train_ngram_lm._train_ngram_lm',side_effect=RuntimeError('tool failed')):
            with self.assertRaisesRegex(RuntimeError,'tool failed'):train_ngram_lm(cfg)
        self.assertEqual((out/'lm_unpruned.bin').read_text(),'old')
        self.assertEqual((out/'build_manifest.json').read_text(),'old manifest')

    def test_lightweight_imports_do_not_load_native_or_dictionary(self):
        code="import sys; import phoneme_to_words_lm as p; import phoneme_to_words_lm.utils as u; assert 'torch' not in sys.modules; assert 'flashlight' not in sys.modules; assert 'cmu_dict' not in u.__dict__; assert p.remove_punctuation('Hello!') == 'hello'"
        subprocess.run([sys.executable,'-B','-c',code],check=True)
        code="import sys; import ngram.text_normalize; assert 'nltk' not in sys.modules; assert 'num2words' not in sys.modules"
        subprocess.run([sys.executable,'-B','-c',code],check=True)


if __name__=='__main__':unittest.main()
