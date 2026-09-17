"""Batch G: split integrity, validation-only selection and repeatable reports."""
import copy
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from phoneme_to_words_lm.evaluation import (bootstrap_interval, choose_finalists,
    freeze_selection, group_key, objective, paired_comparison, pareto_front,
    percentile_stats, prepare_ranking, rank, select_weights, split_indices, subset_indices)
from benchmarks.run_evaluation import (build_report, digest, export_configs, load_config,
    load_selection, prepare_data, save_selection, verified_summary, write_json)


class EvaluationTests(unittest.TestCase):
    def test_objective_matches_requested_tradeoff(self):
        weights = {'wer':1000., 'latency':1.}
        self.assertAlmostEqual(objective(.011,.1,weights),11.1)
        self.assertLess(objective(.01,1.,weights),objective(.011,.1,weights))
        self.assertAlmostEqual(objective(.01,1.1,weights),objective(.011,.1,weights))

    def test_subsets_are_deterministic_and_use_original_indices(self):
        rows = [{'source_index':i*7} for i in range(20)]
        self.assertEqual(subset_indices(rows,5,42),subset_indices(rows,5,42))
        self.assertEqual(len(subset_indices(rows,None,42)),20)
        self.assertTrue(set(subset_indices(rows,5,42)) <= {r['source_index'] for r in rows})
        for invalid in (True,0,-1):
            with self.assertRaises(ValueError): subset_indices(rows,invalid,42)

    def test_group_split_joins_duplicate_references_across_days(self):
        rows = [dict(source_index=i,day_index=i//2,transcription=f'sentence {i}') for i in range(12)]
        rows[2]['transcription'] = rows[0]['transcription']
        validation,test = split_indices(rows,.4,42,'day_index')
        self.assertFalse(set(validation)&set(test))
        self.assertEqual(set(validation)|set(test),set(range(12)))
        self.assertEqual(0 in validation,2 in validation)
        self.assertFalse({rows[i]['day_index'] for i in validation}&{rows[i]['day_index'] for i in test})
        self.assertFalse({rows[i]['transcription'] for i in validation}&{rows[i]['transcription'] for i in test})
        self.assertEqual((validation,test),split_indices(rows,.4,42,'day_index'))

    def test_unusable_group_splits_and_missing_metadata_fail(self):
        rows = [dict(source_index=i,transcription='same prompt') for i in range(4)]
        with self.assertRaisesRegex(ValueError,'independent'):split_indices(rows,.2,42)
        with self.assertRaisesRegex(ValueError,'Missing'):group_key(rows[0],'day_index')
        self.assertEqual(group_key(rows[0],'utterance'),'0')

    def row(self):
        return dict(transcription='correct word',word_seqs=['wrong word','correct word'],
                    beam_scores=[-2.,-1.],raw_llm_scores=[0.,-2.],llm_token_counts=[1,2])

    def test_ranking_preserves_native_ties_and_raw_length_penalty(self):
        prepared = prepare_ranking([self.row()])
        self.assertEqual(rank(prepared,.5,0),([0],[1]))
        self.assertEqual(rank(prepared,1.,0),([1],[0]))
        self.assertEqual(rank(prepared,1.,-2),([0],[1]))
        best,curve = select_weights(prepared,[0.,.25,.5,1.],[-.25,0,.25],(.45,0))
        self.assertEqual(best,dict(alpha=.5,length_penalty=0,edits=0))
        self.assertEqual(len(curve),12)

    def test_empty_and_no_llm_results_have_defined_errors(self):
        empty = dict(transcription='two words',word_seqs=[''],beam_scores=[-math.inf],
                     raw_llm_scores=[None],llm_token_counts=[0])
        self.assertEqual(rank(prepare_ranking([empty]),0,0),([2],[0]))
        row = self.row();row['raw_llm_scores']=[None,None];row['llm_token_counts']=[0,0]
        self.assertEqual(rank(prepare_ranking([row]),1,0),([0],[1]))

    def test_default_tuning_anchor_matches_decoder(self):
        from phoneme_to_words_lm.sweep_contract import decoder_defaults
        row = self.row()
        row['raw_llm_scores'] = [None, None]
        row['llm_token_counts'] = [0, 0]
        best, _ = select_weights(prepare_ranking([row]), [.45, .55], [0.])
        self.assertEqual(best['alpha'], decoder_defaults()['llm_alpha'])

    def test_edit_counts_are_computed_once_before_policy_grid(self):
        import editdistance
        with patch('editdistance.eval',wraps=editdistance.eval) as edit:
            arrays=prepare_ranking([self.row()])
            select_weights(arrays,np.arange(0,2,.05),[-.25,0,.25])
            self.assertEqual(edit.call_count,2)

    def results(self,stage='validation'):
        return [dict(name=name,stage=stage,wer=wer,timing={'mean_seconds':seconds}) for name,wer,seconds in
                [('baseline',.02,.2),('accurate',.01,.9),('balanced',.011,.1),('fast',.018,.05),('dominated',.03,.3)]]

    def test_selection_uses_validation_and_keeps_accuracy_speed_baseline(self):
        rows=self.results();weights={'wer':1000.,'latency':1.}
        self.assertEqual({r['name'] for r in pareto_front(rows)},{'accurate','balanced','fast'})
        self.assertEqual(choose_finalists(rows,'baseline',4,weights),['baseline','accurate','fast','balanced'])
        self.assertEqual(freeze_selection(rows,'baseline',weights),dict(recommended='accurate',accuracy='accurate',fastest='fast',baseline='baseline'))
        with self.assertRaisesRegex(ValueError,'Test'):choose_finalists(self.results('test'),'baseline',4,weights)
        with self.assertRaisesRegex(ValueError,'full validation'):freeze_selection(self.results('test'),'baseline',weights)

    def test_bootstrap_resamples_groups_and_paired_differences(self):
        interval=bootstrap_interval([1,0,2],[10,10,10],['a','a','b'],500,42)
        self.assertEqual(interval['groups'],2)
        self.assertEqual(interval,bootstrap_interval([1,0,2],[10,10,10],['a','a','b'],500,42))
        self.assertIsNone(bootstrap_interval([1],[10],['a'])['low'])
        base=[dict(source_index=i,transcription='same words',evaluation_group=str(i),selected_edits=1,n_words=2) for i in range(3)]
        new=copy.deepcopy(base);new[0]['selected_edits']=0
        paired=paired_comparison(new,base,100,42)
        self.assertAlmostEqual(paired['wer_difference'],-1/6)
        self.assertEqual(paired['improved'],1)
        with self.assertRaisesRegex(ValueError,'indices'):paired_comparison(new[::-1],base)

    def test_timing_statistics_and_invalid_samples(self):
        result=percentile_stats([.1,.2,.3])
        self.assertAlmostEqual(result['mean_seconds'],.2)
        self.assertEqual(result['p50_seconds'],.2)
        self.assertAlmostEqual(result['utterances_per_second'],5.)
        for values in ([],[0],[math.nan],[-1]):
            with self.assertRaises(ValueError):percentile_stats(values)


class EvaluationFilesTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        resource=self.root/'resource';resource.write_text('fixture')
        self.config=dict(output_dir=str(self.root/'out'),validation_logits=str(self.root/'val.pkl'),
                         test_logits=str(self.root/'test.pkl'),bootstrap_group='day_index',
                         decoder={**dict.fromkeys(['lexicon_path','tokens_path','kenlm_model_path'],str(resource)),
                                  'do_llm_rescoring':False,'llm_alpha':.45},
                         configurations=[dict(name='baseline',tune=False),dict(name='retuned')])
        rows=[dict(source_index=i,day_index=i//2,logits=np.zeros((3,4),np.float32),adjusted_len=3,
                   transcription='example '+chr(97+i)+chr(97+i)) for i in range(8)]
        (self.root/'val.pkl').write_bytes(pickle.dumps(rows))
        (self.root/'test.pkl').write_bytes(pickle.dumps([dict(r,transcription=r['transcription']+' test') for r in rows]))

    def load(self,**changes):
        path=self.root/'config.json';path.write_text(json.dumps({**self.config,**changes}))
        return load_config(path)

    def test_config_validates_paths_names_and_units(self):
        cfg,cases=self.load()
        self.assertEqual(cases['baseline']['decoder']['llm_alpha'],.45)
        self.assertEqual(cfg['objective'],{'wer':1000.,'latency':1.})
        for changes in [dict(typo=1),dict(objective={'wer':0,'latency':0}),dict(finalists=2),
                        dict(length_penalties=[math.nan]),dict(configurations=[dict(name='../bad')]),
                        dict(configurations=[dict(name='baseline',tune=True)])]:
            with self.subTest(changes=changes),self.assertRaises((ValueError,TypeError)):
                self.load(**changes)

    def test_separate_files_and_one_file_split_are_manifested(self):
        cfg,_=self.load();data=prepare_data(cfg)
        self.assertEqual(data['split'],'separate_files')
        self.assertEqual(data['shared_reference_identities'],0)
        cfg,_=self.load(test_logits=None,split_group='day_index')
        data=prepare_data(cfg)
        self.assertFalse(set(data['validation_indices'])&set(data['test_indices']))
        self.assertEqual(data['shared_reference_identities'],0)
        cfg,_=self.load(test_logits=self.config['validation_logits'])
        with self.assertRaisesRegex(ValueError,'identical'):prepare_data(cfg)

    def test_omitted_alpha_and_explicit_overrides_match_decoder_ranking(self):
        from phoneme_to_words_lm.sweep_contract import decoder_defaults
        decoder = dict(self.config['decoder'])
        del decoder['llm_alpha']
        entries = [dict(name='baseline', tune=False),
                   dict(name='zero', overrides={'llm_alpha': 0.}),
                   dict(name='explicit', overrides={'llm_alpha': .45})]
        _, cases = self.load(decoder=decoder, configurations=entries)
        self.assertEqual(cases['baseline']['decoder']['llm_alpha'], decoder_defaults()['llm_alpha'])
        self.assertFalse(cases['baseline']['tune'])
        self.assertEqual(cases['zero']['decoder']['llm_alpha'], 0.)
        self.assertEqual(cases['explicit']['decoder']['llm_alpha'], .45)
        row = dict(transcription='correct', word_seqs=['correct', 'other'],
                   beam_scores=[-1., -2.], raw_llm_scores=[-2., 0.], llm_token_counts=[1, 1])
        arrays = prepare_ranking([row])
        self.assertEqual(rank(arrays, cases['baseline']['decoder']['llm_alpha'], 0.)[1], [1])
        self.assertEqual(rank(arrays, cases['explicit']['decoder']['llm_alpha'], 0.)[1], [0])

    def test_frozen_selection_detects_mutation_or_identity_change(self):
        value=dict(experiment_sha256='a',roles={'recommended':'baseline'},configurations={})
        save_selection(self.root,value)
        self.assertEqual(load_selection(self.root,'a'),value)
        with self.assertRaises(ValueError):load_selection(self.root,'b')
        with self.assertRaisesRegex(ValueError,'frozen'):save_selection(self.root,{**value,'roles':{}})
        p=self.root/'selection.json';edited=json.loads(p.read_text());edited['roles']={};p.write_text(json.dumps(edited))
        with self.assertRaises(ValueError):load_selection(self.root,'a')

    def test_completion_marker_checks_job_and_output_content(self):
        job={'a':1};write_json(self.root/'job.json',job);write_json(self.root/'summary.json',{'wer':.1})
        complete=dict(job_sha256=hashlib.sha256((self.root/'job.json').read_bytes()).hexdigest(),
                      outputs={'summary.json':hashlib.sha256((self.root/'summary.json').read_bytes()).hexdigest()})
        write_json(self.root/'complete.json',complete)
        self.assertEqual(verified_summary(self.root,job),{'wer':.1})
        with self.assertRaisesRegex(ValueError,'configuration'):verified_summary(self.root,{'a':2})
        (self.root/'summary.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'output'):verified_summary(self.root,job)

    def test_test_worker_rejects_tuning_before_loading_data(self):
        from benchmarks.evaluation_worker import main
        path=self.root/'job.json';write_json(path,dict(stage='test',tune=True))
        with patch.object(sys,'argv',['worker','--job',str(path)]),self.assertRaisesRegex(ValueError,'cannot tune'):
            main()

    def test_worker_forwards_absent_empty_and_nonempty_context(self):
        from benchmarks.evaluation_worker import decode_one
        class Decoder:
            def offline_decode(self, logits, lengths, contexts):
                self.contexts = contexts
                return [{}]
        decoder = Decoder()
        row = dict(logits=np.zeros((2,4),np.float32),adjusted_len=2)
        for context in (None,'','previous sentence'):
            decode_one(decoder,dict(row,context=context),True,None)
            self.assertEqual(decoder.contexts,None if context is None else [context])

    def test_exported_decoder_and_sweep_preserve_frozen_weights(self):
        cfg,cases=self.load(reorder_logit_columns=False);decoder=cases['baseline']['decoder']
        selection=dict(roles={'recommended':'baseline'},configurations={'baseline':{'decoder':decoder}})
        export_configs(self.root,selection,cfg,dict(validation_path=cfg['validation_logits']))
        saved=json.loads((self.root/'selected_configs/recommended_decoder.json').read_text())
        from sweep.lm_sweep import load_lm_sweep_config
        sweep=load_lm_sweep_config(str(self.root/'selected_configs/recommended_sweep.yaml'))
        self.assertEqual(saved['llm_alpha'],.45)
        self.assertEqual(sweep['llm_alpha'],0.)
        self.assertEqual(sweep['alpha_range']['low'],.45)
        self.assertEqual(sweep['objective_weights'],dict(wer=1000.,time=1.))
        self.assertEqual(sweep['unk_score'],-math.inf)
        self.assertFalse(sweep['reorder_logit_columns'])


if __name__=='__main__':unittest.main()
