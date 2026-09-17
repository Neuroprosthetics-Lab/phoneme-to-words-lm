import math
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch
from phoneme_to_words_lm.llm_scoring import prepare_scoring_inputs,score_sentences,score_prepared,padded_inputs,_score_batch
from phoneme_to_words_lm.decoder import KenLMFlashlightTextLM

class CharacterTokenizer:
    eos_token_id=0
    pad_token_id=0
    def __call__(self, text, **kwargs):
        return dict(input_ids=[ord(c)%63+1 for c in text],offset_mapping=[(i,i+1) for i in range(len(text))])

class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1));self.calls=[]
    @property
    def device(self):return self.weight.device
    def forward(self,input_ids,attention_mask,use_cache):
        self.calls.append((tuple(input_ids.shape),use_cache))
        logits=torch.cos(input_ids.float().unsqueeze(-1)+torch.arange(64).float()/3)
        return SimpleNamespace(logits=logits)

class ScoringTests(unittest.TestCase):
    def setUp(self):self.tok=CharacterTokenizer();self.model=TinyModel()

    def test_single_token_is_scored(self):
        raw,counts=score_sentences(self.model,self.tok,['a','b'])
        self.assertEqual(counts,[1,1]);self.assertTrue(all(x<0 for x in raw));self.assertNotEqual(raw[0],raw[1])

    def test_empty_and_mixed_contexts_are_batch_invariant(self):
        alone,n=score_sentences(self.model,self.tok,['a'],contexts=[''])
        mixed,m=score_sentences(self.model,self.tok,['a','b'],contexts=['','history'],batch_size=2)
        self.assertEqual(n,[1]);self.assertEqual(m,[1,1]);self.assertAlmostEqual(alone[0],mixed[0],places=5)
        none,c=score_sentences(self.model,self.tok,['a']);self.assertEqual(none,alone)

    def test_reference_mask_and_real_eos_with_pad_equals_eos(self):
        seq=['ab','c'];contexts=['hello','']
        requests=prepare_scoring_inputs(self.tok,seq,contexts,score_eos=True)
        self.assertEqual([r.token_count for r in requests],[3,2])
        ids,attention,targets=padded_inputs(requests,self.tok,'cpu')
        logits=self.model(ids,attention,False).logits.float()
        vals=logits.log_softmax(-1)[:,:-1].gather(-1,ids[:,1:,None]).squeeze(-1)
        expected=(vals*targets[:,1:]).sum(-1)
        actual=_score_batch(self.model,self.tok,requests,2)
        torch.testing.assert_close(torch.tensor(actual),expected,atol=1e-5,rtol=1e-6)
        self.assertTrue(all(not cache for _,cache in self.model.calls))

    def test_bucketing_restores_order_and_caps_padding(self):
        texts=['longer candidate','a','medium','bb']
        raw,count=score_sentences(self.model,self.tok,texts,batch_size=3,max_batch_tokens=20)
        self.assertTrue(all(b*t<=20 for (b,t),_ in self.model.calls))
        one=[score_sentences(self.model,self.tok,[s])[0][0] for s in texts]
        torch.testing.assert_close(torch.tensor(raw),torch.tensor(one),atol=1e-5,rtol=1e-6)
        self.assertEqual(count,list(map(len,texts)))

    def test_overlength_never_truncates_or_calls_model(self):
        with self.assertRaisesRegex(ValueError,'never silently truncated'):score_sentences(self.model,self.tok,['hello'],max_length=3)
        with self.assertRaisesRegex(ValueError,'batch size 1'):score_sentences(self.model,self.tok,['hello'],max_batch_tokens=3)
        self.assertEqual(self.model.calls,[])

    def test_zero_target_does_not_call_model(self):
        self.assertEqual(score_sentences(self.model,self.tok,['']),([0.0],[0]))
        self.assertEqual(score_sentences(self.model,self.tok,[]),([],[]))
        self.assertEqual(self.model.calls,[])

    def test_oom_retries_only_smaller_batches(self):
        requests=prepare_scoring_inputs(self.tok,['aa','bb','cc','dd'])
        seen=[]
        def work(model,tok,req,chunk):
            seen.append(len(req))
            if len(req)>1:raise torch.OutOfMemoryError('test allocation')
            return [-float(req[0].input_ids[-1])]
        with patch('phoneme_to_words_lm.llm_scoring._score_batch',side_effect=work):
            result=score_prepared(self.model,self.tok,requests)
        self.assertEqual(seen,[4,2,1,1,2,1,1]);self.assertEqual(len(result),4)
        with patch('phoneme_to_words_lm.llm_scoring._score_batch',side_effect=torch.OutOfMemoryError('test')):
            with self.assertRaisesRegex(RuntimeError,'single'):score_prepared(self.model,self.tok,requests[:1])

    def test_boundary_crossing_token_belongs_to_candidate(self):
        class MergeTokenizer(CharacterTokenizer):
            def __call__(self,text,**kwargs):return dict(input_ids=[1,2],offset_mapping=[(0,1),(1,len(text))])
        request=prepare_scoring_inputs(MergeTokenizer(),['hello'],prefix='x ')[0]
        self.assertEqual(request.target_mask,[False,True])

    def test_invalid_policy_and_context(self):
        for kwargs in [dict(prefix=''),dict(contexts=['a','b']),dict(contexts=[None]),dict(score_eos='yes'),dict(batch_size=0),dict(max_length=0)]:
            with self.subTest(kwargs=kwargs),self.assertRaises((ValueError,TypeError)):score_sentences(self.model,self.tok,['a'],**kwargs)

    def test_nonfinite_model_scores_rejected(self):
        def bad(**kw):return SimpleNamespace(logits=torch.full((*kw['input_ids'].shape,64),float('nan')))
        self.model.forward=bad
        with self.assertRaisesRegex(RuntimeError,'nonfinite'):score_sentences(self.model,self.tok,['a'])

if __name__=='__main__':unittest.main()
