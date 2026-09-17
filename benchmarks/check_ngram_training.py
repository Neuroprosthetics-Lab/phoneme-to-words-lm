"""Build and load tiny word/spelling LMs with real KenLM/SRILM tools."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--lmplz',required=True)
    parser.add_argument('--build-binary',required=True)
    parser.add_argument('--srilm',required=True)
    parser.add_argument('--interpolate',required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    import kenlm
    import ngram.train_ngram_lm as training
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    a=root/'a.txt';b=root/'b.txt'
    a.write_text('the cat sat on the mat\nhello how are you\nwe are ready to go')
    b.write_text('the dog sat on the mat\nthank you for your help\nplease bring me water\n')
    reports=[]
    def run(name,backend,corpora,lm_type='word'):
        config=dict(output_dir=str(root/name),lm_type=lm_type,normalize_workers=1,
                    interpolator_backend=backend,memory='512M',corpora=corpora,
                    lmplz_path=args.lmplz,build_binary_path=args.build_binary,
                    srilm_ngram_path=args.srilm,kenlm_interpolate_path=args.interpolate)
        path=root/(name+'.json');path.write_text(json.dumps(config))
        validated=training.load_and_validate_config(str(path))
        with patch.object(training,'train_single_lm',wraps=training.train_single_lm) as estimate:
            training.train_ngram_lm(validated)
            calls=estimate.call_count
        model=kenlm.Model(str(root/name/'lm_unpruned.bin'))
        score=model.score('hello how are you')
        lexicon=(root/name/'lexicon.txt').read_text()
        assert lexicon and (root/name/'build_manifest.json').is_file()
        entry=dict(name=name,estimation_calls=calls,order=model.order,score=score,
                   lexicon_lines=len(lexicon.splitlines()))
        reports.append(entry)
        (root/'results.json').write_text(json.dumps(reports,indent=2))
        print('PASS',entry,flush=True)
        return root/name
    def text(path,order=2,name=None):
        return dict(normalized_path=str(path),order=order,weight=1,discount_fallback=True,**({'name':name} if name else {}))
    single=run('single_srilm','srilm',[dict(normalized_path=[str(a),str(b)],name='joined',order=2,discount_fallback=True)])
    assert reports[-1]['estimation_calls']==1
    run('single_kenlm','kenlm',[text(a)])
    assert reports[-1]['estimation_calls']==1
    for backend in ('srilm','kenlm'):
        out=run('arpa_only_'+backend,backend,[dict(arpa_path=str(single/'lm.arpa'),order=2)])
        assert reports[-1]['estimation_calls']==0
        assert 'water\t' in (out/'lexicon.txt').read_text()
    run('mixed_srilm','srilm',[dict(arpa_path=str(single/'lm.arpa'),order=2,weight=2),text(b,3)])
    run('mixed_kenlm','kenlm',[text(a,2),text(b,2)])
    intermediate=root/'mixed_kenlm/intermediate/a'
    run('intermediate_reuse','kenlm',[dict(intermediate_path=str(intermediate),order=2,weight=2),text(b,2)])
    assert reports[-1]['estimation_calls']==1
    raw=root/'raw.txt';raw.write_text('Hello, how are you? We have 12 cats.\nThe dog is happy.\n')
    run('raw_word','srilm',[dict(path=str(raw),order=2,discount_fallback=True)])
    run('raw_spelling','kenlm',[dict(path=str(raw),order=2)],lm_type='spelling')


if __name__=='__main__':main()
