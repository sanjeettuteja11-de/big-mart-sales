"""Create the executed, bounded audit notebook and portable handoff archive."""
from pathlib import Path
import json,sys,zipfile,html,time
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parent
DATA=Path('/Users/abhimanyunain/Downloads')


def main():
    nb=nbformat.v4.new_notebook()
    nb.cells=[
        nbformat.v4.new_markdown_cell('# BigMart v67–v69 audit\n\n## tl;dr\n\n'
            'v67 has a **calculated**, not newly observed, public RMSE of 1124.1799. '
            'v68 and v69 are unscored trained refinements. The v69 recipe improved two '
            'fresh-model CV checks by 4.5516 and 3.3099 RMSE points; this is not a public-score forecast.'),
        nbformat.v4.new_markdown_cell('## Context & Methods\n\n'
            'The submission generators `make_submission.py` and `refinement.py` are the training source. '
            'The latter retrains old and suffix-only corrections from raw data on 15 folds. '
            '`validate.py` rebuilds a fresh proxy baseline inside outer folds.\n\n'
            '### Key Assumptions\n\n'
            'Reported scores match the hashed CSVs; public rows remain fixed. The historical ensemble '
            'is a frozen input. Repeated CV uses the same training dataset and is subject to selection bias. '
            'Small exact-value tables are used here because this notebook verifies calculations and artifacts, '
            'rather than presenting a trend or a graphical model comparison.'),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport json, hashlib\nimport numpy as np\nimport pandas as pd\nroot = Path.cwd()\ndata = root.parent / 'data'\nif not data.exists():\n    data = Path('/Users/abhimanyunain/Downloads')\nmanifest = json.loads((root / 'v67_manifest.json').read_text())\nrefinement = json.loads((root / 'refinement_manifest.json').read_text())"),
        nbformat.v4.new_markdown_cell('## Data\n\nRaw inputs and frozen scored files are identified by SHA-256, not just filenames.'),
        nbformat.v4.new_code_cell("records = []\nfor name, item in refinement['inputs'].items():\n    path = data / Path(item['path']).name\n    observed = hashlib.sha256(path.read_bytes()).hexdigest()\n    assert observed == item['sha256']\n    records.append({'input': name, 'hash_verified': observed == item['sha256']})\nprint(pd.DataFrame(records).to_string(index=False))"),
        nbformat.v4.new_markdown_cell('## Results\n\n### Exact affine-line score calculation'),
        nbformat.v4.new_code_cell("x = np.array([0., .75, 1.])\ns = np.array([1140.0228905024, 1126.0148395277, 1124.4058880453])\na,b,c = np.linalg.solve(np.c_[x*x,x,np.ones(3)],s*s)\nalpha = -b/(2*a)\ncalculated = np.sqrt(c-b*b/(4*a))\nassert abs(calculated-manifest['calculated_public_rmse']) < 1e-8\nprint(f'Optimal alpha: {alpha:.12f}\\nCalculated public RMSE: {calculated:.10f}\\nObserved v67 public RMSE: not supplied')"),
        nbformat.v4.new_markdown_cell('### Fresh-model validation\n\nThese are proxy training-data checks, not leaderboard results.'),
        nbformat.v4.new_code_cell("rows=[]\nfor filename in ['validation_11827.json','validation_8461_grouped.json']:\n    check=json.loads((root/filename).read_text())\n    rows.append({'check': 'Product holdout' if check['grouped_by_item'] else 'Outlet stratified',\n                 'baseline': check['baseline_rmse'],\n                 'v68_recipe': check['variants']['1.0']['rmse'],\n                 'v69_recipe': check['variants']['1.5']['rmse']})\nprint(pd.DataFrame(rows).round(4).to_string(index=False))"),
        nbformat.v4.new_markdown_cell('### Submission integrity'),
        nbformat.v4.new_code_cell("keys=['Item_Identifier','Outlet_Identifier']; target='Item_Outlet_Sales'\ntest=pd.read_csv(data/'test_AbJTz2l.csv')\nbase=pd.read_csv(data/'bigmart_submission_v67_exact_quantity_optimum.csv')\noutputs={}\nfor version,item in refinement['output'].items():\n    path=data/Path(item['path']).name\n    assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']\n    frame=pd.read_csv(path)\n    assert frame.shape==(5681,3) and frame[keys].equals(test[keys])\n    assert np.isfinite(frame[target]).all() and (frame[target]>=0).all()\n    outputs[version]=frame[target].to_numpy()\nassert np.allclose(outputs['v69']-base[target],1.5*(outputs['v68']-base[target]),atol=1e-8,rtol=0)\nprint('Passed: row count, column order, identifiers, hashes, finite/nonnegative values and affine relation.')"),
        nbformat.v4.new_markdown_cell('## Takeaways\n\nTry v69 first and v68 second. '
            'AV evaluation is required to establish whether either reaches 1120. '
            'No hidden labels or inferred public-row membership were used. '
            'See `README.md` for exact generator commands and baseline reproduction limits.'),
    ]
    nb.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
    client=NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}})
    client.km=client.create_kernel_manager()
    client.km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    client.execute()
    nbformat.validate(nb)
    nbformat.write(nb,ROOT/'audit.ipynb')
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><title>BigMart audit</title>',
           '<style>body{max-width:1000px;margin:40px auto;font:16px/1.55 system-ui;color:#172033;padding:0 24px}pre{white-space:pre-wrap;background:#f1f5f9;padding:16px;border-radius:8px}.markdown{white-space:pre-wrap;margin:22px 0}h1{font-size:28px}</style>',
           '<h1>Executed BigMart v67–v69 audit</h1>']
    for cell in nb.cells:
        if cell.cell_type=='markdown':
            parts.append('<div class="markdown">'+html.escape(cell.source)+'</div>')
        else:
            for output in cell.get('outputs',[]):
                if output.output_type=='error':raise RuntimeError(output)
                text=output.get('text',output.get('data',{}).get('text/plain',''))
                if text:parts.append('<pre>'+html.escape(text)+'</pre>')
    parts.append('</html>');(ROOT/'audit_preview.html').write_text('\n'.join(parts))
    archive=DATA/'bigmart_v67_v69_reproducible_bundle.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in ROOT.iterdir():
            if path.suffix in {'.py','.md','.json','.csv','.ipynb','.html'} or path.name=='refinement_delta.npz':
                z.write(path,Path(ROOT.name)/path.name)
        previous=ROOT.parent/'bigmart_v65_research'
        for name in ['experiment.py','make_submission.py','requirements-submission.txt','requirements-research.txt']:
            z.write(previous/name,Path(previous.name)/name)
        for name in ['train_v9rqX0R.csv','test_AbJTz2l.csv','sample_submission_8RXa3c6.csv',
                     'bigmart_submission_v63_exact_v60_optimum.csv',
                     'bigmart_submission_v65_quantity_correction.csv','bigmart_submission_v66_quantity_correction.csv',
                     'bigmart_submission_v67_exact_quantity_optimum.csv',
                     'bigmart_submission_v68_denoised_quantity.csv','bigmart_submission_v69_denoised_quantity.csv']:
            z.write(DATA/name,Path('data')/name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        print(f'Bundle verified: {archive} ({len(z.namelist())} entries)')
    print(f'Executed notebook: {ROOT / "audit.ipynb"}')


if __name__=='__main__':main()
