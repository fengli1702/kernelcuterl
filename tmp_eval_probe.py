import json
import os
import re
import sys

sys.path.insert(0, os.getcwd())
from kernelrl.eval.pipeline import run_eval_pipeline

with open('rollout/results/kernelbench_rollout_v9_python.jsonl','r') as f:
    recs=[json.loads(l) for l in f if l.strip()]

for sid in [1,2,9,35,56]:
    rec = next(r for r in recs if r['sample_id'] == sid)
    ref = rec['reference']
    text = rec['samples'][0]['text']
    m = re.search(r'(?is)</think>|</thinking>', text)
    tail = text[m.end():] if m else text
    blocks = re.findall(r"```(?:[a-zA-Z0-9_+#-]*)?\n?(.*?)\n?```", tail, re.DOTALL)
    blocks = [b.strip() for b in blocks if b.strip()]
    if blocks:
        pref = [b for b in blocks if ('class ModelNew' in b or 'class Model ' in b)]
        if pref:
            code = max(pref, key=len)
            method = 'fenced_pref'
        else:
            code = max(blocks, key=len)
            method = 'fenced_last'
    else:
        code = tail.strip()
        method = 'raw_after'

    req = {
        'reference_code': ref,
        'generated_code': code,
        'compile': {'code': code, 'language': 'python'},
        'correctness': {},
        'gates': {
            'require_compile_ok': True,
            'require_correctness_ok': True,
            'require_hack_clean': False,
        },
        'hack_check': {'code': code},
    }

    r = run_eval_pipeline(req)
    print(
        'sample', sid,
        'method', method,
        'len', len(code),
        'compile_status', r['modules']['compile']['status'],
        'compile_rc', r['modules']['compile']['metrics'].get('return_code'),
        'correct_status', r['modules']['correctness']['status'],
        'correct_ok', r['modules']['correctness'].get('ok'),
        'correct_error', (r['modules']['correctness'].get('errors')[0].get('id') if r['modules']['correctness'].get('errors') else None),
        'pipeline_ok', r['ok'],
    )
