import json
import os
import re
import sys
sys.path.insert(0, os.getcwd())

from kernelrl.eval.pipeline import run_eval_pipeline

recs=[json.loads(l) for l in open('rollout/results/kernelbench_rollout_v9_python.jsonl',encoding='utf-8') if l.strip()]

for sid in [1,56]:
    rec = next(r for r in recs if r['sample_id']==sid)
    print('SID', sid, 'start', flush=True)
    text = rec['samples'][0]['text']
    m = re.search(r'(?is)</think>|</thinking>', text)
    tail = text[m.end():] if m else text

    blocks = re.findall(r'```(?:[a-zA-Z0-9_+#-]*)?\n?(.*?)\n?```', tail, re.DOTALL)
    blocks = [b.strip() for b in blocks if b.strip()]
    if blocks:
        pref = [b for b in blocks if ('class ModelNew' in b or 'class Model ' in b)]
        code = max(pref, key=len) if pref else max(blocks, key=len)
        method = 'fenced'
    else:
        code = tail.strip()
        method = 'raw'

    print(' extracted', len(code), method, flush=True)

    req = {
        'reference_code': rec['reference'],
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

    res = run_eval_pipeline(req)
    print(' done', sid, 'ok', res['ok'],
          'compile', res['modules']['compile']['status'],
          'rc', res['modules']['compile']['metrics'].get('return_code'),
          'corr', res['modules']['correctness']['status'],
          flush=True)
