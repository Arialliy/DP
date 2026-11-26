"""Run all staged mx_data tests and the unchanged M1 synthetic suite; CPU only."""
import hashlib
import json
from pathlib import Path
import sys
import time

stage_code = Path('/experiment/runtime/m1_fix/plateau_stage/fixed/code')
sys.path.insert(0, str(stage_code))
import mx_data
import test_mx_data
import test_miss_recovery
assert Path(mx_data.__file__).resolve() == stage_code/'mx_data.py'
assert test_miss_recovery.m1.mxd is mx_data
assert Path(test_miss_recovery.__file__).resolve() == stage_code/'test_miss_recovery.py'
assert hashlib.sha256(Path(test_miss_recovery.__file__).read_bytes()).digest() == hashlib.sha256(Path('/experiment/code/test_miss_recovery.py').read_bytes()).digest()
assert hashlib.sha256(Path(test_miss_recovery.m1.__file__).read_bytes()).digest() == hashlib.sha256(Path('/experiment/code/miss_recovery.py').read_bytes()).digest()
result = {'status': 'running', 'device': 'cpu', 'real_m1_runs': 0, 'model_forwards': 0, 'training_runs': 0,
          'loaded_mx_data': str(Path(mx_data.__file__).resolve()), 'tests': []}
started = time.monotonic()
for module, expected in [(test_mx_data,18),(test_miss_recovery,11)]:
    tests = sorted((name, func) for name,func in vars(module).items() if name.startswith('test_') and callable(func))
    assert len(tests) == expected, (module.__name__, len(tests))
    for name, func in tests:
        before = time.monotonic()
        func()
        row = {'suite':module.__name__,'test':name,'status':'passed','seconds':time.monotonic()-before}
        result['tests'].append(row)
        print(json.dumps(row),flush=True)
result.update(status='passed',seconds=time.monotonic()-started,torch_imported='torch' in sys.modules)
assert result['torch_imported'] is False
Path('/work/results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':'passed','tests':len(result['tests']),'seconds':result['seconds'],'torch_imported':False}),flush=True)
