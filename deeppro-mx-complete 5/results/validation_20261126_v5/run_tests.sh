#!/usr/bin/env bash
set -euo pipefail
LOG=results/validation_20261126_v5
python - <<'PYENV'
import sys,json,torch,numpy,scipy,cv2,PIL
print(json.dumps(dict(python=sys.version,torch=torch.__version__,numpy=numpy.__version__,scipy=scipy.__version__,opencv=cv2.__version__,pillow=PIL.__version__),indent=2))
PYENV
printf '\n== test_gaussian_masks.py (required first) ==\n'
python code/test_gaussian_masks.py
printf '\nGAUSSIAN_TESTS_PASSED\n'
for test_file in code/test_*.py; do
    if [ "$test_file" = code/test_gaussian_masks.py ]; then continue; fi
    printf '\n== %s ==\n' "$test_file"
    python "$test_file"
done
printf '\nALL_TEST_FILES_PASSED\n'
