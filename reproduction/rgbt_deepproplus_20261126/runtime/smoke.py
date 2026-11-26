"""CPU-only environment and legacy NumPy sampler witness."""
import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
import warnings

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--source", help="Optional original DeepPro source root to import")
args = parser.parse_args()

PACKAGES = {
    "numpy": "numpy", "scipy": "scipy", "skimage": "scikit-image",
    "sklearn": "scikit-learn", "cv2": "opencv-python-headless",
    "PIL": "Pillow", "matplotlib": "matplotlib", "tqdm": "tqdm", "thop": "thop",
}
versions = {}
for module, distribution in PACKAGES.items():
    importlib.import_module(module)
    versions[distribution] = importlib.metadata.version(distribution)

# Mirrors the original TrainIRSeqDataLoader's ragged list sampling shape.
samples = [[("im0.png", "mask0.png")],
           [("im0.png", "mask0.png"), ("im1.png", "mask1.png")]]
np.random.seed(46)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    sampled = np.random.choice(samples, p=[1 / 3, 2 / 3])
assert sampled in samples
assert torch.__version__ == "2.2.0a0+81ea7a4", torch.__version__
assert np.__version__ == "1.23.5", np.__version__
source_imports = []
if args.source:
    sys.path.insert(0, args.source)
    for name in ["train", "test", "networks.models.DeepPro-Plus"]:
        importlib.import_module(name)
        source_imports.append(name)
print(json.dumps({
    "status": "pass", "scope": "CPU imports and original ragged NumPy sampler only",
    "python": sys.version, "platform": platform.platform(),
    "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
    "packages": versions, "ragged_sampler": "pass",
    "sampler_warnings": [str(w.message) for w in caught],
    "gpu_tested": False, "source_imports": source_imports,
}, indent=2))
