# DeepPro-Plus RGB-T runtime

This isolated local image retains the NVIDIA PyTorch 24.01 image's Python 3.10.12,
PyTorch 2.2.0a0+81ea7a4, and CUDA 12.3 runtime. The added requirements pin NumPy
1.23.5 because the original training loader passes a ragged list directly to
`np.random.choice`; NumPy 1.24 rejects that operation. These are executable
compatibility versions, not a claim about the authors' unpublished environment.

Build from this directory:

```bash
docker build --progress plain -t deeppro-rgbt:20261126 .
```

The Dockerfile uses PyPI without the base image's optional NVIDIA package index.
It replaces the inherited `opencv` distribution with one coherent
`opencv-python-headless` installation, avoiding overlapping `cv2` files.
The local `build.log` records the successful build, and `image.json` identifies
both the base and completed image. `environment.json` is the CPU smoke-test
output; `pip-freeze.txt` records the resolved Python packages. `smoke.py` verifies
all imports used by the original train/test scripts and the legacy ragged
sampling operation. It does not exercise CUDA or certify GPU readiness.

No model, loss, sampler, dataset, or training source file is changed by this
runtime setup. Training and evaluation must be verified separately.
