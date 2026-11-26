# Official README reproduction environment

Source: `/home/ly/DeepPro-main`, read-only mount. No source edits or runtime patches.
Data: `/home/ly/DeepPro-main/datasets/NUDT-MIRSDT`, read-only mount.

The training command is the README command plus a unique output name:

```bash
python train.py --model DeepPro-Plus --seqlen 40 --dataset NUDT-MIRSDT --datapath ./datasets/NUDT-MIRSDT --log_dir NUDT-MIRSDT_readme_retrain_20261125
```

Defaults: 32 epochs, batch 4, 40 frames, sample_rate 0.1, 128 crop, Adam lr .001, wd .0001, lr times .7 every 10 epochs. Training evaluates all 20 test sequences each epoch at threshold .5 and selects best IoU in the original main. Its automatic test.py call uses the best weights, original 27 thresholds, and threshold .5 for IoU. This follows the current README/defaults; the historical released-checkpoint log instead recorded threshold_eval=.3.

The original script does not call seed_everything and worker NumPy seeds depend on wallclock time. The latest instruction to execute the original code takes precedence over the older seed46 instruction. This run is not labeled seed46. No generator, caching, precision, augmentation, loss, schedule, or checkpoint-selection changes.

Environment: inherited CUDA12.8/Torch2.7 supports the local RTX5090. A separate Docker image downgrades NumPy and its dependent packages to run the original ragged-list sampling without edits. Authors did not publish exact package versions; this is an executable compatible environment, not a claim of matching their original stack. TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 restores the old torch.load behavior for trusted original/local weights. The seeded preflight is a separate process and does not seed training. The inherited image has unused ultralytics missing optional dependencies; it is never imported by this experiment.

## CPU document-follow command

Execute exactly this command; do not fix or substitute it:

```bash
docker run --rm --network none --cpus 4 --memory 8g --entrypoint python --user 1000:1000 -v /home/ly/DeepPro-main:/official:ro -v /home/ly/DeepPro-main/datasets/NUDT-MIRSDT:/data:ro -v /home/ly/DeepPro-main/reproduction/nudt_readme_20261125:/run:ro -w /official -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 -e TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e MPLCONFIGDIR=/tmp/mpl sha256:7d656a4ae1641f8dd9e0d29cd04ceedf61d349a7b7714508a804275f70e842b7 /run/preflight.py --device cpu
```

Expected: exit 0 and `WITNESS` JSON reporting 768 training samples, 192 updates/epoch, 20 test sequences and PASS for the unmodified sampler. NumPy ragged-list deprecation warnings are expected.

## GPU dispatch

`queue_run.py` is an OS scheduler only; it never imports model/training code. It checks source hashes, waits for a GPU below 500 MiB with no compute process, runs the standalone CUDA witness, then executes the unmodified training command in a fresh Python process. It saves STATUS.json, LAUNCH.json, gpu_witness.log, console.log, and official ./log outputs. It checks final test completion because original train.py ignores os.system(test.py) errors. A successful CPU check alone does not certify GPU readiness.

Run directory: `/home/ly/DeepPro-main/reproduction/nudt_readme_20261125`. Image: `sha256:7d656a4ae1641f8dd9e0d29cd04ceedf61d349a7b7714508a804275f70e842b7`.
