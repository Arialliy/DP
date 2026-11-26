v7 isolated CPU preprocessing and optional compatibility support

This folder does not modify code/ or run_mx.sh. run_v7.py can launch ONLY the CPU
profile stage. It contains no GPU scheduler, inference, M0, training, evaluation,
or controls launch. Official DeepPro-Plus missed-detection attribution has priority;
later M0/MXC work requires a separate explicitly chosen command.

The CPU container pins the existing Docker image by SHA256, uses 8 CPUs, 48 GiB
memory and 8 GiB shm. Dataset, code, run_mx.sh, runtime and supplied splits.json
are read-only; results/cache/runs and generated configs/folds.json are writable.
The profile script moves any existing profile/motion directories aside as
documented in RUN.md. No earlier job, GPU allocation, or scheduler is touched.

Review the command:
  python runtime/run_v7.py preview

Freeze source after fixes, before the complete check:
  python runtime/run_v7.py prepare

After bash run_mx.sh check REALLY finishes with exit 0, attest its completed log:
  python runtime/run_v7.py record-check --exit-code 0 --log logs/check.20261126.log
This command records operator-supplied exit evidence; it does not rerun tests.
Do not execute it while the check is running or after a failed/partial check.

Start the reviewed CPU profile durably from the v7 project directory:
  nohup python -u runtime/run_v7.py start > runtime/runner.log 2>&1 < /dev/null &
No container is launched by prepare, record-check, verify-source, or preview.

A passing complete-check witness and unchanged source manifest are required.
A flock prevents duplicate runners; existing status.json or the named profile
container prevents accidental relaunch. Profile completion requires targets.csv,
targets.json, annotation-free motion clips.csv/motion.json and folds.json.
After completion the runner exits; it never queues any subsequent stage.

Monitoring:
  cat runtime/status.json
  tail -f runtime/dp-mxc-v7-prepare-20261126.log
If the host runner is interrupted, its detached container may continue; inspect
it before manual recovery. This runner never kills another container.

GPU compatibility support (for separate reviewed diagnostic commands):
Mount this runtime directory read-only and add PYTHONPATH=/experiment/runtime.
sitecustomize.py will install only the missing upstream Conv3d memory-format
helper. Implementation, license and provenance are copied verbatim from the
validated prior runtime. CPU preprocessing does not enable this startup hook.
HOMOGRAPHY_DIR, VT_ANN_DIR, VT_OLD and DET_RUN are explicitly unset by the CPU
container command; v7 network inputs therefore use results/motion/homographies.
