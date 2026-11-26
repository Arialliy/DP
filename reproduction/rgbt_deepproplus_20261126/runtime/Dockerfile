FROM nvcr.io/nvidia/pytorch:24.01-py3
COPY requirements.txt /tmp/deeppro-rgbt-requirements.txt
RUN PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL= python -m pip install --index-url https://pypi.org/simple --no-cache-dir -r /tmp/deeppro-rgbt-requirements.txt
# NVIDIA ships the differently named `opencv` package with overlapping cv2 files.
# Remove both packages before installing one coherent headless cv2 distribution.
RUN python -m pip uninstall -y opencv opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless && python -c "import shutil; shutil.rmtree('/usr/local/lib/python3.10/dist-packages/cv2', ignore_errors=True)" && PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL= python -m pip install --index-url https://pypi.org/simple --no-cache-dir --no-deps opencv-python-headless==4.8.1.78
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLCONFIGDIR=/tmp/mpl
