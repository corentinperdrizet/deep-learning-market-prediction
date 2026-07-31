FROM python:3.13-slim

WORKDIR /app

# Scientific wheels (torch, pandas, scikit-learn, pyarrow) ship prebuilt
# manylinux binaries, so no compiler toolchain is needed here.
#
# torch is installed from PyTorch's CPU-only wheel index first: the default
# PyPI wheel bundles the full CUDA toolkit (~1.5GB of nvidia-* packages)
# that's useless in a container with no GPU, and drastically slows the
# build. Installing the CPU build up front satisfies requirements.txt's
# "torch>=2.9" so the second install step doesn't pull the CUDA variant.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

EXPOSE 8501 8000

# Overridden by docker-compose.yml for the `api` service; this default runs
# the dashboard so `docker build . && docker run -p 8501:8501 <image>` works
# standalone too.
CMD ["streamlit", "run", "src/app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
