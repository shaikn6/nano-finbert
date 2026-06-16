FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install CPU-only PyTorch first (avoids downloading CUDA)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    fastapi>=0.111.0 \
    uvicorn[standard]>=0.30.0 \
    pydantic>=2.7.0 \
    numpy>=1.26.0 \
    python-dotenv>=1.0.0

# Copy source
COPY src/ ./src/
COPY data/ ./data/

# Non-root user
RUN adduser --disabled-password --gecos "" appuser
USER appuser

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "finbert.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
