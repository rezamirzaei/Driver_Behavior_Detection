FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

RUN pip install --upgrade pip \
    && pip install \
        "category-encoders>=2.6.0,<2.7.0" \
        "imbalanced-learn>=0.11.0,<0.13.0" \
        "matplotlib>=3.7.0,<3.10.0" \
        "numpy>=1.24.0,<1.27.0" \
        "pandas>=2.0.0,<2.1.0" \
        "plotly>=5.18.0" \
        "pydantic>=2.0.0,<3.0.0" \
        "pydantic-settings>=2.2.0,<3.0.0" \
        "scikit-learn>=1.3.0,<1.6.0" \
        "scipy>=1.11.0,<1.14.0" \
        "seaborn>=0.12.0,<0.14.0" \
        "torch>=2.0.0,<2.3.0" \
        "tqdm>=4.65.0" \
        "fastapi>=0.109.1" \
        "uvicorn[standard]>=0.27.0"

COPY src/ src/
COPY data/ data/
COPY main.py README.md ./

RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=5)"

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
