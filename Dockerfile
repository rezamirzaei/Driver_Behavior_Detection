FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency file first for layer caching
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir pip --upgrade && \
    pip install --no-cache-dir ".[all]" 2>/dev/null || \
    pip install --no-cache-dir \
        "category-encoders>=2.6.0,<2.7.0" \
        "imbalanced-learn>=0.11.0,<0.13.0" \
        "matplotlib>=3.7.0,<3.10.0" \
        "numpy>=1.24.0,<1.27.0" \
        "pandas>=2.0.0,<2.1.0" \
        "plotly>=5.18.0" \
        "pydantic>=2.0.0,<3.0.0" \
        "scikit-learn>=1.3.0,<1.6.0" \
        "scipy>=1.11.0,<1.14.0" \
        "seaborn>=0.12.0,<0.14.0" \
        "torch>=2.0.0,<2.3.0" \
        "tqdm>=4.65.0" \
        "fastapi>=0.109.1" \
        "uvicorn[standard]>=0.27.0" \
        "ruff>=0.4.0" \
        "mypy>=1.8.0"

# Copy application code
COPY src/ src/
COPY data/ data/
COPY tests/ tests/
COPY main.py ./

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
