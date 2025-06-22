# Hybrid Dockerfile - install core WITH deps, force conflict packages
FROM python:3.10.16-slim-bullseye

# System deps
RUN apt-get update && apt-get install -y \
    build-essential curl git gcc g++ cmake pkg-config libhdf5-dev libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip wheel setuptools

# Step 1: PyTorch CPU first
RUN pip install --no-cache-dir \
    torch==2.5.1+cpu torchvision==0.20.1+cpu torchaudio==2.5.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Step 2: Core packages WITH dependencies (critical for proper functioning)
RUN pip install --no-cache-dir \
    numpy==2.2.5 \
    scipy==1.15.2 \
    pydantic==2.11.3 \
    fastapi==0.115.9 \
    uvicorn==0.34.3 \
    starlette==0.45.3

# Step 3: Database packages WITH deps
RUN pip install --no-cache-dir \
    pymongo==4.13.0

# Step 4: ChromaDB (may have conflicts but try with deps first)
RUN pip install --no-cache-dir \
    chromadb==0.6.3 || pip install --no-cache-dir --no-deps chromadb==0.6.3

RUN pip install --no-cache-dir --no-deps \
    chroma-hnswlib==0.7.6

# Step 5: Google AI packages - FORCE exact versions but allow some deps
RUN pip install --no-cache-dir \
    protobuf==5.29.4 \
    grpcio==1.71.0 \
    google-auth==2.39.0 \
    google-api-core==2.24.2

# Force specific conflicting Google packages
RUN pip install --no-cache-dir --force-reinstall \
    google-ai-generativelanguage==0.6.17 \
    google-generativeai==0.8.5 \
    google-genai==1.11.0

# Step 6: ML packages WITH dependencies
RUN pip install --no-cache-dir \
    transformers==4.51.3 \
    sentence-transformers==4.1.0 \
    tokenizers==0.21.1 \
    huggingface-hub==0.30.2 \
    safetensors==0.5.3 \
    scikit-learn==1.6.1 \
    pandas==2.3.0

# Step 7: Web utilities WITH deps
RUN pip install --no-cache-dir \
    httpx==0.28.1 \
    aiohttp==3.11.18 \
    requests==2.32.3

# Step 8: Basic utilities WITH deps
RUN pip install --no-cache-dir \
    python-multipart==0.0.20 \
    python-dotenv==1.1.0 \
    email_validator==2.2.0 \
    bcrypt==4.3.0 \
    click==8.1.8 \
    tqdm==4.67.1 \
    tenacity==9.1.2 \
    backoff==2.2.1 \
    rich==14.0.0

# Step 9: LangChain (install what we can, skip conflicts)
RUN pip install --no-cache-dir \
    langchain-core==0.3.55 \
    langsmith==0.3.33 || echo "LangChain core failed, continuing..."

RUN pip install --no-cache-dir --no-deps \
    langchain==0.3.24 \
    langchain-chroma==0.2.3 \
    langchain-community==0.3.22 \
    langchain-google-genai==2.1.3 \
    langchain-huggingface==0.1.2 \
    langchain-text-splitters==0.3.8 || echo "LangChain modules failed, continuing..."

# Step 10: Optional packages (no deps to avoid conflicts)
RUN pip install --no-cache-dir --no-deps \
    openai==1.82.0 \
    haystack-ai==2.14.0 || echo "Optional packages failed, continuing..."

# Step 11: Critical missing dependencies that --no-deps skipped
RUN pip install --no-cache-dir \
    typing_extensions \
    typing_inspect \
    typing-inspection \
    annotated-types \
    pydantic_core \
    packaging \
    certifi \
    charset-normalizer \
    idna \
    six \
    python-dateutil \
    pytz \
    regex \
    joblib \
    cryptography \
    h11 \
    anyio \
    sniffio

# Create user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Environment
ENV PATH="/home/app/.local/bin:/usr/local/bin:$PATH"
ENV PYTHONPATH="/home/app:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=""
ENV USE_GPU=False

# Copy code
COPY --chown=app:app . .
RUN mkdir -p chroma_db data benchmark/results logs tmp

EXPOSE 8001
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]