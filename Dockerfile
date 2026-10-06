# One image for every Python component (services, worker, generator, evaluation); the compose file picks the command.
FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
# bundle the sentence-embedding model so the runtime never needs internet
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/all-MiniLM-L6-v2').save('/models/all-MiniLM-L6-v2')"
COPY services/common services/common
RUN pip install --no-deps ./services/common
COPY synthetic_generator synthetic_generator
COPY evaluation evaluation
COPY services services
ENV ARPMS_SBERT_MODEL=/models/all-MiniLM-L6-v2 ARPMS_RAW_DIR=/raw ARPMS_ARTIFACT_DIR=/artifacts
