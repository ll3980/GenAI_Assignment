FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.18 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/code/.venv/bin:$PATH"
WORKDIR /code

COPY pyproject.toml uv.lock /code/
RUN uv sync --locked --no-dev --no-cache --python /usr/local/bin/python

COPY app /code/app
COPY models /code/models
COPY scripts /code/scripts
COPY examples /code/examples
RUN python -c "from app.embedding_model import calculate_embedding; r = calculate_embedding('apple'); assert r['model_version'] == '3.8.0'; assert r['dimensions'] == len(r['embedding']) == 300; assert r['has_vector']"
RUN python -c "from app.cnn_service import get_classifier; c = get_classifier(); assert c.model.classifier[-1].out_features == 10"

EXPOSE 80
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:80/cnn/health', timeout=3).read()"]
CMD ["/code/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
