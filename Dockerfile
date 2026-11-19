# PaddleOCR PP-OCRv6 API for Railway. Source: https://github.com/bon5co/paddleocr-railway
# CPU only, weights baked in, Bearer API_KEY, stateless (no volume needed).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PADDLE_PDX_CACHE_HOME=/opt/paddlex \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    PADDLE_PDX_MODEL_SOURCE=huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install \
    paddlepaddle==3.3.1 \
    paddleocr==3.7.0 \
    fastapi==0.165.0 \
    uvicorn==0.54.0 \
    python-multipart==0.0.34 \
    pypdfium2==5.15.2

# Bake every PP-OCRv6 tier (det + rec) and the page-orientation classifier. All are
# Apache-2.0 (Hugging Face model cards, PaddlePaddle/PP-OCRv6_*). Downloading at boot
# would put a ~1 minute fetch inside Railway's healthcheck window; downloading inside
# the first request would hit the edge's ~5 minute limit.
RUN python - <<'PY'
from paddleocr import PaddleOCR
for tier in ("medium", "small", "tiny"):
    PaddleOCR(
        text_detection_model_name=f"PP-OCRv6_{tier}_det",
        text_recognition_model_name=f"PP-OCRv6_{tier}_rec",
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
    )
PY
RUN du -sh /opt/paddlex

RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin ocr \
    && chown -R 1000:0 /opt/paddlex && chmod -R g+rwX /opt/paddlex
WORKDIR /app
COPY app.py entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh
USER 1000

EXPOSE 8080
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "/app/app.py"]
