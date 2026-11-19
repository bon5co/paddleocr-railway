#!/usr/bin/env python3
"""PaddleOCR PP-OCRv6 HTTP API for Railway.

One route that matters: POST /ocr takes an image or a PDF and returns every text
line with its confidence and bounding box. Upstream PaddleOCR is a Python library
with no server, no auth and weights that download on first use; this file adds the
parts a hosted deploy needs:

  1. Bearer API_KEY on every route except GET /health.  The service burns real CPU
     per request, so an unguarded public URL is somebody else's free OCR.
  2. GET /health answers 503 until the pipeline is loaded and 200 after, one path
     segment deep, unauthenticated, so Railway's healthcheck only passes a deploy
     that can actually answer POST /ocr.
  3. Weights are baked into the image (see Dockerfile); nothing downloads at boot
     or inside a request, where Railway's edge would abandon it at about 5 minutes.
  4. Paddle's CPU predictor is not thread safe, so requests run one at a time behind
     a lock and the work is sized from the container's cgroup quota, not the host's.
"""

import io
import os
import socket
import sys
import threading
import time

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = 120_000_000

API_KEY = os.environ.get("API_KEY") or ""
TIER = (os.environ.get("OCR_TIER") or "medium").strip().lower()
TIERS = ("medium", "small", "tiny")
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB") or 25)
MAX_PAGES = int(os.environ.get("MAX_PAGES") or 30)
PDF_SCALE = float(os.environ.get("PDF_SCALE") or 2.0)
CPU_THREADS = int(os.environ.get("OCR_CPU_THREADS") or os.environ.get("OMP_NUM_THREADS") or 4)
AUTO_ROTATE_DEFAULT = (os.environ.get("OCR_AUTO_ROTATE") or "true").lower() in ("1", "true", "yes")
PUBLIC_PATHS = {"/health"}

app = FastAPI(
    title="PaddleOCR API",
    version="1.0.0",
    description="PP-OCRv6 text detection and recognition. Authorization: Bearer <API_KEY>.",
)

_state = {"ocr": None, "error": None, "loaded_in": None}
_lock = threading.Lock()


def _load() -> None:
    t0 = time.time()
    try:
        from paddleocr import PaddleOCR

        # enable_mkldnn=False: Paddle 3.x's oneDNN path raises
        # "ConvertPirAttribute2RuntimeAttribute not support" on the first predict.
        ocr = PaddleOCR(
            text_detection_model_name=f"PP-OCRv6_{TIER}_det",
            text_recognition_model_name=f"PP-OCRv6_{TIER}_rec",
            use_doc_orientation_classify=True,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
            cpu_threads=CPU_THREADS,
        )
        # Warm the kernels so the first caller does not pay for them.
        blank = np.full((64, 256, 3), 255, dtype=np.uint8)
        ocr.predict(blank)
        _state["ocr"] = ocr
        _state["loaded_in"] = round(time.time() - t0, 1)
        print(f"[paddleocr] ready tier={TIER} threads={CPU_THREADS} in {_state['loaded_in']}s", flush=True)
    except Exception as exc:  # noqa: BLE001 - reported over /health, never fatal
        _state["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[paddleocr] LOAD FAILED: {_state['error']}", flush=True)


@app.middleware("http")
async def bearer_auth(request: Request, call_next):
    path = request.url.path.rstrip("/") or "/"
    if path in PUBLIC_PATHS:
        return await call_next(request)
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or token != API_KEY:
        return JSONResponse(
            {"detail": "Missing or invalid Authorization: Bearer <API_KEY>."},
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await call_next(request)


@app.get("/health", include_in_schema=False)
def health():
    if _state["ocr"] is None:
        body = {"status": "loading" if not _state["error"] else "failed", "error": _state["error"]}
        return JSONResponse(body, status_code=503)
    return {"status": "ok", "model": f"PP-OCRv6_{TIER}", "threads": CPU_THREADS, "loaded_in_s": _state["loaded_in"]}


def _pages_from_upload(data: bytes, max_pages: int):
    """Yield (page_number, BGR ndarray) for an image or PDF."""
    if data[:5] == b"%PDF-":
        import pypdfium2 as pdfium

        try:
            pdf = pdfium.PdfDocument(data)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(422, f"could not open the PDF: {exc}") from exc
        total = len(pdf)
        for index in range(min(total, max_pages)):
            pil = pdf[index].render(scale=PDF_SCALE).to_pil().convert("RGB")
            yield index + 1, np.ascontiguousarray(np.asarray(pil)[:, :, ::-1]), total
        return
    try:
        pil = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, f"not a readable image or PDF: {exc}") from exc
    yield 1, np.ascontiguousarray(np.asarray(pil)[:, :, ::-1]), 1


def _lines(result, min_confidence: float):
    texts = result["rec_texts"]
    scores = result["rec_scores"]
    polys = result["rec_polys"]
    boxes = result["rec_boxes"]
    out = []
    for i, text in enumerate(texts):
        score = float(scores[i])
        if score < min_confidence or not str(text).strip():
            continue
        out.append(
            {
                "text": str(text),
                "confidence": round(score, 4),
                "box": [int(v) for v in boxes[i]],
                "polygon": [[int(x), int(y)] for x, y in polys[i]],
            }
        )
    return out


@app.post("/ocr")
def ocr_endpoint(
    file: UploadFile = File(..., description="PNG, JPEG, WebP, BMP, TIFF or PDF"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0, description="drop lines scoring below this"),
    rotate: bool = Query(AUTO_ROTATE_DEFAULT, description="fix 90/180/270 degree page rotation first"),
    max_pages: int = Query(MAX_PAGES, ge=1, le=MAX_PAGES, description="PDF pages to read, from the first"),
):
    if _state["ocr"] is None:
        raise HTTPException(503, _state["error"] or "model is still loading, retry in a few seconds")
    data = file.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"file is larger than {MAX_UPLOAD_MB} MB")
    if not data:
        raise HTTPException(422, "empty file")

    t0 = time.time()
    pages, total = [], 1
    for number, array, total in _pages_from_upload(data, max_pages):
        with _lock:
            results = _state["ocr"].predict(array, use_doc_orientation_classify=rotate)
        lines = _lines(results[0], min_confidence) if results else []
        pages.append(
            {
                "page": number,
                "width": int(array.shape[1]),
                "height": int(array.shape[0]),
                "text": "\n".join(line["text"] for line in lines),
                "lines": lines,
            }
        )
    return {
        "model": f"PP-OCRv6_{TIER}",
        "page_count": total,
        "pages_read": len(pages),
        "text": "\n\n".join(page["text"] for page in pages),
        "pages": pages,
        "seconds": round(time.time() - t0, 2),
    }


def main() -> int:
    if not API_KEY:
        print("[paddleocr] FATAL: API_KEY is empty; refusing to publish /ocr to anyone who finds the URL.", file=sys.stderr, flush=True)
        return 1
    if TIER not in TIERS:
        print(f"[paddleocr] FATAL: OCR_TIER={TIER!r} is not one of {', '.join(TIERS)}.", file=sys.stderr, flush=True)
        return 1
    threading.Thread(target=_load, name="load", daemon=True).start()
    # Railway injects PORT and its healthcheck dials that injected port. asyncio sets
    # IPV6_V6ONLY on a "::" listener, so one socket per family: IPv4 for the edge and
    # healthcheck, IPv6 for the private network (<service>.railway.internal).
    port = int(os.environ.get("PORT") or 8080)
    socks = []
    for family, addr in ((socket.AF_INET, "0.0.0.0"), (socket.AF_INET6, "::")):
        try:
            sock = socket.socket(family, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            sock.bind((addr, port))
            sock.set_inheritable(True)
            socks.append(sock)
        except OSError as exc:
            print(f"[paddleocr] cannot bind {addr}:{port}: {exc}", flush=True)
    if not socks:
        return 1
    config = uvicorn.Config(app, log_level=os.environ.get("LOG_LEVEL", "info"))
    uvicorn.Server(config).run(sockets=socks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
