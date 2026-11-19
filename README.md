# paddleocr-railway

HTTP API around [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) PP-OCRv6 for Railway. CPU only, weights baked into the image, Bearer `API_KEY` on every route except `GET /health`.

- `POST /ocr` multipart `file` (image or PDF) -> text, per-line confidence, boxes. Query: `min_confidence`, `rotate`, `max_pages`.
- `GET /health` -> 200 once the model is loaded.
- Env: `API_KEY` (required), `OCR_TIER` (`medium`|`small`|`tiny`, default `medium`), `MAX_UPLOAD_MB`, `MAX_PAGES`, `PDF_SCALE`.

Code is Apache-2.0 like upstream. PP-OCRv6 weights: Apache-2.0 (Hugging Face model cards, `PaddlePaddle/PP-OCRv6_*`).
