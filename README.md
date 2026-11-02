---
title: DocMind AI Backend
emoji: 📄
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
---

# DocMind AI

DocMind AI is a document assistant powered by FastAPI, SQLite, and configurable AI providers.

## Features
- **Isolated workspaces:** Each browser receives a cryptographically random, HttpOnly session cookie. Documents, chunks, and chat history are scoped to that session.
- **AI providers:** Inference uses a configured Hugging Face Router, Groq, or reachable Ollama server.
- **Multi-format Ingestion:** Supports `.pdf`, `.docx`, `.txt`, and `.md` files.
- **Three Modes:**
  - **Ask documents:** Retrieve relevant passages and answer questions.
  - **Summarize:** Create concise summaries of uploaded documents.
  - **Compare documents:** Contrast and compare key themes across multiple documents.

---

## Local Windows setup

1. Install **Python 3.12** and add it to PATH.
2. To use your Ollama model, install Ollama and configure its URL and model name in PowerShell:
   ```bash
   $env:OLLAMA_URL = "http://127.0.0.1:11434"
   $env:OLLAMA_MODEL = "hf.co/<your-user>/<your-Qwen-repository>"
   ollama pull $env:OLLAMA_MODEL
   ```
3. From the repository directory, install dependencies and start the backend:
   ```cmd
   setup_windows.bat
   run_docmindai.bat
   ```
   Open your browser at [http://127.0.0.1:8000](http://127.0.0.1:8000).

Run tests with:
```cmd
.venv\Scripts\python.exe -m pip install -r backend\requirements-dev.txt
.venv\Scripts\python.exe -m pytest -q
```

The backend defaults to port 8000 and honors the `PORT` environment variable. The Windows launcher binds to `127.0.0.1`; the container binds to `0.0.0.0`.

---

## GitHub Pages

The frontend is deployed from `frontend/` by the GitHub Actions workflow whenever changes are pushed to `main`. In the repository settings, open **Settings → Pages** and set **Build and deployment → Source** to **GitHub Actions**. The site is published at `https://praveenkumarkunwar2007-hash.github.io/DocMind/`.

GitHub Pages hosts only the static frontend; document and chat API requests use the backend at `https://docmind-y5n4.onrender.com`. The frontend sends credentialed requests so the backend's HttpOnly session cookie is included. HTTPS cookies use `SameSite=None; Secure` for the cross-origin Pages-to-Render request. Some browsers block third-party cookies; for reliable sessions, open the backend's own URL or use a custom frontend/backend arrangement on the same site.

## Render deployment

Create a Docker web service with the repository root as its root directory and `Dockerfile` as its Dockerfile. The container starts with:

```sh
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
```

Render supplies `PORT`; the container defaults to 8000. `GET /health` returns `{"status":"ok"}`. Configure `FRONTEND_ORIGINS` as a comma-separated list of exact frontend origins, for example:

```text
https://praveenkumarkunwar2007-hash.github.io
```

Configure inference providers in Render's environment settings. Ollama is tried first, then Groq, then Hugging Face by default. Set `PROVIDER_ORDER` to change this order. Ollama requires both `OLLAMA_URL` and `OLLAMA_MODEL`; for a Hugging Face GGUF model, use a name such as `hf.co/owner/repository`. `OLLAMA_URL` must be a host URL, without a model path. The default Groq model is `openai/gpt-oss-20b`; set `GROQ_MODEL` to override it. Hugging Face fallback is enabled only when both `HF_TOKEN` and `HF_MODEL` are set. Requests have a minimum timeout of 120 seconds; set `REQUEST_TIMEOUT` to a higher value if needed.

The backend sends retrieved document passages to each configured inference provider in `PROVIDER_ORDER` until one succeeds. Configure only providers you trust with your documents. Do not put credentials in frontend code. The `.pkl` checkpoint is not loaded or used by the application; it is intentionally excluded from the Docker image. Ollama must be running separately and reachable from Render; `127.0.0.1` only works when Ollama runs in the same container.

Example Render variables (replace the example host, model, and credentials):

```text
OLLAMA_URL=https://your-private-ollama-host
OLLAMA_MODEL=hf.co/your-user/DocMind-Qwen-Model
PROVIDER_ORDER=ollama,groq,hf
REQUEST_TIMEOUT=120
GROQ_API_KEY=your-groq-api-key
GROQ_MODEL=openai/gpt-oss-20b
HF_TOKEN=your-hugging-face-token
HF_MODEL=your-user/your-enabled-model
```

Render's default filesystem is ephemeral. To keep documents and chat history across restarts, attach a persistent disk mounted at `/var/data` and set `DOCMIND_DATA_DIR=/var/data`. Without a persistent disk, stored data can be lost when the service restarts or redeploys. The database is shared by the process but every document, chunk retrieval, deletion, and history query is filtered by its opaque session owner ID.

The Pages frontend can override the backend URL by setting `DOCMIND_API_URL` in browser local storage.

---

## Model checkpoint

Large model checkpoints such as `backend/qwen2.5_0.5b_model.pkl` are local-only and ignored by Git. The application does not load this file; inference uses the configured Ollama, Groq, or Hugging Face provider. Do not force-add model checkpoints to the repository.
