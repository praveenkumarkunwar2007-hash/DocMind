---
title: DocMind AI Backend
emoji: 📄
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# DocMind AI

DocMind AI is a local-first, privacy-focused document assistant powered by FastAPI, SQLite, and Ollama.

## Features
- **Private Knowledge Base:** All documents, chunks, and history are stored locally in SQLite (`backend/data/docmind.db`).
- **Local AI:** Queries and summaries are generated locally via Ollama (`qwen2.5:0.5b`).
- **Multi-format Ingestion:** Supports `.pdf`, `.docx`, `.txt`, and `.md` files.
- **Three Modes:**
  - **Ask documents:** Retrieve relevant passages and answer questions.
  - **Summarize:** Create concise summaries of uploaded documents.
  - **Compare documents:** Contrast and compare key themes across multiple documents.

---

## Prerequisites
1. **Python 3.12** installed and added to PATH.
2. **Ollama** installed with the Qwen 2.5 (0.5B) model:
   ```bash
   ollama pull qwen2.5:0.5b
   ```
3. **Git LFS** (required if committing or pushing large model weights like `.pkl` to GitHub):
   ```bash
   git lfs install
   ```

---

## Quick Start on Windows

1. **Run Setup:**
   Double-click `setup_windows.bat` or run:
   ```cmd
   setup_windows.bat
   ```
   This creates the Python virtual environment (`.venv`) and installs all dependencies from `backend/requirements.txt`.

2. **Start the App:**
   Double-click `run_docmindai.bat` or run:
   ```cmd
   run_docmindai.bat
   ```
   Open your browser at [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

## GitHub Pages

The frontend is deployed from `frontend/` by the GitHub Actions workflow whenever changes are pushed to `main`. In the repository settings, open **Settings → Pages** and set **Build and deployment → Source** to **GitHub Actions**. The first deployment publishes the site at `https://praveenkumarkunwar2007-hash.github.io/DocMind/`.

GitHub Pages hosts only the static frontend; document and chat API requests use the backend at `https://docmind-y5n4.onrender.com`. The backend downloads `qwen2.5_0.5b_model.pkl` from [`Praveen081/DocMind-Qwen-Model`](https://huggingface.co/Praveen081/DocMind-Qwen-Model) when the file is absent. Chat inference is performed separately by Ollama; the checkpoint download does not start Ollama or provide inference. To enable chat on Render, configure `OLLAMA_URL` to a reachable Ollama server and ensure `MODEL_NAME` is installed there (the default is `qwen2.5:0.5b`). Set `DOCMIND_API_URL` in the browser's local storage to override the frontend API URL, or set `HF_MODEL_REPO` on the backend to override the checkpoint repository.

---

## Repository & Git LFS Notes

This repository contains large model files such as `backend/qwen2.5_0.5b_model.pkl` (~1.84 GB).
To push this repository to GitHub without exceeding file size limits:
1. Ensure Git LFS is initialized:
   ```bash
   git lfs install
   ```
2. `.gitattributes` is configured to track `*.pkl` with Git LFS:
   ```
   *.pkl filter=lfs diff=lfs merge=lfs -text
   ```
3. Stage, commit, and push normally:
   ```bash
   git add .
   git commit -m "DocMind AI with Git LFS tracking"
   git push origin main
   ```
