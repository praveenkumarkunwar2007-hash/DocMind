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
