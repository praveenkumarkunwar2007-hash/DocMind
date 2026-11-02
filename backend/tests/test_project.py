"""Offline tests for API, session isolation, and inference provider behavior."""
import sqlite3

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import backend.app as backend
from backend.app import score_passage, split_text


@pytest.fixture
def isolated_database(monkeypatch, tmp_path):
    monkeypatch.setattr(backend, "DB_PATH", tmp_path / "test.sqlite")
    backend.initialize_database()


def test_split_text_returns_overlapping_passages():
    text = "abcdefghijklmnopqrstuvwxyz" * 100
    passages = split_text(text, size=100, overlap=20)
    assert len(passages) > 1
    assert all(passages)
    assert passages[0][80:100] == passages[1][:20]


def test_score_passage_matches_query_terms():
    assert score_passage("neural network", "A neural network can learn patterns") > 0
    assert score_passage("apple", "The document discusses pears and bananas") == 0


def test_stop_words_alone_do_not_match():
    assert score_passage("what is it", "What is it that this document covers") == 0


def test_health_endpoint_returns_ok(isolated_database):
    with TestClient(backend.app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_favicon_svg_is_served():
    with TestClient(backend.app) as client:
        response = client.get("/favicon.svg")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/svg+xml"
    assert "<svg" in response.text


def test_invalid_pdf_upload_returns_clean_client_error(isolated_database):
    with TestClient(backend.app) as client:
        response = client.post(
            "/api/upload",
            files={"file": ("invalid.pdf", b"not a PDF", "application/pdf")},
        )

    assert response.status_code == 400
    assert response.json() == {
        "success": False,
        "error": "Could not read this file. Ensure it is a valid PDF or DOCX.",
    }


def test_main_entrypoint_exports_same_application():
    from main import app as entrypoint_app

    assert entrypoint_app is backend.app


def test_unissued_session_cookie_is_replaced(isolated_database):
    with TestClient(backend.app) as client:
        client.cookies.set(backend.SESSION_COOKIE_NAME, "A" * 43)
        response = client.get("/api/session")
        issued_cookie = response.cookies.get(backend.SESSION_COOKIE_NAME)

    assert response.status_code == 200
    assert issued_cookie != "A" * 43
    assert len(issued_cookie) == 43


def test_pages_origin_gets_credentialed_cors_and_secure_cookie(isolated_database, monkeypatch):
    monkeypatch.setenv("RENDER", "true")
    with TestClient(backend.app, base_url="https://docmind-y5n4.onrender.com") as client:
        response = client.get(
            "/api/session",
            headers={"Origin": "https://praveenkumarkunwar2007-hash.github.io"},
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://praveenkumarkunwar2007-hash.github.io"
    assert response.headers["access-control-allow-credentials"] == "true"
    set_cookie = response.headers["set-cookie"].lower()
    assert "httponly" in set_cookie and "secure" in set_cookie and "samesite=none" in set_cookie


def test_cors_preflight_allows_methods_and_headers():
    with TestClient(backend.app) as client:
        response = client.options(
            "/api/chat",
            headers={
                "Origin": "http://localhost:5500",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type,x-custom-header",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5500"
    assert "POST" in response.headers["access-control-allow-methods"]
    assert response.headers["access-control-allow-headers"] == "content-type,x-custom-header"


def test_sessions_isolate_documents_chat_and_deletion(isolated_database, monkeypatch):
    monkeypatch.setattr(backend, "call_local_model", lambda prompt: prompt)
    with TestClient(backend.app) as user_a, TestClient(backend.app) as user_b:
        upload_a = user_a.post(
            "/api/upload",
            files={"file": ("document.txt", b"Alpha private content belongs to user A.", "text/plain")},
        )
        upload_b = user_b.post(
            "/api/upload",
            files={"file": ("document.txt", b"Beta private content belongs to user B.", "text/plain")},
        )
        assert upload_a.status_code == upload_b.status_code == 200
        document_a = upload_a.json()["document_id"]
        document_b = upload_b.json()["document_id"]

        assert [doc["name"] for doc in user_a.get("/api/documents").json()["documents"]] == ["document.txt"]
        assert [doc["name"] for doc in user_b.get("/api/documents").json()["documents"]] == ["document.txt"]
        assert user_b.delete(f"/api/documents/{document_a}").status_code == 404

        answer_a = user_a.post("/api/chat", json={"question": "Alpha", "mode": "ask"})
        answer_b = user_b.post("/api/chat", json={"question": "Beta", "mode": "ask"})
        assert answer_a.status_code == answer_b.status_code == 200
        assert "Alpha private content" in answer_a.json()["answer"]
        assert "Beta private content" not in answer_a.json()["answer"]
        assert "Beta private content" in answer_b.json()["answer"]
        assert "Alpha private content" not in answer_b.json()["answer"]

        assert len(user_a.get("/api/history").json()["messages"]) == 1
        assert len(user_b.get("/api/history").json()["messages"]) == 1
        assert user_a.delete(f"/api/documents/{document_a}").status_code == 200
        assert [doc["id"] for doc in user_b.get("/api/documents").json()["documents"]] == [document_b]


def test_chat_uses_configured_ollama_and_returns_source_passages(isolated_database, monkeypatch):
    monkeypatch.setattr(backend, "PROVIDER_ORDER", ("ollama",))
    monkeypatch.setattr(backend, "OLLAMA_URL", "http://ollama.internal:11434")
    monkeypatch.setattr(backend, "OLLAMA_MODEL", "hf.co/owner/DocMind-Qwen")
    requests = []

    class OllamaResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"response": "The document says DocMind uses Qwen [Source 1]."}

    def fake_post(url, **kwargs):
        requests.append((url, kwargs))
        return OllamaResponse()

    monkeypatch.setattr(backend.requests, "post", fake_post)
    with TestClient(backend.app) as client:
        upload = client.post(
            "/api/upload",
            files={"file": ("notes.txt", b"DocMind uses Qwen for document answers.", "text/plain")},
        )
        response = client.post("/api/chat", json={"question": "Qwen", "mode": "ask"})

    assert upload.status_code == 200
    assert response.status_code == 200
    assert response.json()["answer"].startswith("The document says DocMind uses Qwen")
    assert response.json()["sources"][0]["name"] == "notes.txt"
    assert "DocMind uses Qwen" in requests[0][1]["json"]["prompt"]
    assert requests[0][1]["json"]["model"] == "hf.co/owner/DocMind-Qwen"


def test_chat_provider_failure_returns_clean_json_error(isolated_database, monkeypatch):
    with TestClient(backend.app) as client:
        uploaded = client.post(
            "/api/upload",
            files={"file": ("notes.txt", b"A searchable topic for testing.", "text/plain")},
        )
        assert uploaded.status_code == 200

        def unavailable_provider(_prompt):
            raise HTTPException(status_code=503, detail="All configured providers failed (ollama: HTTP 404).")

        monkeypatch.setattr(backend, "call_local_model", unavailable_provider)
        response = client.post("/api/chat", json={"question": "searchable", "mode": "ask"})

    assert response.status_code == 503
    assert response.json() == {
        "success": False,
        "error": "AI provider temporarily unavailable",
        "detail": "All configured providers failed (ollama: HTTP 404).",
    }


def test_ollama_model_call_uses_correct_host_and_request_body(monkeypatch):
    monkeypatch.setattr(
        backend,
        "OLLAMA_URL",
        backend.normalize_ollama_url("https://ollama.example/Praveen081/DocMind-Qwen-Model/"),
    )
    monkeypatch.setattr(backend, "OLLAMA_MODEL", "hf.co/Praveen081/DocMind-Qwen-Model")
    monkeypatch.setattr(backend, "REQUEST_TIMEOUT", 120)
    calls = []

    class OllamaResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"response": "custom Qwen answer"}

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return OllamaResponse()

    monkeypatch.setattr(backend.requests, "post", fake_post)
    answer = backend._call_ollama("document context and question")

    assert answer == "custom Qwen answer"
    assert calls[0][0] == "https://ollama.example/api/generate"
    assert calls[0][1]["json"] == {
        "model": "hf.co/Praveen081/DocMind-Qwen-Model",
        "prompt": "document context and question",
        "stream": False,
        "keep_alive": "2m",
        "options": {"num_ctx": 2048, "num_predict": 300, "temperature": 0.2},
    }
    assert calls[0][1]["timeout"] >= 120


def test_model_call_uses_ollama_first_then_falls_back_to_groq(monkeypatch):
    monkeypatch.setattr(backend, "PROVIDER_ORDER", ("ollama", "groq", "hf"))
    monkeypatch.setattr(backend, "OLLAMA_URL", "https://ollama.example")
    monkeypatch.setattr(backend, "OLLAMA_MODEL", "hf.co/Praveen081/DocMind-Qwen-Model")
    monkeypatch.setattr(backend, "GROQ_API_KEY", "test-groq-key")
    monkeypatch.setattr(backend, "GROQ_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setattr(backend, "HF_TOKEN", "")
    monkeypatch.setattr(backend, "HF_MODEL", "")
    calls = []

    class OllamaResponse:
        status_code = 404

        @staticmethod
        def json():
            return {"error": "model not found"}

    class GroqResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "fallback answer"}}]}

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        if url.endswith("/api/generate"):
            return OllamaResponse()
        return GroqResponse()

    monkeypatch.setattr(backend.requests, "post", fake_post)
    assert backend.call_local_model("private document prompt") == "fallback answer"
    assert [call[0] for call in calls] == [
        "https://ollama.example/api/generate",
        "https://api.groq.com/openai/v1/chat/completions",
    ]
    assert calls[1][1]["json"]["model"] == "openai/gpt-oss-20b"
    assert calls[1][1]["json"]["messages"][0]["content"] == "private document prompt"
    assert calls[1][1]["headers"]["Authorization"] == f"Bearer {backend.GROQ_API_KEY}"
    assert all(call[1]["timeout"] >= 120 for call in calls)


def test_malformed_ollama_response_does_not_block_groq_fallback(monkeypatch):
    monkeypatch.setattr(backend, "PROVIDER_ORDER", ("ollama", "groq"))
    monkeypatch.setattr(backend, "OLLAMA_URL", "https://ollama.example")
    monkeypatch.setattr(backend, "OLLAMA_MODEL", "custom-qwen")
    monkeypatch.setattr(backend, "GROQ_API_KEY", "test-groq-key")
    monkeypatch.setattr(backend, "GROQ_MODEL", "openai/gpt-oss-20b")

    class MalformedOllamaResponse:
        status_code = 200

        @staticmethod
        def json():
            return []

    class GroqResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "fallback answer"}}]}

    def fake_post(url, **_kwargs):
        return MalformedOllamaResponse() if url.endswith("/api/generate") else GroqResponse()

    monkeypatch.setattr(backend.requests, "post", fake_post)
    assert backend.call_local_model("private document prompt") == "fallback answer"


def test_hugging_face_failure_does_not_block_groq_fallback(monkeypatch):
    monkeypatch.setattr(backend, "PROVIDER_ORDER", ("hf", "groq"))
    monkeypatch.setattr(backend, "HF_TOKEN", "test-hf-token")
    monkeypatch.setattr(backend, "HF_MODEL", "owner/optional-model")
    monkeypatch.setattr(backend, "GROQ_API_KEY", "test-groq-key")
    monkeypatch.setattr(backend, "GROQ_MODEL", "openai/gpt-oss-20b")
    calls = []

    class HuggingFaceResponse:
        status_code = 402

        @staticmethod
        def json():
            return {"error": "No remaining credits"}

    class GroqResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "Groq fallback"}}]}

    def fake_post(url, **kwargs):
        calls.append(url)
        if "huggingface.co" in url:
            return HuggingFaceResponse()
        return GroqResponse()

    monkeypatch.setattr(backend.requests, "post", fake_post)
    assert backend.call_local_model("private prompt") == "Groq fallback"
    assert calls == [
        "https://router.huggingface.co/v1/chat/completions",
        "https://api.groq.com/openai/v1/chat/completions",
    ]


def test_model_call_skips_unconfigured_providers(monkeypatch):
    monkeypatch.setattr(backend, "PROVIDER_ORDER", ("ollama", "groq", "hf"))
    monkeypatch.setattr(backend, "OLLAMA_URL", "")
    monkeypatch.setattr(backend, "OLLAMA_MODEL", "")
    monkeypatch.setattr(backend, "GROQ_API_KEY", "")
    monkeypatch.setattr(backend, "HF_TOKEN", "")
    monkeypatch.setattr(backend, "HF_MODEL", "")

    def unexpected_request(*_args, **_kwargs):
        pytest.fail("Unconfigured providers must not be called")

    monkeypatch.setattr(backend.requests, "post", unexpected_request)
    with pytest.raises(HTTPException) as error:
        backend.call_local_model("test prompt")

    assert error.value.status_code == 503
    assert "No inference provider is configured" in error.value.detail


def test_groq_is_skipped_when_its_model_is_empty(monkeypatch):
    monkeypatch.setattr(backend, "PROVIDER_ORDER", ("groq",))
    monkeypatch.setattr(backend, "GROQ_API_KEY", "test-groq-key")
    monkeypatch.setattr(backend, "GROQ_MODEL", "")
    monkeypatch.setattr(
        backend.requests,
        "post",
        lambda *_args, **_kwargs: pytest.fail("Groq with no model must be skipped"),
    )

    with pytest.raises(HTTPException) as error:
        backend.call_local_model("test prompt")

    assert error.value.status_code == 503
    assert "No inference provider is configured" in error.value.detail


def test_provider_failure_does_not_expose_html_or_api_keys(monkeypatch):
    monkeypatch.setattr(backend, "PROVIDER_ORDER", ("ollama",))
    monkeypatch.setattr(backend, "OLLAMA_URL", "http://ollama.invalid")
    monkeypatch.setattr(backend, "OLLAMA_MODEL", "custom-qwen")
    monkeypatch.setattr(backend, "GROQ_API_KEY", "never-return-this-key")

    class HtmlErrorResponse:
        status_code = 404

        @staticmethod
        def json():
            raise ValueError("not JSON")

    monkeypatch.setattr(
        backend.requests,
        "post",
        lambda *_args, **_kwargs: HtmlErrorResponse(),
    )
    with pytest.raises(HTTPException) as error:
        backend.call_local_model("private prompt")

    assert error.value.status_code == 503
    assert "HTTP 404" in error.value.detail
    assert "<html" not in error.value.detail.lower()
    assert "never-return-this-key" not in error.value.detail


def test_status_probes_ollama_model_availability(monkeypatch):
    monkeypatch.setattr(backend, "OLLAMA_URL", "http://ollama.invalid")
    monkeypatch.setattr(backend, "OLLAMA_MODEL", "custom-qwen")
    monkeypatch.setattr(backend, "GROQ_API_KEY", "")
    monkeypatch.setattr(backend, "HF_TOKEN", "")
    monkeypatch.setattr(backend, "HF_MODEL", "")

    class TagsResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"models": [{"name": "custom-qwen:latest"}]}

    monkeypatch.setattr(backend.requests, "get", lambda *_args, **_kwargs: TagsResponse())
    with TestClient(backend.app) as client:
        response = client.get("/api/status")

    assert response.status_code == 200
    assert response.json()["ollama_running"] is True
    assert response.json()["model_available"] is True
    assert response.json()["providers"]["ollama"]["reachable"] is True
    assert response.json()["provider"] == "Ollama"


def test_status_reports_configured_and_reachable_providers(monkeypatch):
    monkeypatch.setattr(backend, "PROVIDER_ORDER", ("ollama", "groq", "hf"))
    monkeypatch.setattr(backend, "OLLAMA_URL", "http://ollama.internal:11434")
    monkeypatch.setattr(backend, "OLLAMA_MODEL", "custom-qwen")
    monkeypatch.setattr(backend, "GROQ_API_KEY", "groq-test-key")
    monkeypatch.setattr(backend, "GROQ_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setattr(backend, "HF_TOKEN", "hf-test-key")
    monkeypatch.setattr(backend, "HF_MODEL", "owner/hf-model")

    class OllamaTags:
        status_code = 200

        @staticmethod
        def json():
            return {"models": [{"name": "custom-qwen"}]}

    class ProviderModels:
        status_code = 200

        @staticmethod
        def json():
            return {
                "data": [
                    {"id": "openai/gpt-oss-20b"},
                    {"id": "owner/hf-model"},
                ]
            }

    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        if url.endswith("/api/tags"):
            return OllamaTags()
        return ProviderModels()

    monkeypatch.setattr(backend.requests, "get", fake_get)
    with TestClient(backend.app) as client:
        response = client.get("/api/status")

    providers = response.json()["providers"]
    assert response.status_code == 200
    assert providers["ollama"]["reachable"] is True
    assert providers["groq"]["reachable"] is True
    assert providers["hf"]["reachable"] is True
    assert response.json()["provider"] == "Ollama"
    assert calls[1][1]["headers"]["Authorization"] == f"Bearer {backend.GROQ_API_KEY}"
    assert calls[2][1]["headers"]["Authorization"] == f"Bearer {backend.HF_TOKEN}"


def test_malformed_ollama_status_does_not_block_groq_probe(monkeypatch):
    monkeypatch.setattr(backend, "PROVIDER_ORDER", ("ollama", "groq"))
    monkeypatch.setattr(backend, "OLLAMA_URL", "http://ollama.internal:11434")
    monkeypatch.setattr(backend, "OLLAMA_MODEL", "custom-qwen")
    monkeypatch.setattr(backend, "GROQ_API_KEY", "test-groq-key")
    monkeypatch.setattr(backend, "GROQ_MODEL", "openai/gpt-oss-20b")

    class MalformedOllamaTags:
        status_code = 200

        @staticmethod
        def json():
            return {"models": None}

    class GroqModels:
        status_code = 200

        @staticmethod
        def json():
            return {"data": [{"id": "openai/gpt-oss-20b"}]}

    def fake_get(url, **_kwargs):
        return MalformedOllamaTags() if url.endswith("/api/tags") else GroqModels()

    monkeypatch.setattr(backend.requests, "get", fake_get)
    with TestClient(backend.app) as client:
        response = client.get("/api/status")

    assert response.status_code == 200
    assert response.json()["providers"]["ollama"]["reachable"] is False
    assert response.json()["providers"]["groq"]["reachable"] is True
    assert response.json()["provider"] == "Groq"


def test_legacy_shared_rows_are_not_visible_to_new_sessions(tmp_path, monkeypatch):
    legacy_db = tmp_path / "legacy.sqlite"
    monkeypatch.setattr(backend, "DB_PATH", legacy_db)
    with sqlite3.connect(legacy_db) as db:
        db.executescript(
            """
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                content TEXT NOT NULL,
                uploaded_at TEXT NOT NULL
            );
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                content TEXT NOT NULL
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mode TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            INSERT INTO documents (name, content, uploaded_at)
            VALUES ('old.txt', 'legacy private content', '2026-01-01');
            """
        )

    backend.initialize_database()
    assert backend.list_documents("new-session") == []
