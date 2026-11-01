"""Small offline tests for utility functions; they do not call Ollama."""
import sqlite3

import pytest
from fastapi.testclient import TestClient

import app as backend
from app import score_passage, split_text


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


def test_chat_provider_failure_returns_json_error(isolated_database, monkeypatch):
    with TestClient(backend.app) as client:
        uploaded = client.post(
            "/api/upload",
            files={"file": ("notes.txt", b"A searchable topic for testing.", "text/plain")},
        )
        assert uploaded.status_code == 200

        def unavailable_provider(_prompt):
            from fastapi import HTTPException

            raise HTTPException(status_code=503, detail="No AI provider is available.")

        monkeypatch.setattr(backend, "call_local_model", unavailable_provider)
        response = client.post("/api/chat", json={"question": "searchable", "mode": "ask"})

    assert response.status_code == 503
    assert response.json() == {
        "success": False,
        "error": "No AI provider is available.",
    }


def test_model_call_uses_configured_hugging_face_credential(monkeypatch):
    monkeypatch.setattr(backend, "HF_TOKEN", "test-token")
    monkeypatch.setattr(backend, "GROQ_API_KEY", "")
    calls = []

    class ProviderResponse:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "provider answer"}}]}

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return ProviderResponse()

    monkeypatch.setattr(backend.requests, "post", fake_post)
    assert backend.call_local_model("test prompt") == "provider answer"
    assert calls[0][1]["headers"]["Authorization"] == "Bearer test-token"


def test_model_call_returns_service_unavailable_when_providers_are_offline(monkeypatch):
    monkeypatch.setattr(backend, "HF_TOKEN", "")
    monkeypatch.setattr(backend, "GROQ_API_KEY", "")
    monkeypatch.setattr(backend, "OLLAMA_URL", "http://ollama.invalid")

    def unavailable_provider(*_args, **_kwargs):
        raise backend.requests.ConnectionError("provider offline")

    monkeypatch.setattr(backend.requests, "post", unavailable_provider)
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as error:
        backend.call_local_model("test prompt")

    assert error.value.status_code == 503
    assert "No AI provider is available" in error.value.detail


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
