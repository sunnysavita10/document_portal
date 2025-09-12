import os
from unittest.mock import patch
from fastapi.testclient import TestClient
from api.main import app
from tests.test_utils import create_dummy_pdf
import pandas as pd

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "document-portal"}


@patch("api.main.DocumentAnalyzer")
def test_analyze_document(mock_analyzer):
    mock_analyzer.return_value.analyze_document.return_value = {
        "keys": ["key1", "key2"]
    }
    dummy_pdf_path = "tests/dummy.pdf"
    create_dummy_pdf(dummy_pdf_path)
    with open(dummy_pdf_path, "rb") as f:
        response = client.post(
            "/analyze", files={"file": ("dummy.pdf", f, "application/pdf")}
        )
    os.remove(dummy_pdf_path)
    assert response.status_code == 200
    assert "keys" in response.json()


@patch("api.main.DocumentComparatorLLM")
def test_compare_documents(mock_comparator):
    mock_comparator.return_value.compare_documents.return_value = pd.DataFrame(
        {"col1": [1], "col2": [2]}
    )
    dummy_pdf_path_1 = "tests/dummy1.pdf"
    dummy_pdf_path_2 = "tests/dummy2.pdf"
    create_dummy_pdf(dummy_pdf_path_1)
    create_dummy_pdf(dummy_pdf_path_2)
    with open(dummy_pdf_path_1, "rb") as f1, open(dummy_pdf_path_2, "rb") as f2:
        response = client.post(
            "/compare",
            files={
                "reference": ("dummy1.pdf", f1, "application/pdf"),
                "actual": ("dummy2.pdf", f2, "application/pdf"),
            },
        )
    os.remove(dummy_pdf_path_1)
    os.remove(dummy_pdf_path_2)
    assert response.status_code == 200
    assert "rows" in response.json()
    assert "session_id" in response.json()


@patch("api.main.ChatIngestor")
def test_chat_build_index(mock_ingestor):
    mock_ingestor.return_value.build_retriever.return_value = None
    mock_ingestor.return_value.session_id = "dummy_session_id"
    dummy_pdf_path = "tests/dummy.pdf"
    create_dummy_pdf(dummy_pdf_path)
    with open(dummy_pdf_path, "rb") as f:
        response = client.post(
            "/chat/index",
            files={"files": ("dummy.pdf", f, "application/pdf")},
            data={"session_id": "dummy_session_id"},
        )
    os.remove(dummy_pdf_path)
    assert response.status_code == 200
    assert "session_id" in response.json()


@patch("api.main.ConversationalRAG")
def test_chat_query(mock_rag):
    mock_rag.return_value.invoke.return_value = "This is a dummy answer."
    index_dir = "faiss_index/dummy_session_id"
    os.makedirs(index_dir, exist_ok=True)
    response = client.post(
        "/chat/query",
        data={
            "question": "What is the meaning of life?",
            "session_id": "dummy_session_id",
        },
    )
    os.rmdir(index_dir)
    assert response.status_code == 200
    assert "answer" in response.json()
    assert "session_id" in response.json()
