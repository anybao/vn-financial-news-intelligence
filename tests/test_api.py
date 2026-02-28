from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_sentiment_endpoint():
    # Use mock/raw text that tests the endpoint availability
    payload = {"text": "Cổ phiếu Vinamilk hôm nay đột ngột sụt giảm trong phiên giao dịch."}
    response = client.post("/api/v1/sentiment", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data
    assert "confidence" in data

def test_ner_endpoint():
    payload = {"text": "Hòa Phát báo lãi kỷ lục."}
    response = client.post("/api/v1/ner", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "stocks" in data
    # Notice: 'Hòa Phát' will match HPG based on our rule-dictionary
    assert "HPG" in data["stocks"]
