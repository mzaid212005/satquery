import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.security.auth import (
    hash_password,
    verify_password,
    generate_token,
    verify_token,
    user_repository,
    User,
)
from backend.controller.spatial_nlp import spatial_nlp
from backend.controller.custom_chatbot import custom_chatbot
from data.geotiff_loader import GeoTIFFLoader
from backend.config import settings

client = TestClient(app)


def test_password_hashing():
    pwd = "SecurePassword@2026"
    hash_val, salt = hash_password(pwd)
    assert verify_password(pwd, hash_val, salt) is True
    assert verify_password("WrongPassword", hash_val, salt) is False


def test_token_lifecycle():
    user = User(
        id="usr_test_123",
        email="test.analyst@satquery.ai",
        full_name="Test Analyst",
        role="Geospatial Analyst",
        organization="Test Lab",
        is_active=True,
        created_at=1700000000.0,
    )
    token = generate_token(user)
    assert token is not None
    assert "." in token

    payload = verify_token(token)
    assert payload is not None
    assert payload["sub"] == "usr_test_123"
    assert payload["email"] == "test.analyst@satquery.ai"
    assert payload["role"] == "Geospatial Analyst"


def test_auth_login_admin():
    response = client.post(
        "/api/auth/login",
        json={"email": "admin@satquery.ai", "password": "Admin@1234"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "token" in data
    assert data["user"]["role"] == "Administrator"


def test_auth_login_invalid():
    response = client.post(
        "/api/auth/login",
        json={"email": "admin@satquery.ai", "password": "IncorrectPassword"},
    )
    assert response.status_code == 401


def test_auth_register_and_me():
    email = "researcher.new@satquery.ai"
    reg_response = client.post(
        "/api/auth/register",
        json={
            "full_name": "Dr. Sarah Connor",
            "email": email,
            "password": "Password@123",
            "role": "Researcher",
            "organization": "Satellite Dynamics Lab",
        },
    )
    # Either 200 if new or 409 if already registered from earlier test
    if reg_response.status_code == 200:
        data = reg_response.json()
        token = data["token"]
        assert data["user"]["role"] == "Researcher"
    else:
        login_res = client.post(
            "/api/auth/login",
            json={"email": email, "password": "Password@123"},
        )
        token = login_res.json()["token"]

    # Verify /api/auth/me with Bearer token
    me_response = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert me_response.status_code == 200
    me_data = me_response.json()
    assert me_data["user"]["email"] == email


def test_spatial_nlp_kannada():
    """Verify Kannada pure language translation and speech output."""
    sample_path = settings.sample_dir / "single_optical_scene.tif"
    img = GeoTIFFLoader.load(sample_path, modality="optical")

    # Kannada point diagnostic
    diag_kn = spatial_nlp.analyze_point_location(img, 128, 128, language="kn-IN")
    assert "speech_text" in diag_kn
    # Ensure Kannada script is present
    assert "ಅಕ್ಷಾಂಶ" in diag_kn["speech_text"] or "ವಿಶ್ಲೇಷಣೆ" in diag_kn["speech_text"]
    assert "trace" in diag_kn
    assert diag_kn["trace"]["total_latency_ms"] > 0
    assert len(diag_kn["trace"]["steps"]) >= 2


def test_custom_chatbot_kannada():
    """Verify Kannada conversational assistant response."""
    res_kn = custom_chatbot.respond("ಕಪ್ಪು ಹತ್ತಿ ಮಣ್ಣು ಮತ್ತು ಬೆಳೆಗಳ ಬಗ್ಗೆ ತಿಳಿಸಿ", language="kn-IN")
    assert res_kn["sender"] == "SatQuery Custom Chatbot"
    assert "ಕೃಷಿ" in res_kn["reply"] or "ಮಣ್ಣು" in res_kn["reply"]
    assert "trace" in res_kn or "execution_trace" in res_kn


def test_google_auth_endpoint():
    """Verify POST /api/auth/google auto-provisions and returns JWT."""
    response = client.post(
        "/api/auth/google",
        json={
            "email": "astronaut.cooper@nasa.gov",
            "name": "Joseph Cooper",
            "picture": "https://example.com/cooper.jpg",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "token" in data
    assert data["user"]["email"] == "astronaut.cooper@nasa.gov"
    assert data["user"]["full_name"] == "Joseph Cooper"
    assert data["user"]["role"] == "Geospatial Analyst"

