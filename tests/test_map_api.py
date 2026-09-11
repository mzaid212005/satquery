import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.controller.custom_chatbot import custom_chatbot

client = TestClient(app)


def test_get_carto_dark_config():
    response = client.get("/api/map/carto_dark")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["layer_id"] == "carto_dark"
    assert "basemaps.cartocdn.com" in data["tile_url"]
    assert "/api/map/tiles/carto_dark" in data["proxy_url_template"]
    assert data["max_zoom"] == 20
    assert data["theme"] == "dark"


def test_get_map_layers():
    response = client.get("/api/map/layers")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["default_layer"] == "carto_dark"
    layer_ids = [l["id"] for l in data["layers"]]
    assert "carto_dark" in layer_ids
    assert "esri_satellite" in layer_ids
    assert "osm" in layer_ids


def test_get_carto_dark_tile_proxy():
    # Test valid tile coordinate (zoom 0, x 0, y 0)
    response = client.get("/api/map/tiles/carto_dark/0/0/0.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 50

    # Test invalid coordinate
    res_inv = client.get("/api/map/tiles/carto_dark/0/0/invalid.png")
    assert res_inv.status_code == 400


def test_carto_dark_chatbot_knowledge():
    res_en = custom_chatbot.chat("Explain the CARTO Dark Matter Basemap API and how to use it in GIS mode.")
    assert "CARTO Dark" in res_en["reply"]
    assert "/api/map/carto_dark" in res_en["reply"]
    assert res_en["task_type"] == "carto_dark_map_api"

    res_kn = custom_chatbot.chat("ಕಾರ್ಟೊ ಡಾರ್ಕ್ ಮ್ಯಾಪ್ API ಬಗ್ಗೆ ತಿಳಿಸಿ")
    assert "CARTO" in res_kn["reply"]
    assert "ಬೇಸ್‌ಮ್ಯಾಪ್" in res_kn["reply"]
