import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.controller.custom_chatbot import custom_chatbot
from data.geotiff_loader import GeoTIFFLoader
from backend.config import settings

client = TestClient(app)


def test_custom_chatbot_domain_query_without_image():
    res = custom_chatbot.chat("Suggest crop vegetable soil level etc")
    assert res["sender"] == "SatQuery Custom Chatbot"
    assert "Alluvial Loam" in res["reply"]
    assert "Wheat" in res["reply"]
    assert res["confidence"] >= 0.90


def test_custom_chatbot_black_cotton_soil():
    res = custom_chatbot.chat("What crops and fertilizer for black cotton soil with pH 7.8?")
    assert "Regur / Black Cotton Soil" in res["reply"]
    assert "Cotton" in res["reply"]
    assert "Single Super Phosphate" in res["reply"] or "NPK" in res["reply"]
    assert res["confidence"] >= 0.90


def test_custom_chatbot_spectral_indices():
    res = custom_chatbot.chat("How do I calculate NDVI and NDWI from Sentinel-2 bands?")
    assert "NDVI" in res["reply"]
    assert "(NIR - Red) / (NIR + Red)" in res["reply"]
    assert "NDWI" in res["reply"]
    assert res["task_type"] == "spectral_indices_guide"


def test_custom_chatbot_sar_vs_optical():
    res = custom_chatbot.chat("What is the difference between SAR radar and Optical imagery?")
    assert "Optical" in res["reply"]
    assert "SAR" in res["reply"]
    assert "microwave" in res["reply"].lower()


def test_custom_chatbot_with_image():
    sample_path = settings.sample_dir / "single_optical_scene.tif"
    if sample_path.exists():
        img = GeoTIFFLoader.load(sample_path, modality="optical")
        res = custom_chatbot.chat("Suggest crop vegetable soil level etc object highlight that part", images=[img])
        assert "Agricultural" in res["reply"] or "Alluvial" in res["reply"]
        assert len(res["visual_overlays"]["boxes"]) > 0
        assert res["task_type"] == "region_grounding"


def test_api_samples_catalog():
    response = client.get("/api/samples")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 4
    assert any(s["id"] == "single_optical" for s in data)
    assert any(s["id"] == "bitemporal_flood" for s in data)


def test_api_sample_load_endpoint():
    response = client.get("/api/samples/single_optical/load")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "single_optical"
    assert len(data["images"]) > 0
    assert "preview_base64" in data["images"][0]


def test_api_chat_endpoint_json():
    response = client.post("/api/chat", json={"query": "Suggest crop vegetable soil level"})
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert "Alluvial" in data["reply"] or "Soil" in data["reply"]


def test_custom_chatbot_laterite_and_arid_soils():
    res_lat = custom_chatbot.chat("Evaluate laterite soil profile, pH buffering with lime, and plantation crop suitability.")
    assert "Laterite Soil" in res_lat["reply"]
    assert "Cashew" in res_lat["reply"] or "Coffee" in res_lat["reply"] or "lime" in res_lat["reply"]

    res_arid = custom_chatbot.chat("Recommend drought-tolerant crops and sub-surface irrigation for arid sandy soil.")
    assert "Arid & Desert" in res_arid["reply"] or "Arid" in res_arid["reply"]
    assert "Millet" in res_arid["reply"] or "Bajra" in res_arid["reply"] or "Guar" in res_arid["reply"]


def test_custom_chatbot_infrastructure_and_satellites():
    res_airport = custom_chatbot.chat("Highlight the airport runway, taxiway, and tarmac infrastructure.")
    assert "Airport Runway" in res_airport["reply"] or "Runway" in res_airport["reply"]

    res_sat = custom_chatbot.chat("Compare spatial and spectral resolutions of Sentinel-2 vs ISRO Cartosat-2S and RISAT-1A.")
    assert "Cartosat-2S" in res_sat["reply"]
    assert "RISAT" in res_sat["reply"]
    assert "Sentinel-2" in res_sat["reply"]


def test_custom_chatbot_water_shoreline():
    res_water = custom_chatbot.chat("Delineate the lake shoreline and calculate total surface water coverage percentage.")
    assert "Water Body" in res_water["reply"] or "NDWI" in res_water["reply"]


def test_custom_chatbot_point_diagnostics():
    res_pt = custom_chatbot.chat("Analyze the agricultural crop vigor, soil moisture, and spectral NDVI at point location x: 55, y: 70 (18.5250° N, 73.8590° E).")
    assert res_pt["task_type"] == "point_and_query_diagnostic"
    assert res_pt["point_diagnostic"]["pixel_x"] == 55
    assert res_pt["point_diagnostic"]["pixel_y"] == 70
    assert "Point-and-Query Geospatial Diagnostic" in res_pt["reply"]
    assert len(res_pt["overlays"]["boxes"]) > 0

    res_pt_kn = custom_chatbot.chat("Analyze the agricultural crop vigor, soil moisture, and spectral NDVI at point location x: 120, y: 150.", language="kn-IN")
    assert res_pt_kn["task_type"] == "point_and_query_diagnostic"
    assert res_pt_kn["point_diagnostic"]["pixel_x"] == 120
    assert res_pt_kn["point_diagnostic"]["pixel_y"] == 150


