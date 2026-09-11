import pytest
from backend.config import settings
from backend.controller.orchestrator import orchestrator
from data.geotiff_loader import GeoTIFFLoader
from data.sample_data_generator import SampleDataGenerator


@pytest.fixture(scope="module")
def sample_data():
    SampleDataGenerator.generate_all(settings.sample_dir)
    return settings.sample_dir


def test_representative_query_1_captioning(sample_data):
    """Representative Query 1: Describe the land-cover and major objects visible in this image."""
    img = GeoTIFFLoader.load(sample_data / "single_optical_scene.tif", modality="optical")
    query = "Describe the land-cover and major objects visible in this image."

    res = orchestrator.execute(query=query, images=[img])
    assert res.status == "success"
    assert res.task_type in ["captioning", "scene_description"]
    assert len(res.answer) > 40
    assert res.confidence >= 0.85
    assert len(res.trace.steps) >= 1
    assert res.trace.total_latency_ms > 0


def test_representative_query_2_grounding(sample_data):
    """Representative Query 2: Highlight the water body referred to in the query."""
    img = GeoTIFFLoader.load(sample_data / "single_optical_scene.tif", modality="optical")
    query = "Highlight the water body referred to in the query."

    res = orchestrator.execute(query=query, images=[img])
    assert res.status == "success"
    assert res.task_type == "region_grounding"
    assert len(res.visual_overlays.get("boxes", [])) >= 1
    assert res.visual_overlays.get("matched_entity") == "Water Body"
    assert res.confidence >= 0.85


def test_representative_query_3_change_description(sample_data):
    """Representative Query 3: What changed between these two dates, and where did the change occur?"""
    t1 = GeoTIFFLoader.load(sample_data / "bitemporal_t1_pre_event.tif", modality="optical")
    t2 = GeoTIFFLoader.load(sample_data / "bitemporal_t2_post_event.tif", modality="optical")
    query = "What changed between these two dates, and where did the change occur?"

    res = orchestrator.execute(query=query, images=[t1, t2])
    assert res.status == "success"
    assert res.task_type in ["change_description", "temporal_change"]
    assert "change_direction" in res.visual_overlays
    assert res.visual_overlays["change_ratio_pct"] > 5.0
    assert len(res.visual_overlays.get("change_mask", [])) > 0
    assert res.confidence >= 0.90


def test_representative_query_4_optical_sar_fusion(sample_data):
    """Representative Query 4: Use the optical and SAR images together to identify built-up and water-covered regions."""
    opt = GeoTIFFLoader.load(sample_data / "cross_modal_optical.tif", modality="optical")
    sar = GeoTIFFLoader.load(sample_data / "cross_modal_sar.tif", modality="sar")
    query = "Use the optical and SAR images together to identify built-up and water-covered regions."

    res = orchestrator.execute(query=query, images=[opt, sar])
    assert res.status == "success"
    assert res.task_type == "optical_sar_fusion"
    assert "optical_contributions" in res.metadata
    assert res.visual_overlays.get("water_coverage_pct", 0) > 0
    assert res.confidence >= 0.90


def test_representative_query_5_change_vqa(sample_data):
    """Representative Query 5: Has the built-up area increased, decreased, or remained unchanged?"""
    t1 = GeoTIFFLoader.load(sample_data / "bitemporal_t1_pre_event.tif", modality="optical")
    t2 = GeoTIFFLoader.load(sample_data / "bitemporal_t2_post_event.tif", modality="optical")
    query = "Has the built-up area increased, decreased, or remained unchanged?"

    res = orchestrator.execute(query=query, images=[t1, t2])
    assert res.status == "success"
    assert res.task_type == "change_vqa"
    assert res.visual_overlays["change_direction"] == "increased"
    assert "increased" in res.answer.lower()
    assert res.confidence >= 0.90


def test_representative_query_spatial_point_query(sample_data):
    """Spatial NLP: Point-and-Query localized NDVI & soil analysis."""
    img = GeoTIFFLoader.load(sample_data / "single_optical_scene.tif", modality="optical")
    query = "Analyze the agricultural crop vigor, soil moisture, and spectral NDVI at point location x: 128, y: 140."

    res = orchestrator.execute(query=query, images=[img])
    assert res.status == "success"
    assert res.task_type == "point_and_query_diagnostic"
    assert "point_diagnostic" in res.metadata
    assert res.metadata["point_diagnostic"]["pixel_x"] == 128
    assert res.metadata["point_diagnostic"]["pixel_y"] == 140
    assert "ndvi" in res.metadata["point_diagnostic"]
    assert "ndwi" in res.metadata["point_diagnostic"]
    assert len(res.visual_overlays.get("boxes", [])) >= 1
    assert res.confidence >= 0.90


def test_representative_query_quadrant_analysis(sample_data):
    """Spatial NLP: Quadrant and spatial entity grounding."""
    img = GeoTIFFLoader.load(sample_data / "single_optical_scene.tif", modality="optical")
    query = "Identify what land-cover feature exists in the north-west quadrant and water body in the southern area."

    res = orchestrator.execute(query=query, images=[img])
    assert res.status == "success"
    assert res.confidence >= 0.85


def test_incompatible_input_rejection():
    """Confirms rejection when single image is supplied for bi-temporal query."""
    img = GeoTIFFLoader.load(settings.sample_dir / "single_optical_scene.tif", modality="optical")
    query = "Has the built-up area increased, decreased, or remained unchanged?"

    res = orchestrator.execute(query=query, images=[img])
    assert res.status == "rejected"
    assert res.confidence == 0.0
    assert "rejected" in res.answer.lower()
    assert res.trace.compatibility_status == "rejected"


def test_orchestrator_gis_auto_fallback(sample_data):
    """Verifies orchestrator automatically loads GIS optical scene when no image is uploaded."""
    query = "Highlight the water body referred to in the query."
    res = orchestrator.execute(query=query, images=None)
    assert res.status == "success"
    assert res.task_type == "region_grounding"
    assert len(res.visual_overlays.get("boxes", [])) >= 1
    assert res.confidence >= 0.85


def test_spatial_nlp_multilingual(sample_data):
    """Verifies SpatialNLPEngine produces localized speech output in selected language."""
    from backend.controller.spatial_nlp import spatial_nlp
    img = GeoTIFFLoader.load(sample_data / "single_optical_scene.tif", modality="optical")

    # Hindi
    diag_hi = spatial_nlp.analyze_point_location(img, 128, 128, language="hi-IN")
    assert "speech_text" in diag_hi
    assert "अक्षांश" in diag_hi["speech_text"] or "बिंदु" in diag_hi["speech_text"] or "विश्लेषण" in diag_hi["speech_text"]

    # Kannada
    diag_kn = spatial_nlp.analyze_point_location(img, 128, 128, language="kn-IN")
    assert "speech_text" in diag_kn
    assert "ಅಕ್ಷಾಂಶ" in diag_kn["speech_text"] or "ಪಾಯಿಂಟ್" in diag_kn["speech_text"] or "ವಿಶ್ಲೇಷಣೆ" in diag_kn["speech_text"]

    # Spanish
    diag_es = spatial_nlp.analyze_point_location(img, 128, 128, language="es-ES")
    assert "speech_text" in diag_es
    assert "Diagnóstico" in diag_es["speech_text"] or "latitud" in diag_es["speech_text"]

    # German
    diag_de = spatial_nlp.analyze_point_location(img, 128, 128, language="de-DE")
    assert "speech_text" in diag_de
    assert "Punktanalyse" in diag_de["speech_text"] or "Breitengrad" in diag_de["speech_text"]


def test_custom_chatbot_kannada_domain(sample_data):
    """Verifies CustomSatChatbot returns pure Kannada for soil, water, agriculture and image-grounded queries."""
    from backend.controller.custom_chatbot import custom_chatbot
    img = GeoTIFFLoader.load(sample_data / "single_optical_scene.tif", modality="optical")

    # 1. Soil Knowledge in Kannada
    soil_res = custom_chatbot.respond("ಕಪ್ಪು ಹತ್ತಿ ಮಣ್ಣು ಗುಣಲಕ್ಷಣಗಳು", language="kn-IN")
    assert "ಮಣ್ಣಿನ ಪ್ರೊಫೈಲ್" in soil_res["reply"] or "ಕಪ್ಪು ಹತ್ತಿ ಮಣ್ಣು" in soil_res["reply"]
    assert "ಶಿಫಾರಸು ಮಾಡಲಾದ ಪ್ರಮುಖ ಬೆಳೆಗಳು" in soil_res["reply"]

    # 2. Precision Agriculture in Kannada
    agri_res = custom_chatbot.respond("ಬೆಳೆ, ತರಕಾರಿ ಮತ್ತು ಮಣ್ಣಿನ ಮಟ್ಟದ ಸಲಹೆ ನೀಡಿ", language="kn-IN")
    assert "ನಿಖರ ಕೃಷಿ ಮತ್ತು ಸಮಗ್ರ ಮಣ್ಣಿನ ಮಟ್ಟದ ಸಲಹೆ" in agri_res["reply"]
    assert "ಮೆಕ್ಕಲು ಮಣ್ಣು" in agri_res["reply"]
    assert "ಕಪ್ಪು ಹತ್ತಿ ಮಣ್ಣು" in agri_res["reply"]

    # 3. Water Body Delineation in Kannada
    water_res = custom_chatbot.respond("ನೀರು ಮತ್ತು ನದಿ ಗುರುತಿಸುವಿಕೆ", language="kn-IN")
    assert "ಜಲಮೂಲ ಗುರುತಿಸುವಿಕೆ" in water_res["reply"] or "MNDWI" in water_res["reply"]

    # 4. Image-Grounded Query in Kannada
    img_res = custom_chatbot.respond("ಬೆಳೆ ಮತ್ತು ಮಣ್ಣು ಹೈಲೈಟ್ ಮಾಡಿ", images=[img], language="kn-IN")
    assert "SatQuery AI ಕಸ್ಟಮ್ ಚಾಟ್‌ಬಾಟ್" in img_res["reply"]
    assert "ಗುರಿ ಮಣ್ಣು" in img_res["reply"] or "ಮಣ್ಣಿನ ಮಟ್ಟ" in img_res["reply"]


