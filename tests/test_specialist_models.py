import pytest
from backend.config import settings
from backend.models.change.change_model import BiTemporalChangeModel
from backend.models.grounding.grounding_model import RegionGroundingModel
from backend.models.optical_sar_fusion.fusion_model import OpticalSARFusionModel
from backend.models.vqa_caption.vqa_model import RSVQACaptioningModel
from data.geotiff_loader import GeoTIFFLoader
from data.sample_data_generator import SampleDataGenerator


@pytest.fixture(scope="module")
def sample_data():
    SampleDataGenerator.generate_all(settings.sample_dir)
    return settings.sample_dir


def test_rs_vqa_model(sample_data):
    img = GeoTIFFLoader.load(sample_data / "single_optical_scene.tif", modality="optical")
    model = RSVQACaptioningModel()

    res = model.forward(img, query="What is the dominant land-cover visible in the scene?")
    assert "answer" in res
    assert res["confidence"] >= 0.85
    assert len(res["land_cover_classes"]) >= 1


def test_region_grounding_model(sample_data):
    img = GeoTIFFLoader.load(sample_data / "single_optical_scene.tif", modality="optical")
    model = RegionGroundingModel()

    res = model.forward(img, query="Highlight the water body referred to in the query.")
    assert res["matched_entity"] == "Water Body"
    assert len(res["boxes"]) >= 1
    assert "box_2d" in res["boxes"][0]
    assert len(res["boxes"][0]["box_2d"]) == 4
    # Check normalized coordinate ranges [0, 1]
    ymin, xmin, ymax, xmax = res["boxes"][0]["box_2d"]
    assert 0.0 <= ymin <= ymax <= 1.0
    assert 0.0 <= xmin <= xmax <= 1.0


def test_bitemporal_change_model(sample_data):
    t1 = GeoTIFFLoader.load(sample_data / "bitemporal_t1_pre_event.tif", modality="optical")
    t2 = GeoTIFFLoader.load(sample_data / "bitemporal_t2_post_event.tif", modality="optical")
    model = BiTemporalChangeModel()

    res = model.forward(t1, t2, query="Has the built-up area increased, decreased, or remained unchanged?")
    assert "increased" in res["change_direction"]
    assert res["change_ratio_pct"] > 5.0
    assert len(res["change_mask"]) in [64, 128]  # High-res change grid
    assert "boxes" in res
    assert res["confidence"] >= 0.90


def test_optical_sar_fusion_model(sample_data):
    opt = GeoTIFFLoader.load(sample_data / "cross_modal_optical.tif", modality="optical")
    sar = GeoTIFFLoader.load(sample_data / "cross_modal_sar.tif", modality="sar")
    model = OpticalSARFusionModel()

    res = model.forward(opt, sar, query="Use the optical and SAR images together to identify built-up and water-covered regions.")
    assert "fused_narrative" in res
    assert "optical_contributions" in res
    assert "sar_contributions" in res
    assert res["water_coverage_pct"] > 0
    assert res["builtup_coverage_pct"] > 0
    assert res["confidence"] >= 0.90
