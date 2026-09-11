import pytest
from backend.config import settings
from data.geotiff_loader import GeoTIFFLoader
from data.sample_data_generator import SampleDataGenerator
from data.validator import InputValidator


@pytest.fixture(scope="module")
def sample_data():
    SampleDataGenerator.generate_all(settings.sample_dir)
    return settings.sample_dir


def test_single_image_validation(sample_data):
    img = GeoTIFFLoader.load(sample_data / "single_optical_scene.tif", modality="optical")
    res = InputValidator.validate([img])
    assert res.valid is True
    assert res.input_configuration == "single"
    assert res.image_count == 1
    assert "optical" in res.modalities_declared
    assert len(res.errors) == 0


def test_cross_modal_pair_validation(sample_data):
    img_opt = GeoTIFFLoader.load(sample_data / "cross_modal_optical.tif", modality="optical")
    img_sar = GeoTIFFLoader.load(sample_data / "cross_modal_sar.tif", modality="sar")

    res = InputValidator.validate([img_opt, img_sar], declared_task="optical_sar_fusion")
    assert res.valid is True
    assert res.input_configuration == "cross_modal_pair"
    assert res.image_count == 2
    assert res.co_registration_overlap_pct >= 80.0
    assert len(res.errors) == 0


def test_cross_modal_rejection_for_same_modality(sample_data):
    img1 = GeoTIFFLoader.load(sample_data / "cross_modal_optical.tif", modality="optical")
    img2 = GeoTIFFLoader.load(sample_data / "cross_modal_optical.tif", modality="optical")

    # Trying to do optical-SAR fusion with two optical images
    res = InputValidator.validate([img1, img2], declared_task="optical_sar_fusion")
    assert res.valid is False
    assert any("Cross-modal" in err for err in res.errors)


def test_bitemporal_pair_validation(sample_data):
    t1 = GeoTIFFLoader.load(sample_data / "bitemporal_t1_pre_event.tif", modality="optical")
    t2 = GeoTIFFLoader.load(sample_data / "bitemporal_t2_post_event.tif", modality="optical")

    res = InputValidator.validate([t1, t2], declared_task="change_description")
    assert res.valid is True
    assert res.input_configuration == "bi_temporal_pair"
    assert res.co_registration_overlap_pct >= 90.0


def test_bitemporal_rejection_when_only_one_image_provided(sample_data):
    t1 = GeoTIFFLoader.load(sample_data / "bitemporal_t1_pre_event.tif", modality="optical")

    res = InputValidator.validate([t1], declared_task="change_description")
    assert res.valid is False
    assert any("requires 2 co-registered images" in err for err in res.errors)
