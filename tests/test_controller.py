import pytest
from backend.controller.compatibility_checker import CompatibilityChecker
from backend.controller.query_interpreter import QueryInterpreter
from backend.controller.registry_loader import registry_loader
from backend.controller.trace import TraceRecorder
from backend.config import settings
from data.geotiff_loader import GeoTIFFLoader
from data.sample_data_generator import SampleDataGenerator


@pytest.fixture(scope="module")
def sample_data():
    SampleDataGenerator.generate_all(settings.sample_dir)
    return settings.sample_dir


def test_registry_loading():
    registry = registry_loader.registry
    assert len(registry.models) == 4
    assert "rs_vqa_captioning" in registry.models
    assert "region_grounding" in registry.models
    assert "change_analysis" in registry.models
    assert "optical_sar_fusion" in registry.models


def test_parameter_whitelisting():
    model_entry = registry_loader.get_model_entry_by_task("region_grounding")
    assert model_entry is not None

    # Test parameter filtering with an invalid and out-of-range parameter
    raw_params = {
        "confidence_threshold": 0.85,
        "max_regions": 999,  # max is 20 in schema
        "unauthorized_secret_key": "hack_attempt",
    }
    sanitized = registry_loader.validate_and_filter_parameters(model_entry, raw_params)

    assert "unauthorized_secret_key" not in sanitized
    assert sanitized["confidence_threshold"] == 0.85
    assert sanitized["max_regions"] == 20  # Clamped to max
    assert sanitized["generate_heatmap"] is True  # Default applied


def test_query_interpreter_tasks():
    # 1. Grounding
    q1 = QueryInterpreter.interpret("Highlight the water body referred to in the query.")
    assert q1.task_type == "region_grounding"
    assert "water body" in q1.target_entities

    # 2. Change VQA
    q2 = QueryInterpreter.interpret("Has the built-up area increased, decreased, or remained unchanged?", image_count=2)
    assert q2.task_type == "change_vqa"
    assert q2.required_relationship == "bi_temporal"

    # 3. Optical SAR Fusion
    q3 = QueryInterpreter.interpret("Use the optical and SAR images together to identify built-up and water-covered regions.")
    assert q3.task_type == "optical_sar_fusion"
    assert q3.required_relationship == "cross_modal"


def test_auditable_trace_recording():
    recorder = TraceRecorder(trace_id="test_trace_001", query="Describe this image", inferred_task="captioning")
    recorder.record_step(
        step_name="Test Step",
        tool_or_model="RSVQACaptioningModel",
        duration_ms=12.5,
        parameters={"max_tokens": 128},
        summary="Test execution success",
    )
    trace = recorder.finalize(final_confidence=0.94)

    audit_dict = trace.to_audit_dict()
    assert audit_dict["trace_id"] == "test_trace_001"
    assert audit_dict["status"] == "completed"
    assert audit_dict["final_confidence"] == 0.94
    assert len(audit_dict["steps"]) == 1
    assert "chain_of_thought" not in audit_dict
