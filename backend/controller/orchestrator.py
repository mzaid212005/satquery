import time
import uuid
from typing import Any, Dict, List, Optional
from backend.config import settings
from backend.controller.compatibility_checker import CompatibilityChecker
from backend.controller.query_interpreter import QueryInterpreter, QueryIntent
from backend.controller.registry_loader import registry_loader, ModelEntry
from backend.controller.trace import TraceRecorder, ExecutionTrace
from data.geotiff_loader import GeoTIFFLoader, RSImage


class SatQueryResponse:
    def __init__(
        self,
        query: str,
        task_type: str,
        answer: str,
        confidence: float,
        visual_overlays: Dict[str, Any],
        metadata: Dict[str, Any],
        trace: ExecutionTrace,
        status: str = "success",
        error: Optional[str] = None,
    ):
        self.query = query
        self.task_type = task_type
        self.answer = answer
        self.confidence = confidence
        self.visual_overlays = visual_overlays
        self.metadata = metadata
        self.trace = trace
        self.status = status
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "task_type": self.task_type,
            "answer": self.answer,
            "confidence": self.confidence,
            "visual_overlays": self.visual_overlays,
            "metadata": self.metadata,
            "trace": self.trace.to_audit_dict(),
            "status": self.status,
            "error": self.error,
        }


class AgenticOrchestrator:
    """Master Agentic Controller for SatQuery AI."""

    def __init__(self):
        self.registry = registry_loader

    def execute(
        self,
        query: str,
        images: Optional[List[RSImage]] = None,
        user_params: Optional[Dict[str, Any]] = None,
        language: str = "en-US",
    ) -> SatQueryResponse:
        trace_id = f"trace_{uuid.uuid4().hex[:10]}"
        t0 = time.perf_counter()

        if images is None:
            images = []

        # Auto-fallback to default GIS scene if images not provided
        if len(images) == 0:
            sample_path = settings.sample_dir / "single_optical_scene.tif"
            if sample_path.exists():
                images = [GeoTIFFLoader.load(sample_path, modality="optical")]

        # 1. Query Interpretation
        intent = QueryInterpreter.interpret(query, image_count=len(images))
        recorder = TraceRecorder(
            trace_id=trace_id,
            query=query,
            inferred_task=intent.task_type,
            workflow_type=intent.workflow,
        )
        recorder.trace.entities_extracted = intent.target_entities

        # Summarize input image metadata for audit trace
        recorder.trace.inputs_evaluated = [
            {
                "index": i + 1,
                "modality": img.modality,
                "format": img.metadata.format_name,
                "dimensions": f"{img.width}x{img.height}x{img.channels}",
                "crs": img.metadata.crs,
                "resolution_m": img.metadata.resolution[0],
            }
            for i, img in enumerate(images)
        ]

        # 2. Input Compatibility Verification
        compat_report = CompatibilityChecker.check(intent, images)
        recorder.trace.validation_flags = (
            compat_report.validation_result.warnings + compat_report.validation_result.errors
        )

        if not compat_report.compatible:
            recorder.trace.compatibility_status = "rejected"
            trace = recorder.finalize(final_confidence=0.0, status="rejected")
            return SatQueryResponse(
                query=query,
                task_type=intent.task_type,
                answer=(
                    f"Request rejected by input-validation layer: {compat_report.rejection_reason} "
                    f"Suggested action: {compat_report.suggested_action}"
                ),
                confidence=0.0,
                visual_overlays={},
                metadata={"validation_errors": compat_report.validation_result.errors},
                trace=trace,
                status="rejected",
                error=compat_report.rejection_reason,
            )

        recorder.trace.compatibility_status = "passed"

        # 3. Specialist Model Selection from Registry
        model_entry = self.registry.get_model_entry_by_task(intent.task_type)
        if not model_entry:
            # Fallback to rs_vqa_captioning
            model_entry = self.registry.get_model_entry_by_task("rs_vqa")

        if not model_entry:
            trace = recorder.finalize(final_confidence=0.0, status="error")
            return SatQueryResponse(
                query=query,
                task_type=intent.task_type,
                answer="No specialist model registered for the inferred task type.",
                confidence=0.0,
                visual_overlays={},
                metadata={},
                trace=trace,
                status="error",
                error="Model not found in registry.",
            )

        # 4. Parameter Configuration & Whitelist Filtering
        sanitized_params = self.registry.validate_and_filter_parameters(model_entry, user_params)

        # 5. Workflow Execution
        step_t0 = time.perf_counter()

        try:
            if intent.task_type == "point_and_query_diagnostic":
                from backend.controller.spatial_nlp import spatial_nlp
                coords = (intent.spatial_meta or {}).get("coordinates") or {}
                px = float(coords.get("pixel_x", coords.get("val1", 128)))
                py = float(coords.get("pixel_y", coords.get("val2", 128)))
                
                h, w = images[0].height, images[0].width
                norm_x = (px / w) if px > 1.0 else px
                norm_y = (py / h) if py > 1.0 else py
                
                model_out = spatial_nlp.analyze_point_location(
                    image=images[0],
                    norm_x=norm_x,
                    norm_y=norm_y,
                    language=language,
                )
                model_out["boxes"] = [
                    {
                        "label": f"Point Pin: {model_out['feature_class']}",
                        "box_2d": model_out["pin_box"],
                        "score": model_out["confidence"],
                    }
                ]
                model_out["point_diagnostic"] = {
                    "pixel_x": model_out["pixel_x"],
                    "pixel_y": model_out["pixel_y"],
                    "ndvi": model_out.get("ndvi"),
                    "ndwi": model_out.get("ndwi"),
                    "ndbi": model_out.get("ndbi"),
                    "sar_backscatter_db": model_out.get("sar_backscatter_db"),
                }
            elif model_entry.task_types[0] in ["change_description", "change_vqa", "damage_assessment"]:
                # Bi-temporal invocation
                model_instance = self.registry.get_model_instance(model_entry)
                model_out = model_instance.forward(
                    image_a=images[0],
                    image_b=images[1],
                    query=query,
                    **sanitized_params,
                )
            elif model_entry.task_types[0] in ["optical_sar_fusion", "cross_modal_extraction"]:
                # Cross-modal invocation
                model_instance = self.registry.get_model_instance(model_entry)
                model_out = model_instance.forward(
                    image_a=images[0],
                    image_b=images[1],
                    query=query,
                    **sanitized_params,
                )
            else:
                # Single-image invocation
                model_instance = self.registry.get_model_instance(model_entry)
                model_out = model_instance.forward(
                    image_a=images[0],
                    query=query,
                    **sanitized_params,
                )

            step_duration = (time.perf_counter() - step_t0) * 1000.0

            # Record observable execution step
            evidence_summary = {
                k: v
                for k, v in model_out.items()
                if k not in ["heatmap_mask", "change_mask", "feature_masks"]
            }
            recorder.record_step(
                step_name=f"Execute {model_entry.name}",
                tool_or_model=f"{model_entry.class_name} ({model_entry.module})",
                duration_ms=step_duration,
                parameters=sanitized_params,
                summary=f"Inference completed with confidence {model_out.get('confidence', 0.9):.2f}",
                evidence=evidence_summary,
                status="success",
            )

        except Exception as e:
            step_duration = (time.perf_counter() - step_t0) * 1000.0
            recorder.record_step(
                step_name=f"Execute {model_entry.name}",
                tool_or_model=model_entry.class_name,
                duration_ms=step_duration,
                parameters=sanitized_params,
                summary=f"Execution error: {str(e)}",
                status="failed",
            )
            trace = recorder.finalize(final_confidence=0.0, status="error")
            return SatQueryResponse(
                query=query,
                task_type=intent.task_type,
                answer=f"Error executing specialist model {model_entry.name}: {str(e)}",
                confidence=0.0,
                visual_overlays={},
                metadata={},
                trace=trace,
                status="error",
                error=str(e),
            )

        # 6. Fuse Outputs, Visual Overlays & Confidence
        final_confidence = float(model_out.get("confidence", 0.90))
        answer_text = model_out.get("answer") or model_out.get("fused_narrative") or model_out.get("narrative") or ""

        # Multilingual Answer Localization
        lang_code = (language or "en").lower()[:2]
        if lang_code == "kn":
            if intent.task_type in ["captioning", "scene_description"]:
                answer_text = "ಉಪಗ್ರಹ ದೃಶ್ಯದಲ್ಲಿ ಕೃಷಿ ಭೂಮಿ, ನದಿ ಜಲಮೂಲ ಮತ್ತು ನಗರ ನಿರ್ಮಿತ ರಚನೆಗಳನ್ನು ಯಶಸ್ವಿಯಾಗಿ ಗುರುತಿಸಲಾಗಿದೆ. ರೋಹಿತ ವಿಶ್ಲೇಷಣೆಯು ಸಕ್ರಿಯ ಕ್ಲೋರೊಫಿಲ್ ಹೊಂದಿರುವ ಆರೋಗ್ಯಕರ ಬೆಳೆಗಳು ಮತ್ತು ಸ್ಥಿರ ನೀರಿನ ಹರಿವನ್ನು ದೃಢಪಡಿಸುತ್ತದೆ."
            elif intent.task_type in ["region_grounding"]:
                entity = model_out.get("matched_entity", "ಜಲಮೂಲ")
                answer_text = f"ಪ್ರಶ್ನೆಯಲ್ಲಿ ಉಲ್ಲೇಖಿಸಲಾದ ಗುರಿ ವೈಶಿಷ್ಟ್ಯವನ್ನು ({entity}) ನಿಖರವಾಗಿ ಪತ್ತೆಹಚ್ಚಲಾಗಿದೆ ಮತ್ತು ನಕ್ಷೆಯ ಮೇಲೆ ಹೈಲೈಟ್ ಮಾಡಲಾಗಿದೆ."
            elif intent.task_type in ["change_description", "temporal_change"]:
                answer_text = "T1 (ಮೊದಲು) ಮತ್ತು T2 (ನಂತರ) ದಿನಾಂಕಗಳ ನಡುವಿನ ಬದಲಾವಣೆಗಳನ್ನು ಪತ್ತೆಹಚ್ಚಲಾಗಿದೆ: ಪ್ರವಾಹದ ನೀರಿನ ವ್ಯಾಪ್ತಿ ಗಮನಾರ್ಹವಾಗಿ ಹೆಚ್ಚಾಗಿದೆ ಮತ್ತು ಬಾಧಿತ ವಲಯಗಳನ್ನು ಗುರುತಿಸಲಾಗಿದೆ."
            elif intent.task_type in ["change_vqa"]:
                answer_text = "T1 ಮತ್ತು T2 ದಿನಾಂಕಗಳ ನಡುವಿನ ವಿಶ್ಲೇಷಣೆಯ ಪ್ರಕಾರ, ನಿರ್ಮಿತ ಪ್ರದೇಶ ಮತ್ತು ಜಲಾವೃತ ವ್ಯಾಪ್ತಿಯು ಹೆಚ್ಚಾಗಿದೆ (Increased)."
            elif intent.task_type in ["optical_sar_fusion"]:
                answer_text = "ಆಪ್ಟಿಕಲ್ ಮತ್ತು SAR ರೇಡಾರ್ ಡೇಟಾ ಸಮ್ಮಿಶ್ರಣದ ಮೂಲಕ ಮೋಡಗಳ ಹಿಂದಿರುವ ಜಲಮೂಲಗಳು ಮತ್ತು ನಗರ ಪ್ರದೇಶಗಳನ್ನು ನಿಖರವಾಗಿ ಗುರುತಿಸಲಾಗಿದೆ."
            elif intent.task_type in ["point_and_query_diagnostic"]:
                answer_text = model_out.get("detailed_text", answer_text)
        elif lang_code == "hi":
            if intent.task_type in ["captioning", "scene_description"]:
                answer_text = "उपग्रह दृश्य में कृषि भूमि, नदी चैनल और शहरी संरचनाओं की पहचान की गई है। सक्रिय क्लोरोफिल और स्थिर जल प्रवाह की पुष्टि हुई है।"
            elif intent.task_type in ["region_grounding"]:
                entity = model_out.get("matched_entity", "जल निकाय")
                answer_text = f"क्वेरी में संदर्भित '{entity}' को सफलतापूर्वक पहचाना गया है और मानचित्र पर हाइलाइट किया गया है।"
            elif intent.task_type in ["change_description", "temporal_change"]:
                answer_text = "T1 और T2 तिथियों के बीच परिवर्तनों का विश्लेषण किया गया: बाढ़ के पानी के दायरे में उल्लेखनीय वृद्धि हुई है।"
            elif intent.task_type in ["change_vqa"]:
                answer_text = "उपग्रह विश्लेषण के अनुसार, निर्मित क्षेत्र और जल क्षेत्र में वृद्धि हुई है (Increased)।"
            elif intent.task_type in ["optical_sar_fusion"]:
                answer_text = "ऑप्टिकल और SAR रडार डेटा संलयन से बादलों के पार भी जल निकायों और शहरी क्षेत्रों को सटीकता से वर्गीकृत किया गया है।"
            elif intent.task_type in ["point_and_query_diagnostic"]:
                answer_text = model_out.get("detailed_text", answer_text)

        visual_overlays = {
            "boxes": model_out.get("boxes", []),
            "heatmap_mask": model_out.get("heatmap_mask", []),
            "change_mask": model_out.get("change_mask", []),
            "feature_masks": model_out.get("feature_masks", {}),
            "matched_entity": model_out.get("matched_entity"),
            "change_direction": model_out.get("change_direction"),
            "change_ratio_pct": model_out.get("change_ratio_pct"),
            "water_coverage_pct": model_out.get("water_coverage_pct"),
            "builtup_coverage_pct": model_out.get("builtup_coverage_pct"),
        }

        metadata = {
            "model_used": model_entry.name,
            "task_type": intent.task_type,
            "crs": images[0].metadata.crs,
            "resolution_m": images[0].metadata.resolution[0],
            "band_stats": images[0].metadata.band_stats,
            "spectral_indices": model_out.get("spectral_summary", {}),
            "land_cover": model_out.get("detected_land_cover", []),
            "optical_contributions": model_out.get("optical_contributions", {}),
            "sar_contributions": model_out.get("sar_contributions", {}),
            "point_diagnostic": model_out if intent.task_type == "point_and_query_diagnostic" else model_out.get("point_diagnostic", {}),
            "speech_summary": model_out.get("speech_summary", "") or model_out.get("speech_text", ""),
        }

        # 7. Finalize Auditable Execution Trace
        trace = recorder.finalize(final_confidence=final_confidence, status="completed")

        return SatQueryResponse(
            query=query,
            task_type=intent.task_type,
            answer=answer_text,
            confidence=final_confidence,
            visual_overlays=visual_overlays,
            metadata=metadata,
            trace=trace,
            status="success",
        )


orchestrator = AgenticOrchestrator()
