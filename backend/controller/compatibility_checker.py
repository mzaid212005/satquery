from typing import List, Optional
from pydantic import BaseModel, Field
from backend.controller.query_interpreter import QueryIntent
from data.geotiff_loader import RSImage
from data.validator import InputValidator, ValidationResult


class CompatibilityReport(BaseModel):
    compatible: bool
    task_type: str
    rejection_reason: Optional[str] = None
    validation_result: ValidationResult
    suggested_action: Optional[str] = None


class CompatibilityChecker:
    """Verifies that uploaded remote sensing inputs match what the inferred task requires."""

    @classmethod
    def check(
        cls,
        intent: QueryIntent,
        images: List[RSImage],
    ) -> CompatibilityReport:
        # First run base input validation
        validation = InputValidator.validate(
            images=images,
            declared_task=intent.task_type,
            expected_relationship=intent.required_relationship,
        )

        if not validation.valid:
            return CompatibilityReport(
                compatible=False,
                task_type=intent.task_type,
                rejection_reason="; ".join(validation.errors),
                validation_result=validation,
                suggested_action="Ensure images match format (GeoTIFF/TIFF), valid CRS, and co-registration overlap.",
            )

        # Task specific checks
        # 1. Change VQA / Change description
        if intent.task_type in ["change_vqa", "change_description"]:
            if len(images) < 2:
                return CompatibilityReport(
                    compatible=False,
                    task_type=intent.task_type,
                    rejection_reason=(
                        f"Task '{intent.task_type}' requires 2 bi-temporal co-registered images (T1 pre-event and T2 post-event). "
                        f"Received only {len(images)} image."
                    ),
                    validation_result=validation,
                    suggested_action="Please upload a second co-registered image of the same area taken at a different acquisition time.",
                )

        # 2. Optical-SAR Fusion
        if intent.task_type in ["optical_sar_fusion", "cross_modal_extraction"]:
            if len(images) < 2:
                return CompatibilityReport(
                    compatible=False,
                    task_type=intent.task_type,
                    rejection_reason=(
                        "Cross-modal fusion requires both an Optical/Multispectral image AND a SAR radar image. "
                        f"Received only {len(images)} image."
                    ),
                    validation_result=validation,
                    suggested_action="Upload two co-registered images: one Optical/Multispectral and one SAR.",
                )

            modalities = [img.modality.lower() for img in images]
            has_opt = any(m in ["optical", "multispectral"] for m in modalities)
            has_sar = any(m == "sar" for m in modalities)

            if not (has_opt and has_sar):
                return CompatibilityReport(
                    compatible=False,
                    task_type=intent.task_type,
                    rejection_reason=(
                        f"Cross-modal optical-SAR fusion rejected: inputs do not contain complementary modalities. "
                        f"Declared modalities are: {modalities}. Must have 1 Optical/Multispectral and 1 SAR."
                    ),
                    validation_result=validation,
                    suggested_action="Provide one optical or multispectral image and one SAR (e.g. Sentinel-1, RISAT) image.",
                )

        # 3. Single image tasks (VQA, captioning, grounding)
        if intent.task_type in ["rs_vqa", "captioning", "region_grounding"]:
            if len(images) > 1:
                # We can accept 2 images by either processing the first one with a warning, or noting it
                validation.warnings.append(
                    f"Task '{intent.task_type}' operates on a single image. Using Image 1 (primary) for analysis."
                )

        return CompatibilityReport(
            compatible=True,
            task_type=intent.task_type,
            rejection_reason=None,
            validation_result=validation,
        )
