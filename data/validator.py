from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from data.geotiff_loader import RSImage


class ValidationResult(BaseModel):
    valid: bool
    input_configuration: str  # "single", "cross_modal_pair", "bi_temporal_pair", "invalid"
    image_count: int
    modalities_declared: List[str] = Field(default_factory=list)
    crs_declared: List[str] = Field(default_factory=list)
    spatial_resolutions: List[Dict[str, float]] = Field(default_factory=list)
    co_registration_overlap_pct: Optional[float] = None
    dimension_match: bool = True
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class InputValidator:
    """Rigorous input-validation layer for remote-sensing imagery."""

    @staticmethod
    def calculate_bounds_overlap_pct(
        b1: Tuple[float, float, float, float],
        b2: Tuple[float, float, float, float],
    ) -> float:
        """
        Calculates Intersection-over-Union (IoU) percentage between two bounding boxes.
        Bounds format: (min_x, min_y, max_x, max_y)
        """
        ix_min = max(b1[0], b2[0])
        iy_min = max(b1[1], b2[1])
        ix_max = min(b1[2], b2[2])
        iy_max = min(b1[3], b2[3])

        if ix_max <= ix_min or iy_max <= iy_min:
            return 0.0

        intersection_area = (ix_max - ix_min) * (iy_max - iy_min)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union_area = area1 + area2 - intersection_area

        if union_area <= 0:
            return 0.0

        return round(float(intersection_area / union_area) * 100.0, 2)

    @classmethod
    def validate(
        cls,
        images: List[RSImage],
        declared_task: Optional[str] = None,
        expected_relationship: Optional[str] = None,  # "single", "cross_modal", "bi_temporal"
    ) -> ValidationResult:
        """
        Validates 1 or 2 images against format, modality, CRS, and co-registration rules.
        """
        count = len(images)
        modalities = [img.modality for img in images]
        crss = [img.metadata.crs for img in images]
        resolutions = [
            {"x": img.metadata.resolution[0], "y": img.metadata.resolution[1]}
            for img in images
        ]
        errors: List[str] = []
        warnings: List[str] = []

        if count == 0:
            return ValidationResult(
                valid=False,
                input_configuration="invalid",
                image_count=0,
                errors=["No images provided. Remote-sensing analysis requires at least 1 image."],
            )

        if count > 2:
            return ValidationResult(
                valid=False,
                input_configuration="invalid",
                image_count=count,
                errors=[f"Unsupported image count ({count}). SatQuery accepts 1 or 2 images."],
            )

        # 1. Single Image Validation
        if count == 1:
            config = "single"
            img = images[0]

            # Format verification
            if img.metadata.format_name.lower() not in ["geotiff", "tiff", "png", "jpeg", "jpg"]:
                errors.append(
                    f"Unsupported format '{img.metadata.format_name}'. Must be GeoTIFF/TIFF (or PNG/JPEG for benchmarks)."
                )

            # Modality check
            if img.modality not in ["optical", "multispectral", "sar"]:
                errors.append(f"Invalid modality tag '{img.modality}'. Must be 'optical', 'multispectral', or 'sar'.")

            # Check task compatibility
            if expected_relationship in ["cross_modal", "bi_temporal"]:
                errors.append(
                    f"Declared relationship '{expected_relationship}' requires 2 images, but only 1 was provided."
                )
            if declared_task in ["change_description", "change_vqa", "optical_sar_fusion"]:
                errors.append(
                    f"Task '{declared_task}' requires 2 co-registered images, but only 1 was supplied."
                )

            return ValidationResult(
                valid=len(errors) == 0,
                input_configuration=config,
                image_count=1,
                modalities_declared=modalities,
                crs_declared=crss,
                spatial_resolutions=resolutions,
                errors=errors,
                warnings=warnings,
            )

        # 2. Pair Image Validation (Count == 2)
        img1, img2 = images[0], images[1]
        dim_match = (img1.height == img2.height) and (img1.width == img2.width)

        if not dim_match:
            warnings.append(
                f"Image dimension mismatch: Image A ({img1.width}x{img1.height}) vs Image B ({img2.width}x{img2.height}). "
                "Resampling/alignment will be performed for model inference."
            )

        # Co-registration check
        overlap_pct = cls.calculate_bounds_overlap_pct(img1.metadata.bounds, img2.metadata.bounds)
        if overlap_pct < 50.0:
            errors.append(
                f"Co-registration failure: Spatial overlap between images is {overlap_pct}%, which is below the 50% threshold. "
                "Ensure both images depict the same geographic region."
            )
        elif overlap_pct < 85.0:
            warnings.append(
                f"Sub-optimal spatial overlap ({overlap_pct}%). Cropping to intersection bounding box recommended."
            )

        # Modality relationship verification
        has_optical = any(m in ["optical", "multispectral"] for m in modalities)
        has_sar = any(m == "sar" for m in modalities)

        is_cross_modal = has_optical and has_sar
        is_same_sensor_type = (modalities[0] == modalities[1]) or (
            has_optical and not has_sar
        )

        # Infer or verify input configuration
        if expected_relationship == "cross_modal" or declared_task == "optical_sar_fusion":
            config = "cross_modal_pair"
            if not is_cross_modal:
                errors.append(
                    f"Cross-modal analysis requires 1 Optical/Multispectral + 1 SAR image. Received: {modalities}."
                )
        elif expected_relationship == "bi_temporal" or declared_task in ["change_description", "change_vqa"]:
            config = "bi_temporal_pair"
            if is_cross_modal:
                warnings.append(
                    "Bi-temporal change detection requested on cross-modal (Optical + SAR) pair. "
                    "Cross-modal change detection is supported, but sensor-invariant feature alignment will be applied."
                )
        else:
            config = "cross_modal_pair" if is_cross_modal else "bi_temporal_pair"

        return ValidationResult(
            valid=len(errors) == 0,
            input_configuration=config,
            image_count=2,
            modalities_declared=modalities,
            crs_declared=crss,
            spatial_resolutions=resolutions,
            co_registration_overlap_pct=overlap_pct,
            dimension_match=dim_match,
            errors=errors,
            warnings=warnings,
        )
