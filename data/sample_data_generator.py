import json
from pathlib import Path
from typing import Tuple
import numpy as np
import tifffile
from backend.config import settings


class SampleDataGenerator:
    """Generates synthetic, georeferenced multi-sensor GeoTIFF imagery for testing and demonstration."""

    @staticmethod
    def generate_all(output_dir: Path = settings.sample_dir) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        SampleDataGenerator.create_single_optical(output_dir / "single_optical_scene.tif")
        SampleDataGenerator.create_bitemporal_pair(
            output_dir / "bitemporal_t1_pre_event.tif",
            output_dir / "bitemporal_t2_post_event.tif",
        )
        SampleDataGenerator.create_cross_modal_pair(
            output_dir / "cross_modal_optical.tif",
            output_dir / "cross_modal_sar.tif",
        )

    @staticmethod
    def create_single_optical(filepath: Path, size: Tuple[int, int] = (256, 256)) -> None:
        """Creates a 3-band optical scene with distinct water body, built-up area, and forest."""
        h, w = size
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # 1. Background vegetation (Greenish)
        img[:, :] = [34, 139, 34]  # Forest green

        # 2. Add an agricultural checkerboard pattern
        for y in range(0, h // 2, 32):
            for x in range(0, w // 2, 32):
                if (x + y) % 64 == 0:
                    img[y : y + 30, x : x + 30] = [124, 180, 50]  # Light crop green

        # 3. Add a winding water body (Deep blue) in the southern half
        y_coords, x_coords = np.ogrid[:h, :w]
        river_center = 160 + 20 * np.sin(x_coords / 30.0)
        river_mask = (y_coords > river_center - 22) & (y_coords < river_center + 22)
        img[river_mask] = [20, 60, 160]  # River blue

        # 4. Add an urban built-up cluster (Gray / red roofs) in north-east quadrant
        urban_mask = (y_coords < 120) & (x_coords > 140)
        img[urban_mask] = [170, 165, 160]  # Concrete gray
        # Add street grid
        for i in range(140, w, 20):
            img[:120, i : i + 3] = [80, 80, 80]
        for j in range(0, 120, 20):
            img[j : j + 3, 140:] = [80, 80, 80]

        # Metadata dictionary
        meta = {
            "title": "Single Optical Urban-Water Scene",
            "crs": "EPSG:32643",
            "resolution": {"x": 10.0, "y": 10.0},
            "bounds": {"min_x": 500000.0, "min_y": 4200000.0, "max_x": 502560.0, "max_y": 4202560.0},
            "modality": "optical",
            "sensor": "Sentinel-2 MSI",
        }

        tifffile.imwrite(
            filepath,
            img,
            photometric="rgb",
            description=json.dumps(meta),
        )

    @staticmethod
    def create_bitemporal_pair(t1_path: Path, t2_path: Path, size: Tuple[int, int] = (256, 256)) -> None:
        """
        Creates T1 (pre-event) and T2 (post-event) images.
        T1: River at normal gauge, compact urban region.
        T2: River flooded across lowlands (+35% water coverage) and expanded built-up zone.
        """
        h, w = size
        y_coords, x_coords = np.ogrid[:h, :w]

        # --- T1 Image (Pre-event) ---
        t1 = np.zeros((h, w, 3), dtype=np.uint8)
        t1[:, :] = [45, 130, 45]  # Vegetation

        # River T1
        river_center = 140 + 15 * np.sin(x_coords / 25.0)
        river_mask_t1 = (y_coords > river_center - 15) & (y_coords < river_center + 15)
        t1[river_mask_t1] = [25, 75, 170]

        # Built-up T1 (Compact in top-left)
        urban_t1 = (y_coords < 100) & (x_coords < 100)
        t1[urban_t1] = [160, 155, 150]

        # --- T2 Image (Post-event: Flood + Urban Expansion) ---
        t2 = np.zeros((h, w, 3), dtype=np.uint8)
        t2[:, :] = [45, 130, 45]

        # Flooded River T2 (Substantially wider + inundated fields)
        river_mask_t2 = (y_coords > river_center - 38) & (y_coords < river_center + 38)
        t2[river_mask_t2] = [20, 65, 150]

        # Built-up T2 (Expanded from (100, 100) to (140, 130))
        urban_t2 = (y_coords < 120) & (x_coords < 135)
        t2[urban_t2] = [175, 170, 165]

        bounds = {"min_x": 620000.0, "min_y": 3100000.0, "max_x": 622560.0, "max_y": 3102560.0}
        meta_t1 = {
            "title": "Bi-Temporal T1 (Pre-Event)",
            "acquisition_date": "2024-03-15",
            "crs": "EPSG:32643",
            "resolution": {"x": 10.0, "y": 10.0},
            "bounds": bounds,
            "modality": "optical",
        }
        meta_t2 = {
            "title": "Bi-Temporal T2 (Post-Event: Flooding & Growth)",
            "acquisition_date": "2024-08-20",
            "crs": "EPSG:32643",
            "resolution": {"x": 10.0, "y": 10.0},
            "bounds": bounds,
            "modality": "optical",
        }

        tifffile.imwrite(t1_path, t1, photometric="rgb", description=json.dumps(meta_t1))
        tifffile.imwrite(t2_path, t2, photometric="rgb", description=json.dumps(meta_t2))

    @staticmethod
    def create_cross_modal_pair(opt_path: Path, sar_path: Path, size: Tuple[int, int] = (256, 256)) -> None:
        """
        Creates co-registered Optical and SAR pair.
        Optical: Shows optical spectral reflectance with partial cloud obscuration.
        SAR: C-band radar backscatter penetrating clouds; water is very dark (specular),
             built-up is very bright (double bounce corner reflectors).
        """
        h, w = size
        y_coords, x_coords = np.ogrid[:h, :w]

        # Common ground features
        # Lake in center-right
        lake_mask = ((x_coords - 170) ** 2 / (55**2) + (y_coords - 140) ** 2 / (40**2)) <= 1.0
        # Industrial / built-up grid in bottom-left
        builtup_mask = (x_coords < 110) & (y_coords > 130)

        # 1. Optical Scene
        opt = np.zeros((h, w, 3), dtype=np.uint8)
        opt[:, :] = [50, 140, 55]  # Vegetation green
        opt[lake_mask] = [20, 80, 180]  # Lake water
        opt[builtup_mask] = [170, 170, 170]  # Buildings

        # Add thin semi-transparent cloud over the center-top
        cloud_mask = ((x_coords - 130) ** 2 / (70**2) + (y_coords - 80) ** 2 / (35**2)) <= 1.0
        opt[cloud_mask] = np.clip(opt[cloud_mask].astype(np.int16) + 90, 0, 255).astype(np.uint8)

        # 2. SAR Scene (Dual polarization / calibrated backscatter in dB -> scaled to uint8)
        # Specular water = very low backscatter (dark ~10-25)
        # Rough vegetation = medium volume scattering (~80-110)
        # Built-up = strong double-bounce backscatter (~210-255)
        # Clouds = totally transparent to microwave C-band
        sar = np.random.normal(95, 12, (h, w)).clip(60, 130).astype(np.uint8)  # Speckled terrain
        sar[lake_mask] = np.random.normal(15, 5, np.sum(lake_mask)).clip(5, 30).astype(np.uint8)
        sar[builtup_mask] = np.random.normal(225, 15, np.sum(builtup_mask)).clip(180, 255).astype(np.uint8)

        # Repeat to 3 channels for visualization/uniformity or write as 2-band / 3-band
        sar_3ch = np.stack([sar, sar, sar], axis=-1)

        bounds = {"min_x": 710000.0, "min_y": 2850000.0, "max_x": 712560.0, "max_y": 2852560.0}
        meta_opt = {
            "title": "Cross-Modal Optical (Sentinel-2)",
            "crs": "EPSG:32643",
            "resolution": {"x": 10.0, "y": 10.0},
            "bounds": bounds,
            "modality": "optical",
        }
        meta_sar = {
            "title": "Cross-Modal SAR (Sentinel-1 C-Band)",
            "crs": "EPSG:32643",
            "resolution": {"x": 10.0, "y": 10.0},
            "bounds": bounds,
            "modality": "sar",
        }

        tifffile.imwrite(opt_path, opt, photometric="rgb", description=json.dumps(meta_opt))
        tifffile.imwrite(sar_path, sar_3ch, photometric="rgb", description=json.dumps(meta_sar))


if __name__ == "__main__":
    SampleDataGenerator.generate_all()
    print("Sample datasets generated successfully!")
