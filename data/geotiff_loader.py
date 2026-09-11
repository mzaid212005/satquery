import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import tifffile


class RSImageMetadata:
    def __init__(
        self,
        format_name: str,
        width: int,
        height: int,
        channels: int,
        dtype: str,
        crs: Optional[str] = None,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        resolution: Optional[Tuple[float, float]] = None,
        nodata: Optional[float] = None,
        band_stats: Optional[List[Dict[str, float]]] = None,
    ):
        self.format_name = format_name
        self.width = width
        self.height = height
        self.channels = channels
        self.dtype = dtype
        self.crs = crs or "EPSG:4326"  # Default geographic or georeferenced
        self.bounds = bounds or (0.0, 0.0, float(width), float(height))
        self.resolution = resolution or (10.0, 10.0)  # Default 10m (Sentinel standard)
        self.nodata = nodata
        self.band_stats = band_stats or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": self.format_name,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "dtype": self.dtype,
            "crs": self.crs,
            "bounds": {
                "min_x": self.bounds[0],
                "min_y": self.bounds[1],
                "max_x": self.bounds[2],
                "max_y": self.bounds[3],
            },
            "resolution_meters": {
                "x": self.resolution[0],
                "y": self.resolution[1],
            },
            "band_stats": self.band_stats,
        }


class RSImage:
    """Standardized Remote Sensing Image Representation."""

    def __init__(
        self,
        data: np.ndarray,  # (H, W, C) float32 normalized [0, 1]
        metadata: RSImageMetadata,
        modality: str = "optical",  # "optical", "multispectral", "sar"
        filepath: Optional[Path] = None,
    ):
        self.data = data
        self.metadata = metadata
        self.modality = modality.lower()
        self.filepath = filepath

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def channels(self) -> int:
        return self.data.shape[2]

    def get_rgb_preview(self) -> np.ndarray:
        """Returns 3-channel uint8 array suitable for display."""
        if self.channels >= 3:
            # First 3 bands as RGB
            rgb = self.data[:, :, :3]
        elif self.channels == 2:
            # Dual-pol SAR (VV, VH) -> compose false-color (VV, VH, VV/VH)
            vv = self.data[:, :, 0]
            vh = self.data[:, :, 1]
            ratio = np.clip(vv / (vh + 1e-4), 0, 1)
            rgb = np.stack([vv, vh, ratio], axis=-1)
        else:
            # Single-channel grayscale to 3 channels
            single = self.data[:, :, 0]
            rgb = np.stack([single, single, single], axis=-1)

        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        return rgb

    def to_base64_png(self) -> str:
        """Converts RGB preview to base64 PNG data URL."""
        preview = self.get_rgb_preview()
        img = Image.fromarray(preview)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"


class GeoTIFFLoader:
    """Loads GeoTIFF, TIFF, and standard benchmark raster imagery."""

    @staticmethod
    def load(
        file_source: Union[str, Path, bytes],
        modality: str = "optical",
        filename: Optional[str] = None,
    ) -> RSImage:
        filename_str = str(filename or file_source)
        is_tiff = any(
            filename_str.lower().endswith(ext)
            for ext in [".tif", ".tiff", ".geotiff"]
        )

        if isinstance(file_source, bytes):
            bio = io.BytesIO(file_source)
            if is_tiff:
                raw_data, meta = GeoTIFFLoader._read_tiff_stream(bio)
            else:
                raw_data, meta = GeoTIFFLoader._read_pil_stream(bio, format_hint=filename_str)
        else:
            path = Path(file_source)
            if is_tiff or path.suffix.lower() in [".tif", ".tiff", ".geotiff"]:
                with open(path, "rb") as f:
                    raw_data, meta = GeoTIFFLoader._read_tiff_stream(f)
            else:
                raw_data, meta = GeoTIFFLoader._read_pil_file(path)

        # Standardize array shape to (H, W, C)
        if raw_data.ndim == 2:
            raw_data = raw_data[:, :, np.newaxis]
        elif raw_data.ndim == 3 and raw_data.shape[0] < raw_data.shape[2]:
            # Transpose (C, H, W) -> (H, W, C) if needed
            raw_data = np.transpose(raw_data, (1, 2, 0))

        # Normalize to float32 [0, 1]
        data_norm, stats = GeoTIFFLoader._normalize_bands(raw_data)
        meta.band_stats = stats
        meta.width = data_norm.shape[1]
        meta.height = data_norm.shape[0]
        meta.channels = data_norm.shape[2]

        return RSImage(
            data=data_norm,
            metadata=meta,
            modality=modality,
            filepath=Path(file_source) if isinstance(file_source, (str, Path)) else None,
        )

    @staticmethod
    def _read_tiff_stream(stream: io.BytesIO) -> Tuple[np.ndarray, RSImageMetadata]:
        with tifffile.TiffFile(stream) as tif:
            arr = tif.asarray()
            first_page = tif.pages[0]
            tags = first_page.tags

            # Extract basic geometry
            h, w = arr.shape[:2] if arr.ndim >= 2 else (1, 1)
            c = arr.shape[2] if arr.ndim == 3 else 1

            # Extract georeferencing if present in tags
            crs = "EPSG:32643"  # Standard UTM zone for RS evaluation
            res_x, res_y = 10.0, 10.0
            min_x, min_y, max_x, max_y = 500000.0, 4200000.0, 500000.0 + w * res_x, 4200000.0 + h * res_y

            # Check if custom metadata dict is present in description
            desc = tags.get("ImageDescription")
            if desc and desc.value:
                try:
                    desc_meta = json.loads(desc.value)
                    crs = desc_meta.get("crs", crs)
                    if "bounds" in desc_meta:
                        b = desc_meta["bounds"]
                        min_x, min_y, max_x, max_y = b.get("min_x", min_x), b.get("min_y", min_y), b.get("max_x", max_x), b.get("max_y", max_y)
                    if "resolution" in desc_meta:
                        r = desc_meta["resolution"]
                        res_x, res_y = r.get("x", res_x), r.get("y", res_y)
                except Exception:
                    pass

            meta = RSImageMetadata(
                format_name="GeoTIFF",
                width=w,
                height=h,
                channels=c,
                dtype=str(arr.dtype),
                crs=crs,
                bounds=(min_x, min_y, max_x, max_y),
                resolution=(res_x, res_y),
            )
            return arr, meta

    @staticmethod
    def _read_pil_file(path: Path) -> Tuple[np.ndarray, RSImageMetadata]:
        with Image.open(path) as img:
            arr = np.array(img)
            fmt = img.format or "Raster"
            w, h = img.size
            c = arr.shape[2] if arr.ndim == 3 else 1
            meta = RSImageMetadata(
                format_name=fmt,
                width=w,
                height=h,
                channels=c,
                dtype=str(arr.dtype),
                crs="EPSG:4326",
                bounds=(0.0, 0.0, float(w), float(h)),
                resolution=(1.0, 1.0),
            )
            return arr, meta

    @staticmethod
    def _read_pil_stream(stream: io.BytesIO, format_hint: str) -> Tuple[np.ndarray, RSImageMetadata]:
        with Image.open(stream) as img:
            arr = np.array(img)
            fmt = img.format or Path(format_hint).suffix.replace(".", "").upper() or "Raster"
            w, h = img.size
            c = arr.shape[2] if arr.ndim == 3 else 1
            meta = RSImageMetadata(
                format_name=fmt,
                width=w,
                height=h,
                channels=c,
                dtype=str(arr.dtype),
                crs="EPSG:4326",
                bounds=(0.0, 0.0, float(w), float(h)),
                resolution=(1.0, 1.0),
            )
            return arr, meta

    @staticmethod
    def _normalize_bands(data: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, float]]]:
        """Normalizes imagery to [0, 1] float32 while preserving inter-band radiometric ratios."""
        data_float = data.astype(np.float32)
        h, w, c = data_float.shape
        stats = []

        global_max = float(np.max(data_float))
        global_min = float(np.min(data_float))

        for band_idx in range(c):
            band = data_float[:, :, band_idx]
            b_min, b_max = float(np.min(band)), float(np.max(band))
            b_mean, b_std = float(np.mean(band)), float(np.std(band))

            stats.append({
                "band": band_idx + 1,
                "min": round(b_min, 3),
                "max": round(b_max, 3),
                "mean": round(b_mean, 3),
                "std": round(b_std, 3),
            })

        if global_max <= 1.01:
            normalized = np.clip(data_float, 0.0, 1.0)
        elif global_max <= 255.0:
            normalized = data_float / 255.0
        elif global_max <= 4096.0:  # 12-bit Sentinel
            normalized = np.clip(data_float / 4095.0, 0.0, 1.0)
        elif global_max <= 10000.0:  # Sentinel-2 L2A BOA reflectance (scale 10000)
            normalized = np.clip(data_float / 10000.0, 0.0, 1.0)
        else:
            normalized = (data_float - global_min) / (global_max - global_min + 1e-5)

        return normalized.astype(np.float32), stats
