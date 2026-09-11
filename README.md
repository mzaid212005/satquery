# SatQuery AI 🛰️

**SatQuery AI** is an agentic, query-driven vision-language assistant for remote-sensing imagery. Unlike generic VLMs, SatQuery AI interprets natural-language queries, validates multi-modal geospatial inputs, routes requests to specialist remote-sensing models via a declarative tool registry, and produces evidence-grounded answers with visual overlays, confidence estimation, and an auditable execution trace.

---

## 🌟 Core Features

- **Remote-Sensing Domain Adaptation**: Adapted on multisensor satellite data (BigEarthNet: optical multispectral + SAR).
- **Multi-Input Validation Layer**: Rigorous checking for:
  - Single-image (Optical / Multispectral / SAR) in GeoTIFF/TIFF and PNG/JPEG.
  - Cross-modal pairs (Co-registered Optical + SAR).
  - Bi-temporal pairs (Pre & Post acquisition co-registered imagery).
  - Spatial resolution, CRS/projection, bounds overlap, and grid alignment.
- **Config-Driven Specialist Tool Registry (`backend/registry.yaml`)**:
  - `rs_vqa_captioning`: Remote sensing visual question answering and scene description.
  - `region_grounding`: Text-guided region detection (bounding boxes & spatial heatmaps).
  - `change_analysis`: Bi-temporal change description, change-VQA, and change masks.
  - `optical_sar_fusion`: Joint complementary feature extraction across optical and SAR.
- **Agentic Controller**:
  - Query interpreter with intent classification & entity extraction.
  - Input-task compatibility verification.
  - Parameter whitelisting and dynamic tool invocation.
  - Chained workflows (e.g. Grounding → VQA).
  - Output fusion & confidence score calibration.
  - Auditable execution trace (no internal chain-of-thought leaks).
- **Interactive Web GUI & REST API**:
  - Dual canvas viewer with bounding box overlays, change difference heatmaps, and optical/SAR swipe/blend.
  - Query quick-picks for representative prompts.
  - 1-click sample dataset loader.
  - Downloadable JSON and PDF audit reports.
- **Comprehensive Benchmark Evaluation**:
  - RSVQA (VQA)
  - VRSBench (Captioning & Grounding)
  - CDVQA (Bi-temporal change VQA)
  - Generalization test on unseen ISRO/SAC sensor pair (Cartosat-2S + RISAT SAR).

---

## 🚀 Quick Start

### 1. Requirements
Ensure Python 3.10+ is installed.

```bash
pip install -r requirements.txt
```

### 2. Start the Backend API
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### 3. Launch Frontend Web GUI
Simply open `frontend/index.html` in your web browser, or serve it via:
```bash
python -m http.server 3000 --directory frontend
```

### 4. Run Automated Test Suite
```bash
pytest -v tests/
```
