import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from backend.config import settings
from backend.controller.custom_chatbot import custom_chatbot
from backend.controller.orchestrator import orchestrator
from backend.controller.registry_loader import registry_loader
from backend.controller.spatial_nlp import spatial_nlp, SpatialNLPEngine
from backend.api.auth_routes import get_current_user
from data.geotiff_loader import GeoTIFFLoader, RSImage
from data.sample_data_generator import SampleDataGenerator
from data.validator import InputValidator

router = APIRouter(prefix="/api")

SAMPLE_CATALOG = [
    {
        "id": "single_optical",
        "title": "Single Optical Scene (Sentinel-2 Urban & Agricultural Water Basin)",
        "description": "3-band optical imagery (256x256) showing river, agricultural parcels, and urban development.",
        "task_type": "captioning",
        "images": [
            {"filename": "single_optical_scene.tif", "modality": "optical", "label": "Image A (Optical RGB)"}
        ],
        "default_query": "Describe the land-cover and major objects visible in this image.",
        "quick_queries": [
            "Describe the land-cover and major objects visible in this image.",
            "Highlight the water body referred to in the query.",
            "Suggest crop, vegetable, soil level etc object highlight that part",
        ],
    },
    {
        "id": "bitemporal_flood",
        "title": "Bi-Temporal Flood & Urban Expansion (T1 Pre-Event vs T2 Post-Event)",
        "description": "Co-registered optical pair demonstrating flood inundation (+35% water) and urban sprawl.",
        "task_type": "change_description",
        "images": [
            {"filename": "bitemporal_t1_pre_event.tif", "modality": "optical", "label": "Image A (T1 Pre-Event)"},
            {"filename": "bitemporal_t2_post_event.tif", "modality": "optical", "label": "Image B (T2 Post-Event)"},
        ],
        "default_query": "What changed between these two dates, and where did the change occur?",
        "quick_queries": [
            "What changed between these two dates, and where did the change occur?",
            "Has the built-up area increased, decreased, or remained unchanged?",
            "Identify the flooded zones and damage extent.",
        ],
    },
    {
        "id": "cross_modal_fusion",
        "title": "Cross-Modal Optical + SAR (Sentinel-2 Cloud Penetration & Sentinel-1)",
        "description": "Joint optical reflectance with semi-transparent clouds and C-band SAR double-bounce backscatter.",
        "task_type": "optical_sar_fusion",
        "images": [
            {"filename": "cross_modal_optical.tif", "modality": "optical", "label": "Image A (Optical RGB)"},
            {"filename": "cross_modal_sar.tif", "modality": "sar", "label": "Image B (SAR Radar)"},
        ],
        "default_query": "Use the optical and SAR images together to identify built-up and water-covered regions.",
        "quick_queries": [
            "Use the optical and SAR images together to identify built-up and water-covered regions.",
            "Compare the radar backscatter with the optical chlorophyll reflectance.",
        ],
    },
    {
        "id": "heldout_isro",
        "title": "ISRO Cartosat-2S High-Res Optical + RISAT SAR",
        "description": "Held-out generalization test on Indian Space Research Organisation sensor suite.",
        "task_type": "optical_sar_fusion",
        "images": [
            {"filename": "heldout_cartosat2s_optical.tif", "modality": "optical", "label": "Image A (Cartosat-2S)"},
            {"filename": "heldout_risat_sar.tif", "modality": "sar", "label": "Image B (RISAT SAR)"},
        ],
        "default_query": "Use the optical and SAR images together to identify built-up and water-covered regions.",
        "quick_queries": [
            "Use the optical and SAR images together to identify built-up and water-covered regions.",
            "Evaluate urban boundaries and water features across ISRO sensors.",
        ],
    },
]


def resolve_sample_file(filename: str) -> Path:
    p = settings.sample_dir / filename
    if not p.exists():
        p = settings.sample_dir / "heldout_isro" / filename
    if not p.exists():
        # Ensure synthetic sample files exist
        SampleDataGenerator.generate_all(settings.sample_dir)
        p = settings.sample_dir / filename
        if not p.exists():
            p = settings.sample_dir / "heldout_isro" / filename
    return p


@router.get("/health")
def health_check():
    return {"status": "ok", "service": "SatQuery AI", "version": settings.version}


@router.get("/registry")
def get_registry():
    return registry_loader.registry.model_dump()


@router.get("/samples")
def get_sample_datasets():
    """Returns the catalog of ready-to-test multi-sensor sample datasets."""
    return SAMPLE_CATALOG


@router.get("/samples/{sample_id}/load")
def load_sample_dataset(sample_id: str):
    """Loads and returns base64 image previews and metadata for a specific sample scenario."""
    sample = next((s for s in SAMPLE_CATALOG if s["id"] == sample_id), None)
    if not sample:
        raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found.")

    loaded_images = []
    for s_img in sample["images"]:
        fpath = resolve_sample_file(s_img["filename"])
        rs_img = GeoTIFFLoader.load(fpath, modality=s_img["modality"])
        loaded_images.append({
            "filename": s_img["filename"],
            "modality": s_img["modality"],
            "label": s_img.get("label", s_img["filename"]),
            "preview_base64": rs_img.to_base64_png(),
            "metadata": rs_img.metadata.to_dict(),
        })

    return {
        "id": sample["id"],
        "title": sample["title"],
        "description": sample["description"],
        "task_type": sample["task_type"],
        "default_query": sample["default_query"],
        "quick_queries": sample.get("quick_queries", []),
        "images": loaded_images,
    }


@router.post("/validate")
async def validate_imagery(
    image_a: Optional[UploadFile] = File(None),
    modality_a: str = Form("optical"),
    image_b: Optional[UploadFile] = File(None),
    modality_b: Optional[str] = Form(None),
    sample_id: Optional[str] = Form(None),
    declared_task: Optional[str] = Form(None),
):
    images: List[RSImage] = []

    if sample_id:
        sample = next((s for s in SAMPLE_CATALOG if s["id"] == sample_id), None)
        if sample:
            for s_img in sample["images"]:
                fpath = resolve_sample_file(s_img["filename"])
                images.append(GeoTIFFLoader.load(fpath, modality=s_img["modality"]))
    else:
        if image_a:
            bytes_a = await image_a.read()
            images.append(GeoTIFFLoader.load(bytes_a, modality=modality_a, filename=image_a.filename))
        if image_b:
            bytes_b = await image_b.read()
            images.append(GeoTIFFLoader.load(bytes_b, modality=modality_b or "sar", filename=image_b.filename))

    res = InputValidator.validate(images, declared_task=declared_task)
    return res.model_dump()


@router.post("/query")
async def execute_query(
    query: str = Form(...),
    sample_id: Optional[str] = Form(None),
    image_a: Optional[UploadFile] = File(None),
    modality_a: str = Form("optical"),
    image_b: Optional[UploadFile] = File(None),
    modality_b: Optional[str] = Form(None),
    parameters_json: Optional[str] = Form(None),
    language: Optional[str] = Form("en-US"),
):
    images: List[RSImage] = []

    # 1. Load imagery from uploaded files or selected sample dataset (optional)
    if sample_id:
        sample = next((s for s in SAMPLE_CATALOG if s["id"] == sample_id), None)
        if not sample:
            raise HTTPException(status_code=404, detail=f"Sample dataset '{sample_id}' not found.")
        for s_img in sample["images"]:
            fpath = resolve_sample_file(s_img["filename"])
            images.append(GeoTIFFLoader.load(fpath, modality=s_img["modality"]))
    else:
        if image_a:
            bytes_a = await image_a.read()
            images.append(GeoTIFFLoader.load(bytes_a, modality=modality_a, filename=image_a.filename))
        if image_b:
            bytes_b = await image_b.read()
            images.append(GeoTIFFLoader.load(bytes_b, modality=modality_b or "sar", filename=image_b.filename))

    # Auto-load default GIS scene if no image provided
    if len(images) == 0:
        sample_path = resolve_sample_file("single_optical_scene.tif")
        if sample_path.exists():
            images.append(GeoTIFFLoader.load(sample_path, modality="optical"))

    # Parse parameters
    user_params = {}
    if parameters_json:
        try:
            user_params = json.loads(parameters_json)
        except Exception:
            pass

    # 2. Execute agentic orchestration pipeline
    response = orchestrator.execute(
        query=query,
        images=images,
        user_params=user_params,
        language=language or "en-US",
    )

    resp_dict = response.to_dict()

    # Include image previews in response for frontend rendering
    if len(images) > 0:
        resp_dict["input_previews"] = [img.to_base64_png() for img in images]
    return resp_dict


@router.post("/chat")
async def chat_endpoint(request: Request):
    """
    Connects any query to the Custom Satellite Chatbot.
    Supports both JSON and multipart/form-data.
    Executes agricultural/soil advisories, object grounding highlights,
    spectral indices, and conversational remote sensing Q&A in any selected language.
    """
    content_type = request.headers.get("content-type", "")
    images: List[RSImage] = []
    user_params = {}
    query = ""
    session_id = None
    sample_id = None
    language = "en-US"

    if "application/json" in content_type:
        body = await request.json()
        query = body.get("query") or body.get("message") or ""
        session_id = body.get("session_id")
        sample_id = body.get("sample_id")
        language = body.get("language") or "en-US"
        user_params = body.get("parameters") or {}
    else:
        form = await request.form()
        query = form.get("query") or form.get("message") or ""
        session_id = form.get("session_id")
        sample_id = form.get("sample_id")
        language = form.get("language") or "en-US"
        params_str = form.get("parameters_json")
        if params_str:
            try:
                user_params = json.loads(params_str)
            except Exception:
                pass

        image_a = form.get("image_a")
        image_b = form.get("image_b")
        modality_a = form.get("modality_a") or "optical"
        modality_b = form.get("modality_b") or "sar"

        if image_a and hasattr(image_a, "read"):
            bytes_a = await image_a.read()
            images.append(GeoTIFFLoader.load(bytes_a, modality=modality_a, filename=getattr(image_a, "filename", "image_a.tif")))
        if image_b and hasattr(image_b, "read"):
            bytes_b = await image_b.read()
            images.append(GeoTIFFLoader.load(bytes_b, modality=modality_b, filename=getattr(image_b, "filename", "image_b.tif")))

    if sample_id and len(images) == 0:
        sample = next((s for s in SAMPLE_CATALOG if s["id"] == sample_id), None)
        if sample:
            for s_img in sample["images"]:
                fpath = resolve_sample_file(s_img["filename"])
                images.append(GeoTIFFLoader.load(fpath, modality=s_img["modality"]))

    result = custom_chatbot.respond(
        query=query,
        session_id=session_id,
        images=images if len(images) > 0 else None,
        parameters=user_params,
        language=language,
    )

    if len(images) > 0:
        result["input_previews"] = [img.to_base64_png() for img in images]

    return result


@router.post("/point_query")
async def analyze_point_query(
    request: Request,
    image_a: Optional[UploadFile] = File(None),
    norm_x: Optional[float] = Form(None),
    norm_y: Optional[float] = Form(None),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
    sample_id: Optional[str] = Form(None),
    custom_question: Optional[str] = Form(None),
    language: Optional[str] = Form("en-US"),
):
    """Executes localized point-and-query analysis at a specific coordinate or map click location."""
    images: List[RSImage] = []

    if image_a is not None and image_a.filename:
        content_a = await image_a.read()
        if len(content_a) > 0:
            images.append(GeoTIFFLoader.load_bytes(content_a, filename=image_a.filename, modality="optical"))

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        norm_x = body.get("norm_x", norm_x or 0.5)
        norm_y = body.get("norm_y", norm_y or 0.5)
        lat = body.get("lat", lat)
        lon = body.get("lon", lon)
        sample_id = body.get("sample_id", sample_id)
        custom_question = body.get("custom_question", custom_question)
        language = body.get("language", language or "en-US")

    if norm_x is None:
        norm_x = 0.5
    if norm_y is None:
        norm_y = 0.5

    # Load image from sample or default GIS optical scene
    if len(images) == 0 and sample_id:
        sample = next((s for s in SAMPLE_CATALOG if s["id"] == sample_id), None)
        if sample and len(sample["images"]) > 0:
            fpath = resolve_sample_file(sample["images"][0]["filename"])
            images.append(GeoTIFFLoader.load(fpath, modality=sample["images"][0]["modality"]))
    if len(images) == 0:
        sample_path = resolve_sample_file("single_optical_scene.tif")
        if sample_path.exists():
            images.append(GeoTIFFLoader.load(sample_path, modality="optical"))

    if len(images) == 0:
        raise HTTPException(status_code=400, detail="No imagery available to perform point-and-query analysis.")

    target_img = images[0]
    result = spatial_nlp.analyze_point_location(
        image=target_img,
        norm_x=float(norm_x),
        norm_y=float(norm_y),
        lat=float(lat) if lat is not None else None,
        lon=float(lon) if lon is not None else None,
        language=language or "en-US",
    )

    result["visual_overlays"] = {
        "boxes": [
            {
                "label": f"Point Pin: {result['feature_class']}",
                "box_2d": result["pin_box"],
                "score": result["confidence"],
            }
        ]
    }
    result["answer"] = result["detailed_text"]
    result["reply"] = result["detailed_text"]
    result["task_type"] = "point_and_query_diagnostic"
    result["diagnostic"] = {
        "land_cover_class": result.get("feature_class"),
        "confidence": result.get("confidence"),
        "pixel_x": result.get("pixel_x"),
        "pixel_y": result.get("pixel_y"),
        "ndvi": result.get("ndvi"),
        "ndwi": result.get("ndwi"),
        "ndbi": result.get("ndbi"),
        "sar_backscatter_db": result.get("sar_backscatter_db"),
        "diagnostic_summary": result.get("diagnostic_summary"),
        "soil_advisory": result.get("soil_advisory"),
        "soil_info": result.get("soil_info"),
        "crop_advisory": result.get("crop_advisory"),
    }

    return result


@router.get("/chat/history/{session_id}")
def get_chat_history(session_id: str):
    history = custom_chatbot.sessions.get(session_id, [])
    return {"session_id": session_id, "messages": history}


@router.post("/report/download")
def download_report(payload: Dict[str, Any]):
    """Generates and downloads an auditable execution report (JSON or PDF)."""
    fmt = payload.get("format", "json").lower()
    query = payload.get("query", "Satellite Imagery Query")
    task = payload.get("task_type", "Remote Sensing Analysis")
    answer = payload.get("answer", "")
    confidence = payload.get("confidence", 0.0)
    trace = payload.get("trace", {})

    if fmt == "pdf":
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib import colors

            buf = io.BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
            styles = getSampleStyleSheet()

            story = []
            title_style = ParagraphStyle("Title", parent=styles["Heading1"], fontSize=18, textColor=colors.HexColor("#1e293b"))
            h2_style = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=14, textColor=colors.HexColor("#0f766e"))
            body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=14)

            # Header
            story.append(Paragraph("SatQuery AI — Auditable Execution Report", title_style))
            story.append(Spacer(1, 10))

            # Query Summary Table
            meta_data = [
                ["Field", "Value"],
                ["Natural Language Query", query],
                ["Inferred Task", task],
                ["Overall Confidence", f"{float(confidence)*100:.1f}%"],
                ["Timestamp", trace.get("timestamp", "N/A")],
                ["Total Latency", f"{trace.get('total_latency_ms', 0):.2f} ms"],
            ]
            t = Table(meta_data, colWidths=[150, 370])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ]))
            story.append(t)
            story.append(Spacer(1, 15))

            # Output Answer
            story.append(Paragraph("Synthesized Evidence & Answer", h2_style))
            story.append(Spacer(1, 6))
            story.append(Paragraph(answer, body_style))
            story.append(Spacer(1, 15))

            # Execution Trace Steps
            story.append(Paragraph("Auditable Execution Trace Steps", h2_style))
            story.append(Spacer(1, 6))

            steps_table = [["Step", "Model / Tool", "Latency", "Status", "Summary"]]
            for s in trace.get("steps", []):
                steps_table.append([
                    str(s.get("step_id", "")),
                    s.get("tool_or_model", ""),
                    f"{s.get('duration_ms', 0):.1f} ms",
                    s.get("status", "success"),
                    s.get("summary", ""),
                ])

            if len(steps_table) > 1:
                st = Table(steps_table, colWidths=[35, 140, 65, 55, 225])
                st.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]))
                story.append(st)

            doc.build(story)
            pdf_bytes = buf.getvalue()
            return Response(
                content=pdf_bytes,
                media_type="application/pdf",
                headers={"Content-Disposition": "attachment; filename=satquery_execution_report.pdf"},
            )
        except Exception as e:
            # Fallback to JSON if reportlab fails
            return JSONResponse(content={"error": f"PDF generation failed: {str(e)}", "payload": payload})

    return Response(
        content=json.dumps(payload, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=satquery_audit_trace.json"},
    )
