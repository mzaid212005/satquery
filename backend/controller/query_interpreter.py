import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from backend.controller.spatial_nlp import SpatialNLPEngine


class QueryIntent(BaseModel):
    raw_query: str
    task_type: str
    workflow: str  # "single_model", "chained"
    chained_tasks: List[str] = Field(default_factory=list)
    target_entities: List[str] = Field(default_factory=list)
    spatial_meta: Optional[Dict[str, Any]] = None
    required_relationship: str  # "single", "bi_temporal", "cross_modal"
    min_images_required: int = 1
    max_images_required: int = 1
    confidence: float = 0.95
    rationale_summary: str = ""


class QueryInterpreter:
    """Interprets natural language queries, classifies remote-sensing task types, and extracts geographic entities."""

    # Remote sensing entity gazetteer / vocabulary
    ENTITY_PATTERNS = {
        "water body": [
            r"\bwater\b",
            r"\bwater\s*body\b",
            r"\briver\b",
            r"\blake\b",
            r"\breservoir\b",
            r"\bpond\b",
            r"\bcoast\b",
            r"\bcanal\b",
            r"\bocean\b",
            r"\bsea\b",
            r"\bflooding\b",
            r"\bflood\b",
        ],
        "built-up": [
            r"\bbuilt[\s-]?up\b",
            r"\burban\b",
            r"\bbuilding(s)?\b",
            r"\bsettlement(s)?\b",
            r"\bhouses?\b",
            r"\bresidential\b",
            r"\bcommercial\b",
            r"\bindustrial\b",
            r"\bconcrete\b",
            r"\bcity\b",
            r"\btown\b",
        ],
        "runway / airport": [
            r"\brunway(s)?\b",
            r"\bairfield\b",
            r"\bairport\b",
            r"\btarmac\b",
            r"\bhangar\b",
            r"\baircraft\b",
        ],
        "vegetation / forest": [
            r"\bforest(s)?\b",
            r"\bvegetation\b",
            r"\btrees?\b",
            r"\bwoodland\b",
            r"\bcanopy\b",
            r"\bgreener(y)?\b",
        ],
        "agricultural land": [
            r"\bagricultur(al|e)\b",
            r"\bcrops?\b",
            r"\bcropland\b",
            r"\bvegetable(s)?\b",
            r"\bsoil\b",
            r"\bsoil\s*level\b",
            r"\bfarms?\b",
            r"\bfields?\b",
            r"\bpaddy\b",
            r"\bpasture\b",
            r"\bfarming\b",
        ],
        "port / harbor": [
            r"\bport\b",
            r"\bharbor\b",
            r"\bdock(s)?\b",
            r"\bships?\b",
            r"\bvessels?\b",
        ],
        "road / infrastructure": [
            r"\broad(s)?\b",
            r"\bhighway\b",
            r"\bbridge\b",
            r"\brail(way)?\b",
        ],
    }

    @classmethod
    def extract_entities(cls, query: str) -> List[str]:
        q_lower = query.lower()
        extracted = []
        for entity_name, patterns in cls.ENTITY_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, q_lower):
                    extracted.append(entity_name)
                    break
        return extracted

    @classmethod
    def interpret(cls, query: str, image_count: int = 1) -> QueryIntent:
        q = query.strip()
        q_lower = q.lower()
        entities = cls.extract_entities(q)
        spatial_info = SpatialNLPEngine.extract_spatial_entities(q)

        # 0. Point-and-Query Spatial Diagnostic Detection
        if spatial_info["has_spatial_ref"] and (
            spatial_info["spatial_type"] == "coordinate_point"
            or any(w in q_lower for w in ["what is at", "identify point", "diagnose coordinate", "at latitude", "at point", "pinned", "location", "point location", "coordinates"])
        ):
            return QueryIntent(
                raw_query=q,
                task_type="point_and_query_diagnostic",
                workflow="single_model",
                target_entities=entities,
                spatial_meta=spatial_info,
                required_relationship="single",
                min_images_required=1,
                max_images_required=1,
                confidence=0.98,
                rationale_summary="Query specifies explicit spatial coordinates/quadrant for localized point analysis.",
            )

        # 1. Optical-SAR Fusion Task Detection
        # Representative queries:
        # "Use the optical and SAR images together to identify built-up and water-covered regions."
        if any(
            phrase in q_lower
            for phrase in [
                "optical and sar",
                "sar and optical",
                "both images together",
                "use both images",
                "cross-modal",
                "radar and optical",
                "optical and radar",
                "fuse optical",
                "fusion of optical",
            ]
        ):
            return QueryIntent(
                raw_query=q,
                task_type="optical_sar_fusion",
                workflow="single_model",
                target_entities=entities or ["built-up", "water body"],
                spatial_meta=spatial_info,
                required_relationship="cross_modal",
                min_images_required=2,
                max_images_required=2,
                confidence=0.98,
                rationale_summary="Query explicitly requests joint optical and SAR fusion analysis.",
            )

        # 2. Bi-Temporal Change Detection & Change-VQA
        # Representative queries:
        # "What changed between these two dates, and where did the change occur?"
        # "Has the built-up area increased, decreased, or remained unchanged?"
        change_keywords = [
            "what changed",
            "between these two dates",
            "between the two images",
            "has the built-up area increased",
            "increased, decreased, or remained unchanged",
            "temporal change",
            "difference between",
            "change detection",
            "pre-event and post-event",
            "damage assessment",
            "did the",
            "how has",
        ]
        is_change_query = any(k in q_lower for k in change_keywords) or (
            ("changed" in q_lower or "change" in q_lower) and ("between" in q_lower or "dates" in q_lower or "increased" in q_lower or "decreased" in q_lower)
        )

        if is_change_query or (image_count == 2 and ("change" in q_lower or "difference" in q_lower)):
            is_polar_vqa = any(
                q_lower.startswith(w) for w in ["has", "did", "is", "was", "how", "will", "can"]
            ) or ("increased" in q_lower or "decreased" in q_lower or "unchanged" in q_lower)
            task = "change_vqa" if is_polar_vqa else "change_description"
            return QueryIntent(
                raw_query=q,
                task_type=task,
                workflow="single_model",
                target_entities=entities,
                required_relationship="bi_temporal",
                min_images_required=2,
                max_images_required=2,
                confidence=0.97,
                rationale_summary="Query indicates bi-temporal change comparison or change-based visual question answering.",
            )

        # 3. Grounding / Region Localization
        # Representative query:
        # "Highlight the water body referred to in the query."
        grounding_keywords = [
            "highlight",
            "localize",
            "locate",
            "detect the",
            "bounding box",
            "bbox",
            "find the",
            "where is",
            "point to",
            "draw a box",
            "segment the",
            "highlight that",
            "highlight that party",
            "highlight that part",
            "show where",
            "identify and highlight",
        ]
        if any(k in q_lower for k in grounding_keywords):
            return QueryIntent(
                raw_query=q,
                task_type="region_grounding",
                workflow="single_model",
                target_entities=entities or ["target_region"],
                required_relationship="single",
                min_images_required=1,
                max_images_required=1,
                confidence=0.95,
                rationale_summary="Query asks to highlight or localize spatial objects/regions.",
            )

        # 4. Captioning / Scene Description
        # Representative query:
        # "Describe the land-cover and major objects visible in this image."
        captioning_keywords = [
            "describe",
            "description",
            "caption",
            "overview of this image",
            "summarize the scene",
            "what does this scene show",
        ]
        if any(k in q_lower for k in captioning_keywords):
            return QueryIntent(
                raw_query=q,
                task_type="captioning",
                workflow="single_model",
                target_entities=entities,
                required_relationship="single",
                min_images_required=1,
                max_images_required=1,
                confidence=0.94,
                rationale_summary="Query asks for a comprehensive scene description or land-cover caption.",
            )

        # 5. Single-Image RS VQA (Default for question-style queries)
        return QueryIntent(
            raw_query=q,
            task_type="rs_vqa",
            workflow="single_model",
            target_entities=entities,
            required_relationship="single",
            min_images_required=1,
            max_images_required=1,
            confidence=0.90,
            rationale_summary="Interpreted as Remote Sensing Visual Question Answering query.",
        )
