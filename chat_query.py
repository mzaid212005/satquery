import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from backend.config import settings
from backend.controller.custom_chatbot import custom_chatbot
from backend.controller.orchestrator import orchestrator
from data.geotiff_loader import GeoTIFFLoader
from data.validator import InputValidator


def run_query(query: str, image_a_path: str = None, image_b_path: str = None, modality_a: str = "optical", modality_b: str = "sar"):
    images = []
    if image_a_path:
        path_a = Path(image_a_path)
        if not path_a.exists():
            path_a = settings.sample_dir / image_a_path
            if not path_a.exists():
                path_a = settings.sample_dir / "heldout_isro" / image_a_path
        if path_a.exists():
            images.append(GeoTIFFLoader.load(path_a, modality=modality_a))

    if image_b_path:
        path_b = Path(image_b_path)
        if not path_b.exists():
            path_b = settings.sample_dir / image_b_path
            if not path_b.exists():
                path_b = settings.sample_dir / "heldout_isro" / image_b_path
        if path_b.exists():
            images.append(GeoTIFFLoader.load(path_b, modality=modality_b))

    print("\n" + "=" * 70)
    print("[SATQUERY AI] EXECUTING CHATBOT / QUERY PIPELINE")
    print("=" * 70)
    print(f"Query: \"{query}\"")
    if images:
        print(f"Image A: {images[0].modality.upper()} ({images[0].width}x{images[0].height}, {images[0].metadata.format_name})")
        if len(images) > 1:
            print(f"Image B: {images[1].modality.upper()} ({images[1].width}x{images[1].height}, {images[1].metadata.format_name})")
    else:
        print("Mode: Knowledge-Guided Conversational Chatbot (No imagery attached)")
    print("-" * 70)

    res = custom_chatbot.chat(
        query=query,
        images=images if len(images) > 0 else None,
    )

    print(f"\n[Inferred Task]: {res['task_type']}")
    print(f"[Confidence]: {res['confidence'] * 100:.1f}%\n")
    print(f"[SatQuery Custom Chatbot Response]:\n{res['reply']}\n")

    overlays = res.get("visual_overlays", {})
    if overlays.get("boxes"):
        print("[Visual Grounding Detections]:")
        for i, box in enumerate(overlays["boxes"], 1):
            print(f"  {i}. {box['label']} (Score: {(box.get('score', 0.9))*100:.1f}%) -> BBox: {box['box_2d']}")
    if overlays.get("change_direction"):
        print(f"[Change Analysis]: Direction = {overlays['change_direction']}, Modified Pixels = {overlays.get('change_ratio_pct')}%")
    if overlays.get("water_coverage_pct"):
        print(f"[Cross-Modal Coverage]: Water = {overlays['water_coverage_pct']}%, Built-Up = {overlays['builtup_coverage_pct']}%")

    trace = res.get("trace", {})
    if trace.get("steps"):
        print("\n[Auditable Trace Steps]:")
        for step in trace["steps"]:
            print(f"  * Step {step.get('step_id')}: {step.get('step_name')} ({step.get('tool_or_model')}) | {step.get('duration_ms', 0):.1f} ms | Status: {step.get('status')}")
        print(f"  Total Pipeline Latency: {trace.get('total_latency_ms', 0):.2f} ms")
    print("=" * 70 + "\n")
    return res


def interactive_mode():
    print("\n" + "=" * 70)
    print("[SATQUERY AI] INTERACTIVE CHATBOT & QUERY CONSOLE")
    print("=" * 70)
    print("Ask any satellite, precision agriculture, soil, or remote sensing query.")
    print("Type 'exit' or 'quit' to close.\n")

    session_id = f"cli_{Path('.').stat().st_mtime}"

    while True:
        try:
            q = input("\n💬 Enter Query (or press Enter for sample prompt): ").strip()
            if q.lower() in ["exit", "quit", "q"]:
                break
            if not q:
                q = "Suggest crop, vegetable, soil level etc object highlight that part"

            attach = input("📷 Attach image? (y/n, default: n): ").strip().lower()
            if attach == "y":
                default_dir = settings.sample_dir
                img_a = input(f"📁 Image A Path [default: {default_dir / 'single_optical_scene.tif'}]: ").strip()
                if not img_a:
                    img_a = str(default_dir / "single_optical_scene.tif")
                mod_a = input("📡 Image A Modality (optical/multispectral/sar) [default: optical]: ").strip() or "optical"
                img_b = input("📁 Image B Path (optional, press Enter to skip): ").strip()
                mod_b = "sar"
                if img_b:
                    mod_b = input("📡 Image B Modality (optical/multispectral/sar) [default: sar]: ").strip() or "sar"
                run_query(query=q, image_a_path=img_a, image_b_path=img_b or None, modality_a=mod_a, modality_b=mod_b)
            else:
                run_query(query=q)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting SatQuery console.")
            break


def main():
    parser = argparse.ArgumentParser(description="SatQuery AI Custom Query CLI & Chat Tool")
    parser.add_argument("--query", "-q", type=str, help="Natural language query")
    parser.add_argument("--image-a", "-a", type=str, default=None, help="Path to primary remote sensing image")
    parser.add_argument("--modality-a", type=str, default="optical", choices=["optical", "multispectral", "sar"], help="Modality of image A")
    parser.add_argument("--image-b", "-b", type=str, default=None, help="Path to secondary image (for pair tasks)")
    parser.add_argument("--modality-b", type=str, default="sar", choices=["optical", "multispectral", "sar"], help="Modality of image B")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive console session")
    args = parser.parse_args()

    if args.interactive or (not args.query and not args.image_a):
        interactive_mode()
    else:
        run_query(
            query=args.query,
            image_a_path=args.image_a,
            image_b_path=args.image_b,
            modality_a=args.modality_a,
            modality_b=args.modality_b,
        )


if __name__ == "__main__":
    main()
