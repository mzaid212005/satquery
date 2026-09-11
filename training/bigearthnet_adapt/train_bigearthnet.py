import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from backend.models.backbone.rs_backbone import RSVisionBackbone, BIGEARTHNET_19_CLASSES
from training.bigearthnet_adapt.dataset import BigEarthNetDataset


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: str) -> float:
    model.train()
    total_loss = 0.0
    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        out = model(images)
        logits = out["logits"]
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / max(1, len(loader))


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            out = model(images)
            probs = out["probabilities"]
            all_preds.append((probs > 0.5).cpu().numpy())
            all_targets.append(labels.numpy())

    # Calculate micro F1 score
    preds = (probs > 0.5).float()
    correct = (preds == labels.to(device)).float().sum()
    accuracy = correct / (labels.numel())
    return float(accuracy.item())


def main():
    parser = argparse.ArgumentParser(description="Fine-tune RS Vision Backbone on BigEarthNet.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="backend/models/weights", help="Checkpoint output directory")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[BigEarthNet Domain Adaptation] Device: {device} | Epochs: {args.epochs} | LR: {args.lr}")

    train_dataset = BigEarthNetDataset(split="train", num_synthetic_samples=48)
    val_dataset = BigEarthNetDataset(split="val", num_synthetic_samples=16)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = RSVisionBackbone(in_channels=3, embed_dim=256, num_classes=len(BIGEARTHNET_19_CLASSES))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {loss:.4f} | Val Accuracy: {val_acc*100:.2f}%")

    checkpoint_path = out_dir / "rs_backbone_bigearthnet.pth"
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "classes": BIGEARTHNET_19_CLASSES,
    }, checkpoint_path)
    print(f"[BigEarthNet Domain Adaptation] Checkpoint successfully saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
