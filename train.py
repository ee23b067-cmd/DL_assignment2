import argparse
import os

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from losses.iou_loss import IoULoss
from models.multitask import MultiTaskPerceptionModel


# -----------------------------
# DEVICE SELECTION
# -----------------------------
def get_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")


    return torch.device("cuda")


def save_checkpoint(path, model, epoch, best_metric):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
        },
        path,
    )


# -----------------------------
# DATA LOADER
# -----------------------------
def get_dataloader(root_dir, batch_size, split="train", need_bbox=False):
    dataset = OxfordIIITPetDataset(
        root=root_dir,
        split=split,
        transform=None,
        bbox=need_bbox,
        target_size=(224, 224),
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=0)


# -----------------------------
# CLASSIFIER
# -----------------------------
def train_classifier(root_dir, epochs=25, batch_size=32, lr=3e-4):
    device = get_device()
    print(f"Using device: {device}")
    
    train_loader = get_dataloader(root_dir, batch_size, split="train", need_bbox=False)
    val_loader = get_dataloader(root_dir, batch_size, split="test", need_bbox=False)

    model = VGG11Classifier(num_classes=37).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    os.makedirs("checkpoints", exist_ok=True)

    print("\n🚀 Training Classifier...")
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for i, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

            print(f"Epoch {epoch+1} | Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        train_acc = 100 * correct / total
        
        # Validation phase
        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, pred = outputs.max(1)
                val_total += labels.size(0)
                val_correct += pred.eq(labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        print(f"Epoch {epoch+1}: Train Loss={total_loss/len(train_loader):.4f}, Train Acc={train_acc:.2f}%, Val Loss={val_loss/len(val_loader):.4f}, Val Acc={val_acc:.2f}%")
        
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint("checkpoints/classifier.pth", model, epoch + 1, best_acc)
            print(f"✨ New best Val Acc: {val_acc:.2f}%. Saved checkpoint.")

    print(f"✅ Training Finished. Best Val Acc: {best_acc:.2f}%")


# -----------------------------
# LOCALIZER
# -----------------------------
def train_localizer(root_dir, epochs=25, batch_size=32, lr=3e-4):
    device = get_device()

    train_loader = get_dataloader(root_dir, batch_size, split="train", need_bbox=True)
    val_loader = get_dataloader(root_dir, batch_size, split="test", need_bbox=True)

    model = VGG11Localizer().to(device)

    mse = nn.MSELoss()
    iou = IoULoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    os.makedirs("checkpoints", exist_ok=True)

    print("\n🚀 Training Localizer...")
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            boxes = batch["bbox"].to(device)

            # Convert to cx, cy, w, h (normalized)
            cx = boxes[:, 0]
            cy = boxes[:, 1]
            w = boxes[:, 2]
            h = boxes[:, 3]

            target = torch.stack([cx, cy, w, h], dim=1)

            optimizer.zero_grad()
            out = model(images)

            loss = mse(out, target) + iou(out, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                boxes = batch["bbox"].to(device)
                cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                target = torch.stack([cx, cy, w, h], dim=1)
                out = model(images)
                loss = mse(out, target) + iou(out, target)
                val_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            save_checkpoint("checkpoints/localizer.pth", model, epoch + 1, best_loss)
            print(f"✨ New best Val Loss: {avg_val_loss:.4f}. Saved checkpoint.")

    print("✅ Finished Localizer")


# -----------------------------
# SEGMENTATION
# -----------------------------
def train_segmentation(root_dir, epochs=25, batch_size=16, lr=3e-4):
    device = get_device()

    train_loader = get_dataloader(root_dir, batch_size, split="train", need_bbox=False)
    val_loader = get_dataloader(root_dir, batch_size, split="test", need_bbox=False)

    model = VGG11UNet(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    os.makedirs("checkpoints", exist_ok=True)

    print("\n🚀 Training UNet...")
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            out = model(images)

            loss = criterion(out, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                out = model(images)
                loss = criterion(out, masks)
                val_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            save_checkpoint("checkpoints/unet.pth", model, epoch + 1, best_loss)
            print(f"✨ New best Val Loss: {avg_val_loss:.4f}. Saved checkpoint.")

    print("✅ Finished UNet")


# -----------------------------
# MULTITASK
# -----------------------------
def train_multitask(
    root_dir,
    epochs=25,
    batch_size=16,
    lr=1e-4,
    classifier_path="checkpoints/classifier.pth",
    localizer_path="checkpoints/localizer.pth",
    unet_path="checkpoints/unet.pth",
):
    device = get_device()

    train_loader = get_dataloader(root_dir, batch_size, split="train", need_bbox=True)
    val_loader = get_dataloader(root_dir, batch_size, split="test", need_bbox=True)

    model = MultiTaskPerceptionModel(
        num_breeds=37,
        seg_classes=3,
        in_channels=3,
        classifier_path=classifier_path,
        localizer_path=localizer_path,
        unet_path=unet_path,
        drive=False,
    ).to(device)

    cls_criterion = nn.CrossEntropyLoss()
    mse_criterion = nn.MSELoss()
    iou_criterion = IoULoss()
    seg_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    os.makedirs("checkpoints", exist_ok=True)

    print("\n🚀 Training MultiTask Model...")
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            boxes = batch["bbox"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            outputs = model(images)

            cls_loss = cls_criterion(outputs["classification"], labels)
            bbox_loss = mse_criterion(outputs["localization"], boxes) + iou_criterion(outputs["localization"], boxes)
            seg_loss = seg_criterion(outputs["segmentation"], masks)
            loss = cls_loss + bbox_loss + seg_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                boxes = batch["bbox"].to(device)
                masks = batch["mask"].to(device)

                outputs = model(images)
                cls_loss = cls_criterion(outputs["classification"], labels)
                bbox_loss = mse_criterion(outputs["localization"], boxes) + iou_criterion(outputs["localization"], boxes)
                seg_loss = seg_criterion(outputs["segmentation"], masks)
                loss = cls_loss + bbox_loss + seg_loss
                val_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            save_checkpoint("checkpoints/multitask.pth", model, epoch + 1, best_loss)
            print(f"✨ New best Val Loss: {avg_val_loss:.4f}. Saved checkpoint.")

    print("✅ Finished MultiTask Model")


# -----------------------------
# MAIN
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train task-specific or multitask models.")
    parser.add_argument("--mode", choices=["all", "classifier", "localizer", "segmenter", "multitask"], default="all")
    parser.add_argument("--root", default="oxford_pet")
    args = parser.parse_args()

    if args.mode == "classifier":
        print("Training Classification head...")
        train_classifier(args.root)
    elif args.mode == "localizer":
        print("Training Localization head...")
        train_localizer(args.root)
    elif args.mode == "segmenter":
        print("Training Segmentation head...")
        train_segmentation(args.root)
    elif args.mode == "multitask":
        print("Training MultiTask model...")
        train_multitask(args.root)
    else:
        print("Training Classification head...")
        train_classifier(args.root)
        print("Training Localization head...")
        train_localizer(args.root)
        print("Training Segmentation head...")
        train_segmentation(args.root)


if __name__ == "__main__":
    main()
