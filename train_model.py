import os
import sys
import io
import cv2
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Fix Unicode / charmap encoding errors on Windows when output is redirected
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

# Determine base paths: check local workspace directory first, then external fallback
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DEFECT_DIR = os.path.join(BASE_DIR, "models", "Diffected with no greyscale")
LOCAL_NORMAL_DIR = os.path.join(BASE_DIR, "models", "Non diffected with greyscale")

EXTERNAL_DEFECT_DIR = r"C:/Users/MOWLIESWARAN/OneDrive/Documents/mini-project-main/Diffected with no greyscale"
EXTERNAL_NORMAL_DIR = r"C:/Users/MOWLIESWARAN/OneDrive/Documents/mini-project-main/Non diffected with greyscale"

DEFECT_DIR = LOCAL_DEFECT_DIR if os.path.exists(LOCAL_DEFECT_DIR) else EXTERNAL_DEFECT_DIR
NORMAL_DIR = LOCAL_NORMAL_DIR if os.path.exists(LOCAL_NORMAL_DIR) else EXTERNAL_NORMAL_DIR
SAVE_PATH = os.path.join(BASE_DIR, "models", "defect_classifier.pth")

os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

IMG_SIZE = 256
BATCH_SIZE = 64  
EPOCHS = 3 
LR = 2e-4
VALIDATION_SPLIT = 0.15
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Defect Directory: {DEFECT_DIR}")
print(f"Normal Directory: {NORMAL_DIR}")

def load_images(folder, label):
    data = []
    if not os.path.exists(folder):
        return data
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                path = os.path.join(root, f)
                try:
                    if os.path.getsize(path) > 1000:  
                        data.append((path, label))
                except OSError:
                    continue
    return data

class MetalDefectDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        
        # Read image safely (handling unicode paths on Windows)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            try:
                with open(img_path, "rb") as f:
                    img_bytes = bytearray(f.read())
                img = cv2.imdecode(np.asarray(img_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            except Exception:
                # Return empty image if unreadable
                img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        
        # Convert 1-channel grayscale to 3-channel grayscale for ResNet
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, label

def train_epoch(loader, model, criterion, optimizer, device):
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
            print(
                f"  Batch {batch_idx + 1}/{len(loader)} | "
                f"Loss: {loss.item():.4f} | "
                f"Acc: {100 * correct / total:.2f}%",
                flush=True
            )
    
    epoch_loss = total_loss / len(loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate(loader, model, criterion, device):
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(loader)
    avg_acc = 100 * correct / total
    return avg_loss, avg_acc

def main():
    print("\nLoading images...")
    normal_data = load_images(NORMAL_DIR, 0)  
    defect_data = load_images(DEFECT_DIR, 1)  

    print(f"Non-defected (Normal) images: {len(normal_data)}")
    print(f"Defected images: {len(defect_data)}")

    assert len(defect_data) > 0, f"No defect images found in: {DEFECT_DIR}"
    assert len(normal_data) > 0, f"No normal images found in: {NORMAL_DIR}"

    print(f"Class imbalance ratio: {len(defect_data) / len(normal_data):.1f}:1")

    all_data = normal_data + defect_data
    labels_all = [lbl for _, lbl in all_data]

    train_data, val_data = train_test_split(
        all_data,
        test_size=VALIDATION_SPLIT,
        stratify=labels_all,
        random_state=RANDOM_SEED
    )

    print(f"\nData Split:")
    print(f"  Train: {len(train_data)} images")
    print(f"  Validation: {len(val_data)} images")

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_ds = MetalDefectDataset(train_data, transform=train_transform)
    val_ds = MetalDefectDataset(val_data, transform=val_transform)

    print("\nDatasets created")

    train_labels = [lbl for _, lbl in train_data]
    class_counts = np.bincount(train_labels)
    total_samples = len(train_labels)

    class_weights = total_samples / (len(class_counts) * class_counts)

    print(f"\nClass Resampling Weights (for WeightedRandomSampler):")
    print(f"   Normal (0): {class_weights[0]:.4f}")
    print(f"   Defected (1): {class_weights[1]:.4f}")

    sample_weights = [class_weights[lbl] for lbl in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=use_pin_memory
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=use_pin_memory
    )

    print(f"\nDataLoaders created")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")

    print("\nBuilding model...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.to(device)
    print(f"ResNet18 loaded (Binary Classification: Normal vs Defected)")

    # WeightedRandomSampler already balances batches; standard CrossEntropyLoss is used to avoid double penalization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    print(f"Loss: CrossEntropyLoss (batch-balanced via WeightedRandomSampler)")
    print(f"Optimizer: Adam (lr={LR})")

    print("\n" + "="*70)
    print("TRAINING STARTED")
    print("="*70)

    train_losses, val_losses = [], []   
    train_accs, val_accs = [], []
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 70)
        
        train_loss, train_acc = train_epoch(train_loader, model, criterion, optimizer, device)
        val_loss, val_acc = validate(val_loader, model, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        scheduler.step(val_loss)
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"   Train - Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}%")
        print(f"   Val   - Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"   Best validation accuracy! Model saved.")

    print("\n" + "="*70)
    print(f"TRAINING COMPLETED - Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {SAVE_PATH}")
    print("="*70)

    # Save training curves plot
    try:
        plot_path = os.path.join(BASE_DIR, "models", "training_history.png")
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss", marker='o')
        plt.plot(range(1, EPOCHS + 1), val_losses, label="Val Loss", marker='s')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss vs. Epochs")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        plt.subplot(1, 2, 2)
        plt.plot(range(1, EPOCHS + 1), train_accs, label="Train Acc", marker='o')
        plt.plot(range(1, EPOCHS + 1), val_accs, label="Val Acc", marker='s')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy vs. Epochs")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"Training history plot saved to: {plot_path}")
    except Exception as e:
        print(f"Could not generate training plot: {e}")

if __name__ == "__main__":
    main()