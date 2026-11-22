import os
import sys
from collections import Counter
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from dataset_loader import get_dataloaders
from model import BioMamba
from loss import BioDiffusionLoss

def plot_curves(history, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    epochs_range = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(14, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2, marker='o')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss', color='#ff7f0e', linewidth=2, marker='s')
    plt.title('Training & Validation Loss Curve', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Train Accuracy', color='#2ca02c', linewidth=2, marker='o')
    plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy', color='#d62728', linewidth=2, marker='s')
    plt.title('Training & Validation Accuracy Curve', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('Accuracy (%)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "training_validation_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"📊 Training & Validation curves saved to: {plot_path}")

def plot_confusion_matrix(all_preds, all_targets, class_names, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(all_targets, all_preds)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues', 
        xticklabels=class_names, 
        yticklabels=class_names,
        cbar=True
    )
    plt.title('BioMamba Normalized Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=11, fontweight='bold')
    plt.ylabel('Ground Truth Class', fontsize=11, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"🎯 Confusion matrix saved to: {cm_path}")

def main():
    if not torch.cuda.is_available():
        threads = os.cpu_count() or 4
        torch.set_num_threads(threads)
        device = torch.device("cpu")
        print(f"🚀 Training BioMamba on CPU utilizing {threads} threads.")
    else:
        device = torch.device("cuda")
        print(f"🚀 Training BioMamba on GPU: {torch.cuda.get_device_name(0)}")

    candidate_paths = [
        os.path.join(os.getcwd(), "dataset"),
        r"F:\Desktop\Abdul\College\IV - SEMESTER\NCML\Project\Crop Diseases Dataset\Crop Diseases\dataset",
        os.getcwd()
    ]
    
    dataset_path = None
    for p in candidate_paths:
        if os.path.exists(p) and any(os.path.isdir(os.path.join(p, d)) for d in os.listdir(p)):
            dataset_path = p
            break
            
    if dataset_path is None:
        raise FileNotFoundError("Could not find dataset directory.")

    print(f"📁 Reading dataset from: {dataset_path}")

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        root_dir=dataset_path, batch_size=32, num_workers=0
    )

    num_classes = len(class_names)

    # Balanced class weights
    all_train_labels = [y for _, y in train_loader.dataset]
    counts = Counter(all_train_labels)
    total_samples = len(all_train_labels)
    weights = [total_samples / (num_classes * counts[i]) for i in range(num_classes)]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    model = BioMamba(num_classes=num_classes, d_model=128, d_state=16).to(device)
    criterion = BioDiffusionLoss(alpha=0.002, class_weights=class_weights, label_smoothing=0.02)

    # Target: 15 epochs
    total_epochs = 15
    start_epoch = 11

    os.makedirs("weights", exist_ok=True)
    best_weights_path = os.path.join("weights", "biomamba_best.pth")
    checkpoint_file = os.path.join("weights", "checkpoint_latest.pth")

    # Load existing checkpoint
    resume_target = None
    if os.path.exists(checkpoint_file):
        resume_target = checkpoint_file
    elif os.path.exists(best_weights_path):
        resume_target = best_weights_path

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    if resume_target:
        print(f"🔄 Loading weights from: {resume_target}")
        checkpoint = torch.load(resume_target, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        saved_epoch = checkpoint.get('epoch', 10)
        start_epoch = saved_epoch + 1
        if 'history' in checkpoint:
            history = checkpoint['history']
        print(f"✅ Checkpoint verified. Resuming from Epoch {start_epoch} to {total_epochs}.")
    else:
        print("⚠️ No previous weights found. Starting from Epoch 1.")
        start_epoch = 1

    # Low fine-tuning learning rate to push past 98%
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(total_epochs - start_epoch + 1), eta_min=1e-6
    )

    print("\n========================= EXTENDED FINE-TUNING =========================")
    print("🎯 Target: Push Validation Accuracy past 98.5% and extract review plots.")

    best_val_acc = checkpoint.get('val_acc', 97.85) if resume_target else 0.0

    try:
        for epoch in range(start_epoch, total_epochs + 1):
            model.train()
            running_loss = 0.0
            correct, total = 0, 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                logits, ssm_states, _ = model(images)
                loss, ce, diff = criterion(logits, labels, ssm_states)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(train_loader):
                    b_acc = (preds == labels).sum().item() / labels.size(0) * 100
                    print(f"Epoch [{epoch}/{total_epochs}] | Batch [{batch_idx+1}/{len(train_loader)}] | "
                          f"Loss: {loss.item():.4f} | Batch Acc: {b_acc:.1f}%")

            scheduler.step()
            epoch_train_loss = running_loss / len(train_loader)
            epoch_train_acc = (correct / total) * 100

            # Validation Loop
            model.eval()
            val_loss = 0.0
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for val_img, val_lbl in val_loader:
                    val_img, val_lbl = val_img.to(device), val_lbl.to(device)
                    val_out, v_ssm, _ = model(val_img)
                    v_l, _, _ = criterion(val_out, val_lbl, v_ssm)
                    val_loss += v_l.item()
                    val_correct += (val_out.argmax(dim=1) == val_lbl).sum().item()
                    val_total += val_lbl.size(0)

            epoch_val_loss = val_loss / len(val_loader)
            epoch_val_acc = (val_correct / val_total) * 100

            history['train_loss'].append(epoch_train_loss)
            history['val_loss'].append(epoch_val_loss)
            history['train_acc'].append(epoch_train_acc)
            history['val_acc'].append(epoch_val_acc)

            print(f"\n🌟 Epoch {epoch} -> Train Acc: {epoch_train_acc:.2f}% | Val Acc: {epoch_val_acc:.2f}% | "
                  f"Val Loss: {epoch_val_loss:.4f}\n")

            # Save state
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'class_names': class_names,
                'val_acc': epoch_val_acc,
                'history': history
            }, checkpoint_file)

            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'class_names': class_names,
                    'val_acc': epoch_val_acc,
                    'history': history
                }, best_weights_path)
                print(f"🔥 New Best Val Accuracy: {best_val_acc:.2f}%! Saved to {best_weights_path}\n")

    except KeyboardInterrupt:
        print("\n⏸️ Training paused by user. Proceeding to evaluation and plot generation...")

    # Final Evaluation & Confusion Matrix on Held-Out Test Set
    print("\n========================= GENERATING EVALUATION METRICS =========================")
    print("Evaluating best model on held-out test data for confusion matrix...")
    
    checkpoint = torch.load(best_weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for test_img, test_lbl in test_loader:
            test_img = test_img.to(device)
            out, _, _ = model(test_img)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(test_lbl.numpy())

    # Generate output plots
    plot_curves(history, save_dir="plots")
    plot_confusion_matrix(all_preds, all_targets, class_names, save_dir="plots")

    test_accuracy = (np.array(all_preds) == np.array(all_targets)).mean() * 100
    print(f"\n✅ All Work Finished!")
    print(f"🏆 Final Test Split Accuracy: {test_accuracy:.2f}%")
    print(f"📁 Check 'plots/' directory for your slide-ready PNG charts.")

if __name__ == "__main__":
    main()