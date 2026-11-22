import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

class CropDiseaseDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        return image, label

def load_data_from_directory(root_dir):
    all_files = []
    all_labels = []
    valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    
    category_dirs = []
    for dirpath, _, _ in os.walk(root_dir):
        files_in_dir = []
        for ext in valid_exts:
            files_in_dir.extend(glob.glob(os.path.join(dirpath, ext)))
        if len(files_in_dir) > 0:
            category_dirs.append((dirpath, sorted(files_in_dir)))

    category_dirs = sorted(category_dirs, key=lambda x: os.path.basename(x[0]))
    class_names = [os.path.basename(c[0]) for c in category_dirs]
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    total_images = 0
    for path, f_list in category_dirs:
        cls_name = os.path.basename(path)
        cls_idx = class_to_idx[cls_name]
        all_files.extend(f_list)
        all_labels.extend([cls_idx] * len(f_list))
        print(f"📂 Loaded {len(f_list)} images from {path}")
        total_images += len(f_list)

    print(f"\n✅ Total Images Loaded: {total_images}")
    print(f"🏷️  Classes ({len(class_names)}): {class_names}\n")
    return all_files, all_labels, class_names

def get_dataloaders(root_dir, batch_size=32, num_workers=0):
    file_paths, labels, class_names = load_data_from_directory(root_dir)

    train_files, test_files, train_lbl, test_lbl = train_test_split(
        file_paths, labels, test_size=0.30, random_state=42, stratify=labels
    )
    val_files, test_files, val_lbl, test_lbl = train_test_split(
        test_files, test_lbl, test_size=0.50, random_state=42, stratify=test_lbl
    )

    train_tfms = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_tfms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = CropDiseaseDataset(train_files, train_lbl, transform=train_tfms)
    val_ds = CropDiseaseDataset(val_files, val_lbl, transform=eval_tfms)
    test_ds = CropDiseaseDataset(test_files, test_lbl, transform=eval_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    return train_loader, val_loader, test_loader, class_names