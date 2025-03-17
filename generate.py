# generate.py (Optimized with Batch Processing, Path Normalization, and Duplicate Handling)
import os
import numpy as np
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class MobileNetEmbedder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        self.model.classifier = torch.nn.Identity()

    def forward(self, x):
        return self.model(x)

class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        return self.transform(image), image_path

    def __len__(self):
        return len(self.image_paths)

def load_existing_embeddings(embedding_path='embeddings.npz'):
    """Load existing embeddings and paths from a file."""
    if os.path.exists(embedding_path) and os.path.getsize(embedding_path) > 0:
        try:
            data = np.load(embedding_path)
            return data['embeddings'], data['paths'].tolist()
        except (EOFError, OSError, KeyError):
            print("Warning: embeddings.npz is corrupted or incomplete. Creating new embeddings.")
            return np.array([]), []
    return np.array([]), []


def generate_embeddings(image_folder, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MobileNetEmbedder().to(device)
    model.eval()

    # Normalize image folder path
    image_folder = image_folder.rstrip('/')  # Remove trailing slash if present

    # Load existing embeddings and paths
    existing_embeddings, existing_paths = load_existing_embeddings()

    # Get all image paths in the folder
    all_image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
                       if f.endswith(('.jpg', '.jpeg', '.png', '.svg'))]

    # Filter out images that already have embeddings
    new_image_paths = [path for path in all_image_paths
                       if os.path.relpath(path, 'static').replace('\\', '/') not in existing_paths]

    if not new_image_paths:
        print("No new images to process.")
        return

    # Process only new images
    dataset = ImageDataset(new_image_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    new_embeddings = []
    new_paths = []
    with torch.no_grad():
        for batch, (images, image_paths) in enumerate(dataloader):
            images = images.to(device)
            embeds = model(images).cpu().numpy()
            new_embeddings.append(embeds)
            new_paths.extend([os.path.relpath(p, 'static').replace('\\', '/') for p in image_paths])

    # Combine new embeddings with existing ones
    if existing_embeddings.size > 0:
        embeddings = np.vstack([existing_embeddings] + new_embeddings)
        paths = existing_paths + new_paths
    else:
        embeddings = np.vstack(new_embeddings)
        paths = new_paths

    # Save updated embeddings
    np.savez_compressed('embeddings.npz', embeddings=embeddings, paths=paths)
    print(f"Processed {len(new_image_paths)} new images. Total embeddings: {len(paths)}.")

# Example usage:
generate_embeddings(image_folder='static/images', batch_size=64)
