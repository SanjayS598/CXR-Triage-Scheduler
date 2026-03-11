import torch
import torchxrayvision as xrv
import skimage.io
import torchvision.transforms as transforms
import os
import pandas as pd
from tqdm import tqdm

class HealthyEmbeddingComputer:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self.model = xrv.models.DenseNet(weights="densenet121-res224-all").to(self.device)
        self.model.eval()

        # Same transform as your extractor
        self.transform = transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224)
        ])

    def process_image(self, img_path: str):
        img = skimage.io.imread(img_path)
        if img.ndim == 3:
            img = img.mean(2)[None, ...]  # convert to single channel
        elif img.ndim == 2:
            img = img[None, ...]
        img = self.transform(img)
        img = torch.from_numpy(img).float().to(self.device)
        return img

    def compute_mean_embedding(self, csv_path: str, image_folder: str, save_path: str):
        """
        Compute mean embedding from healthy images listed in CSV.
        """
        # --- Read CSV and filter healthy images ---
        df = pd.read_csv(csv_path)
        healthy_df = df[df['Finding Labels'] == 'No Finding']
        image_files = healthy_df['Image Index'].tolist()

        mean_emb = None
        count = 0

        for img_file in tqdm(image_files, desc="Computing healthy embeddings"):
            img_path = os.path.join(image_folder, img_file)
            if not os.path.exists(img_path):
                continue  # skip missing files

            img = self.process_image(img_path)
            
            with torch.no_grad():
                emb = self.model.features(img[None, ...])  # [1, 1024]

            if mean_emb is None:
                mean_emb = emb.clone()
            else:
                # Online mean calculation
                mean_emb = mean_emb + (emb - mean_emb)/(count+1)

            count += 1

        # Save to disk
        torch.save(mean_emb.squeeze(0).cpu(), save_path)
        print(f"Mean embedding saved to {save_path}")
        return mean_emb

# --- Usage ---
computer = HealthyEmbeddingComputer(device="cuda")

csv_path = "healthy_imgs/sample_labels.csv"
image_folder = "healthy_imgs/images"
save_path = "healthy_mean_embedding.pt"

mean_embedding = computer.compute_mean_embedding(csv_path, image_folder, save_path)
