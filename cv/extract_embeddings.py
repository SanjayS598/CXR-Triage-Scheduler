import torch
import torchxrayvision as xrv
import skimage.io
from skimage import io
import torchvision.transforms as transforms
import torch.nn.functional as F

class extractor():
    def __init__(self, healthy_embedding_path: str="healthy_mean_weight/healthy_mean_embedding.pt", device: str="cuda"):
        self.device = torch.device(device)
        self.model = xrv.models.DenseNet(weights="densenet121-res224-all").to(self.device)
        self.model.eval()

        self.healthy_embedding = torch.load(healthy_embedding_path).to(self.device)

        if self.healthy_embedding.dim() >= 3:
            self.healthy_emb = F.adaptive_avg_pool2d(self.healthy_embedding, (1, 1)).view(1, -1)
        else:
            self.healthy_emb = self.healthy_embedding.view(1, -1)
        
        # self.healthy_emb = self.healthy_embedding.unsqueeze(0)  # shape [1, 1024]

        # transformation for the nn
        self.transform = transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224)
        ])

    def process(self, img_path: str):
        img = io.imread(img_path)
        # If the image is 3D (has color/alpha channels), average them to grayscale
        if len(img.shape) == 3:
            img = img.mean(2)[None, ...]
        # If the image is already 2D (grayscale), just add the channel dimension
        else:
            img = img[None, ...] 
            
        img = self.transform(img)
        img = torch.from_numpy(img).float()
        return img.to(self.device)

    def compute_embedding(self, img: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.model.features(img[None, ...])
            pooled = F.adaptive_avg_pool2d(features, (1, 1))
            emb = pooled.view(1, -1)   # shape [1, 1024]
        return emb

    def compute_unhealthy_score(self, img_path: str, metric: str = "euclidean") -> float:
        img = self.process(img_path)
        embedding = self.compute_embedding(img)

        if metric == "cosine":
            score = 1 - F.cosine_similarity(embedding, self.healthy_emb)
        elif metric == "euclidean":
            score = F.pairwise_distance(embedding, self.healthy_emb)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        return score.item()
