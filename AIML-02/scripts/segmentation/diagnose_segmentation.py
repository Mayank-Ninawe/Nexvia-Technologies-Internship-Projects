from pathlib import Path

import torch
import yaml
from monai.data import DataLoader
from monai.inferers import sliding_window_inference

from ml.segmentation.model import create_spleen_unet, dice_score
from ml.segmentation.preprocessing import IMAGE_KEY, LABEL_KEY, create_dataset

config = yaml.safe_load(Path("ml/segmentation/config.yaml").read_text())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(
    "models/segmentation/best_spleen_unet.pt",
    map_location=device,
    weights_only=False,
)

model = create_spleen_unet()
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device).eval()

dataset = create_dataset(config, limit=1)
loader = DataLoader(dataset, batch_size=1, num_workers=0)
sample = next(iter(loader))
image, target = sample[IMAGE_KEY].to(device), sample[LABEL_KEY].to(device)

with torch.no_grad():
    logits = sliding_window_inference(image, (96, 96, 96), 1, model, overlap=0.5, mode="gaussian")
    probs = torch.sigmoid(logits)

print("Checkpoint epoch:", checkpoint.get("epoch"))
print("Logits min/max/mean:", logits.min().item(), logits.max().item(), logits.mean().item())
print("Probabilities min/max/mean:", probs.min().item(), probs.max().item(), probs.mean().item())
print("Probability >= 0.5 count:", int((probs >= 0.5).sum().item()))
print("Target spleen pixels:", int(target.sum().item()))
print("Dice:", dice_score(logits, target))
