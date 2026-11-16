import rasterio
import torch

with rasterio.open("Site3_Oct_20261003_ms.tif") as src:
    print("Band count:", src.count)
    print("Band descriptions:", src.descriptions)
    print("Color interpretation:", src.colorinterp)

checkpoint = torch.load(
    "lantana_net_model_converted.pth",
    map_location="cpu",
    weights_only=False
)

print("Input features:", checkpoint["input_features"])
print("Model weight shapes:")

for key, value in checkpoint["model_state"].items():
    print(key, value.shape)