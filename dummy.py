import os
import torch
import torchvision
import torchvision.datasets as datasets

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Import + instantiate a model (random weights, no download)
model = torchvision.models.resnet18(weights=None)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {model.__class__.__name__} ({n_params:,} parameters)")

# Load CIFAR-10 from the copy staged into local scratch by job.sh
data_root = os.environ["SLURM_TMPDIR"]
ds = datasets.CIFAR10(root=data_root, train=True, download=False)
print(f"Dataset: CIFAR-10 with {len(ds)} training samples")
print(f"First sample label index: {ds[0][1]}")
