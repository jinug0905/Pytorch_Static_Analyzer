import os
import torch

# -=-=-=-=-=- Config -=-=-=-=-=-=-
MODEL_NAME = 'resnet50'
# MODEL_NAME = 'resnet101'
# MODEL_NAME = 'vgg16'

BATCH_SIZE = 1

TS_DIR = '../data'
TS_FILE = f"{MODEL_NAME}b{BATCH_SIZE}.pt"
TS_PATH = os.path.join(TS_DIR, TS_FILE)

# -=-=-=-=-=- Device setup -=-=-=-=-=-
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
is_cuda = device.type == 'cuda'

def bytes_to_mb(x): return x / (1024 ** 2)

model = torch.hub.load('pytorch/vision:v0.10.0', MODEL_NAME, pretrained=True).to(device).eval()
example_input = torch.randn(1, 3, 224, 224, device=device)

with torch.no_grad():
    scripted = torch.jit.trace(model, example_input)
    
scripted.save(TS_PATH)

model = torch.jit.load(TS_PATH, map_location=device).eval()

x = torch.randn(BATCH_SIZE, 3, 224, 224, device=device)

# Reset memory stats
if is_cuda:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

before = torch.cuda.memory_allocated(device) if is_cuda else 0

# Inference
with torch.no_grad():
    y = model(x)
    if is_cuda: torch.cuda.synchronize()

after = torch.cuda.memory_allocated(device) if is_cuda else 0
peak = torch.cuda.max_memory_allocated(device) if is_cuda else 0

# Weight memory (model resident size)
param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
weight_mb = bytes_to_mb(param_bytes + buffer_bytes)

# TorchScript file size
ts_size_mb = bytes_to_mb(os.path.getsize(TS_PATH))

print(f"=== Inference Memory Report {MODEL_NAME}, batch size : {BATCH_SIZE} ===")
print(f"Device: {device}")
print(f"Weights memory:        {weight_mb:.5f} MB")
print(f"TorchScript file:      {ts_size_mb:.5f} MB")

print(f"Allocated before:      {bytes_to_mb(before):.5f} MB")
print(f"Allocated after:       {bytes_to_mb(after):.5f} MB")
print(f"Peak during inference: {bytes_to_mb(peak):.5f} MB")