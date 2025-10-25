import os
from pathlib import Path
import torch
import torch.nn as nn

# 1. simple MLP 
class SimpleMLP(nn.Module):
    def __init__(self, in_dim=32, hidden=64, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def main():
    batch_size = 8
    in_dim = 32
    hidden = 64
    out_dim = 10
    dtype = torch.float32

    # Instantiate model with random example input
    model = SimpleMLP(in_dim, hidden, out_dim).eval()
    example_input = torch.randn(batch_size, in_dim, dtype=dtype)

    # 2. Export to TorchScript which translate into Tensor IR
    # torch.jit.trace -> execute model with sample input and record the sequence of operations
    #                        (can't capture control flow) but good for simple one.
    # torch.jit.script -> pasrses python code, analyze AST, and convert to TorchScript
    with torch.no_grad():
        scripted = torch.jit.trace(model, example_input)

    # 3. Save the scripted model
    ts_path = "../data/mlp_scripted.pt"
    scripted.save(str(ts_path))

    # 4. Save Tensor IR in text
    ir_text = str(scripted.inlined_graph)
    ir_path = "../data/mlp_ir.txt"
    with open(ir_path, "w") as f:
        f.write(ir_text)

    # code save just in case
    # code_path = "../data/mlp_code.txt"
    # with open(code_path, "w") as f:
    #     f.write(scripted.code)

    print(f"Saved TorchScript model to: {ts_path}")
    print(f"Saved Tensor IR (inlined graph) to: {ir_path}")

if __name__ == "__main__":
    main()