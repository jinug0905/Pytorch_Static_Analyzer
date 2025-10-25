# module load GCCcore/12.3.0 Python/3.11.3
# module load GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.7.0

import torch, pprint
m = torch.jit.load("mlp_scripted.pt")

# High-level summary
print()
print("-=-=-=-=-= Start of module Tree -=-=-=-=-=-=-=-")
print()
print(m)                     # module tree
print()
print("-=-=-=-=-= End of module Tree -=-=-=-=-=-=-=-")
print()


print("-=-=-=-=-= Start of module TorchScript for methods -=-=-=-=-=-=-=-")
print()
print(m.code)                # pretty-printed TorchScript for methods
print()
print("-=-=-=-=-= End of module TorchScript for methods -=-=-=-=-=-=-=-")
print()

print("-=-=-=-=-= Start of module inlined graph -=-=-=-=-=-=-=-")
print(" This is exactly the same as mlp_ir.txt file. Also, I used for c++ file to analyze")
print()
print(m.inlined_graph)       # single, flattened graph (ops like aten::linear)
print()
print("-=-=-=-=-= End of module inlined graph -=-=-=-=-=-=-=-")
print()
print()

# Parameters/buffers
for n,p in m.named_parameters():
    print("PARAM", n, p.shape, p.dtype)
for n,b in m.named_buffers():
    print("BUFFER", n, b.shape, b.dtype)