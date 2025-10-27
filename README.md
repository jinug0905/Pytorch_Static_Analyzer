# PyTorch Static Analyzer

A static memory analysis tool for PyTorch **TorchScript** models — designed for deployment scenarios on resource‑constrained systems. It estimates the peak **GPU memory usage for inference** without executing the model.

---

## Project Information

**Author:** Jinug Lee  
**Environment:** Texas A&M Grace HPC Cluster  
**Last Updated:** October 2025  

---

## 1. Overview

This build system compiles a **C++ static analyzer** that reads and analyzes TorchScript models  
(e.g., `resnet50_scripted.pt`) using PyTorch’s internal **JIT IR (Intermediate Representation)**.

Instead of installing PyTorch manually, the project uses Grace HPC’s module‑based system:

- PyTorch 2.7.0‑foss‑2023b (CUDA‑enabled)
- CUDA 12.9.0
- GCCcore 13.2.0

---

## 2. Folder Structure

```plaintext
staticAnalyzer/
│
├── 1_ExportingIR/                 # Python: exports TorchScript
│   ├── export_model.py
│   └── tensorirVenv/
│
├── 2_EstimateMem/                 # C++: performs static analysis
│   ├── CMakeLists.txt
│   ├── main.cpp
│   └── build/
│
└── data/                          # TorchScript model & IR artifacts
    ├── resnet50_scripted.pt
    ├── model_ir.txt
    └── model_code.txt
```

---

## 3. Build System Design Goals

The CMake‑based build system:

1. Automatically detects PyTorch & CUDA paths using EasyBuild modules  
2. Eliminates the need for user‑supplied `-D` configuration flags  
3. Creates missing `CUDA::nvToolsExt` target when using CUDA 12.9  
4. Embeds **RPATH** for automatic shared‑library discovery  
5. Embeds `DATA_DIR` in the binary to locate the `.pt` model  
6. Ensures reproducibility across Grace login nodes  

---

## 4. Environment Variables on Grace HPC

The build system uses:

- `EBROOTPYTORCH` → Torch libraries + TorchConfig.cmake  
- `EBROOTCUDA` → CUDA toolkit + NVTX3 headers  

---

## 5. CMake Behavior — Step‑by‑Step

- Validate environment modules  
- Locate Torch CMake config in site‑packages  
- Patch NVTX linkage  
- Import Torch targets (`torch`, `c10`, etc.)  
- Define `DATA_DIR`  
- Embed Torch library paths in RPATH  

---

## 6. Usage (Grace HPC)

```bash
# Load environment
module purge
module load PyTorch/2.7.0-foss-2023b CUDA/12.9.0

# Build
cd 2_EstimateMem
rm -rf build && mkdir build && cd build
cmake ..
cmake --build . -j

# Run analyzer
./torch_ir_mem
```

---

## 7. Results — Static vs Runtime Memory

Batch Size = 1

| Model     | Params (MB) | Peak Activ. (MB) | Total Est. (MB) | Runtime Peak GPU (MB) |
|-----------|-------------|-----------------|-----------------|----------------------|
| ResNet‑50 | 97.70       | 97.68           | ~195.38         | 218.29               |
| ResNet‑101| 170.34      | 170.33          | ~340.67         | 363.58               |
| VGG‑16    | 527.79      | 527.79          | ~1055.58        | 1104.75              |

✅ Static results closely match runtime measurement  
(discrepancy → cuDNN workspace + caching overhead)

### Scaling Example — Runtime Reference Only

| Model     | bs=1 | bs=2 | bs=3 |
|-----------|------|------|------|
| ResNet‑50 | 218.29 | 226.99 | 237.28 |
| ResNet‑101| 363.58 | 372.94 | 382.12 |
| VGG‑16    | 1104.75| 1142.30| 1179.67|

=> Memory scales **almost linearly** → static extension for batch size >1 is trivial.

---

## 8. Advisor Key Notes

- Reproducible build w/ HPC module ecosystem  
- Static GPU inference memory estimator (no runtime needed)  
- Uses TorchScript freeze + shape propagation + IR liveness analysis  
- Strong accuracy correlation with profiler results  

---

## 9. Troubleshooting

| Issue | Solution |
|------|----------|
| CMake finds wrong libs | Ensure modules loaded before build |
| Runtime linker failure | Clear build/ → rebuild |
| Missing IR inlining | Re‑export TorchScript w/ `freeze()` |

---

## 10. Future Work

- Batch‑size parameterization beyond bs=1  
- Memory timeline visualization  
- Multiple‑model benchmarking automation  

---

## Summary

This project bridges **deep learning deployment** and **system‑level planning** by providing a:

>  Fast and accurate static peak GPU memory estimation  
>  Deployable on HPC environments without model execution  

The build process is **fully automated** and **Grace HPC‑friendly** ✅

---
