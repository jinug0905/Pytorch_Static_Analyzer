# Pytorch_Static_Analyzer
This project implements a static memory analysis tool for PyTorch TorchScript models. Specifically designed for deployment scenarios on resource-constrained systems, it statically estimates the peak GPU memory required for inference without running the model.

PYTORCH STATIC ANALYZER — BUILD SYSTEM DOCUMENTATION

Author: Jinug Lee
Environment: Texas A&M Grace HPC Cluster
Last Updated: October 2025

1. OVERVIEW
-----------
This document explains how and why the CMakeLists.txt file in this project was written.

The goal of the build system is to compile a C++ static analyzer that reads and analyzes
a TorchScript model (e.g., resnet50_scripted.pt) to estimate its memory usage using
PyTorch’s internal JIT IR (Intermediate Representation).

Grace HPC uses a module-based software system, so instead of installing PyTorch manually,
the build configuration leverages existing cluster modules for:

• PyTorch 2.7.0-foss-2023b (CUDA-enabled)
• CUDA 12.9.0
• GCCcore 13.2.0

2. FOLDER STRUCTURE
-------------------
staticAnalyzer/
│
├── 1_ExportingIR/                 (Python: exports TorchScript)
│   ├── export_model.py
│   └── tensorirVenv/
│
├── 2_EstimateMem/                 (C++: performs static analysis)
│   ├── CMakeLists.txt
│   ├── main.cpp
│   └── build/
│
└── data/                          (TorchScript model & IR artifacts)
    ├── resnet50_scripted.pt
    ├── model_ir.txt
    └── model_code.txt

3. DESIGN GOALS OF THE BUILD SYSTEM
-----------------------------------
The CMakeLists.txt was designed to:
1. Automatically detect PyTorch and CUDA paths via EasyBuild environment modules.
2. Remove the need for user-supplied -D flags.
3. Create missing CUDA::nvToolsExt target when using CUDA 12.9 (NVTX header-only).
4. Embed RPATH for runtime shared library discovery.
5. Bake DATA_DIR into the binary for locating .pt models.
6. Ensure reproducibility on any login node that loads the same modules.

4. ENVIRONMENT VARIABLES ON GRACE HPC
-------------------------------------
(EBROOTPYTORCH, EBROOTCUDA) are used to locate:
• TorchConfig.cmake
• Torch shared libraries
• CUDA toolkit and NVTX3 headers

5. HOW CMAKE WORKS (STEP-BY-STEP)
---------------------------------
• Validate environment modules
• Locate PyTorch CMake configuration under site-packages
• Patch NVTX
• Import Torch targets (torch, c10, etc.)
• Set DATA_DIR definition in compiler
• Embed Torch library path in RPATH

6. USAGE ON GRACE HPC
---------------------
(1) Load modules
    module purge
    module load PyTorch/2.7.0-foss-2023b CUDA/12.9.0

(2) Build
    cd 2_EstimateMem
    rm -rf build && mkdir build && cd build
    cmake ..
    cmake --build . -j

(3) Run analyzer
    ./torch_ir_mem

7. RESULTS — STATIC ESTIMATION vs RUNTIME PROFILING
---------------------------------------------------
"Before" = model weights + runtime contexts,
"Peak" = max live activations during the forward pass.

Batch Size = 1
-----------------------------------------------
Model       | Params (MB) | Peak Activ. (MB) | Total Est. (MB) | Runtime Peak GPU (MB)
------------|-------------|-----------------|-----------------|---------------------
ResNet-50   | 97.70       | 97.68           | ~195.38         | 218.29
ResNet-101  | 170.34      | 170.33          | ~340.67         | 363.58
VGG-16      | 527.79      | 527.79          | ~1055.58        | 1104.75

✅ Observation:
Static estimates closely match runtime GPU utilization, differing mainly by cuDNN
workspace allocations and caching overhead.

Scaling with Batch Size (Runtime — Reference Only)
---------------------------------------------------
(bs=1 → bs=3 example)

Model       | bs=1 Peak (MB) | bs=2 Peak (MB) | bs=3 Peak (MB)
------------|----------------|----------------|----------------
ResNet-50   | 218.29         | 226.99         | 237.28
ResNet-101  | 363.58         | 372.94         | 382.12
VGG-16      | 1104.75        | 1142.30        | 1179.67

✅ Memory increases nearly linearly with batch size → static prediction for other
batch sizes is mathematically straightforward.

8. KEY POINTS FOR THE ADVISOR
-----------------------------
• Automated build compatible with HPC module ecosystem.
• C++ TorchScript-based static inference memory estimator.
• Uses freeze, shape propagation, and liveness analysis in JIT IR.
• Memory estimates align strongly with real GPU profiler measurements.

9. TROUBLESHOOTING
------------------
• Ensure modules are loaded before building.
• Clear build/ folder if switching modules or compiler versions.
• If IR does not inline submodules: re-export TorchScript with freeze() applied.

10. FUTURE IMPROVEMENTS
-----------------------
• Full batch-size parameterization (predict N > 1 statically).
• Visual memory timeline generation.
• Multi-model benchmarking automation.

SUMMARY
-------
This project bridges PyTorch model deployment and systems resource planning by
providing reliable static peak memory estimation without executing the model.
The build system is fully HPC-aware and reproducible on Texas A&M Grace infrastructure.
