PYTORCH STATIC ANALYZER — BUILD SYSTEM DOCUMENTATION

Author: Jinug Lee
Environment: Texas A&M Grace HPC Cluster
Last Updated: October 2025

1.	OVERVIEW

This document explains how and why the CMakeLists.txt file in this project was written.

The goal of the build system is to compile a C++ static analyzer that reads and analyzes a TorchScript model (saved as mlp_scripted.pt) to estimate its memory usage using PyTorch’s internal JIT IR (Intermediate Representation).

Grace HPC uses a module-based software system, so instead of installing PyTorch manually, the build configuration leverages the existing cluster modules for:

• PyTorch 2.7.0-foss-2023b
• CUDA 12.9.0
• GCCcore 13.2.0

2.	FOLDER STRUCTURE

staticAnalyzer/
│
├── 1_ExportingIR/                 (Python: exports TorchScript IR)
│   ├── 1_exportIr.py
│   └── tensorirVenv/
│
├── 2_EstimateMem/                 (C++: performs static analysis)
│   ├── CMakeLists.txt
│   ├── main.cpp
│   └── build/                     (created after cmake build)
│
└── data/                          (shared artifacts between Python and C++)
├── mlp_scripted.pt            (TorchScript model, binary)
├── mlp_ir.txt                 (textual IR representation)
└── mlp_code.txt               (readable TorchScript code)

3.	DESIGN GOALS OF THE BUILD SYSTEM

The CMakeLists.txt was designed to:
	1.	Automatically detect correct PyTorch and CUDA paths from Grace module environment variables (EBROOTPYTORCH and EBROOTCUDA).
	2.	Avoid the need for any manual -D flags when running cmake ..
	3.	Handle PyTorch’s CUDA linkage by defining missing CUDA::nvToolsExt targets if necessary.
	4.	Embed RPATHs in the compiled binary so it can locate shared libraries without LD_LIBRARY_PATH.
	5.	Bake the project’s data/ folder path into the binary as DATA_DIR.
	6.	Be reproducible and self-documenting — anyone loading the same modules can build successfully.
	7.	ENVIRONMENT VARIABLES ON GRACE

⸻

Grace uses EasyBuild, which defines the following variables when modules are loaded:

Variable         : EBROOTPYTORCH
Example Value    : /sw/eb/sw/PyTorch/2.7.0-foss-2023b
Purpose          : Root path of the installed PyTorch package

Variable         : EBROOTCUDA
Example Value    : /sw/eb/sw/CUDA/12.9.0
Purpose          : Root path of the installed CUDA toolkit

From these, the build system constructs internal paths automatically:

Component        : TorchConfig.cmake
Path Template    : $EBROOTPYTORCH/lib/python3.11/site-packages/torch/share/cmake/Torch

Component        : Torch Shared Libraries
Path Template    : $EBROOTPYTORCH/lib/python3.11/site-packages/torch/lib

Component        : CUDA Toolkit
Path Template    : $EBROOTCUDA

Component        : NVTX (header-only in CUDA 12.9)
Path Template    : $EBROOTCUDA/include/nvtx3/

5.	HOW CMAKE WORKS (STEP BY STEP)

STEP 1 – Check Environment
CMake verifies that EBROOTPYTORCH and EBROOTCUDA are defined.
If not, it stops and prints an error asking to load the modules:
module load GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.7.0 CUDA/12.9.0

STEP 2 – Locate TorchConfig.cmake
The configuration file lives inside the Python site-packages tree:
$EBROOTPYTORCH/lib/python3.11/site-packages/torch/share/cmake/Torch
If that directory does not exist (different Python version), the script
performs a fallback search under python3.* automatically.

STEP 3 – Handle CUDA and NVTX
PyTorch 2.7 is built with CUDA support and expects a target called CUDA::nvToolsExt.
CUDA 12.9 no longer ships libnvToolsExt.so (legacy NVTX2) and instead provides
header-only nvtx3 headers.
The CMakeLists.txt therefore checks:
- If legacy library exists → create an IMPORTED SHARED target.
- Otherwise → create an INTERFACE target pointing at the nvtx3 header directory:
add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
set_target_properties(CUDA::nvToolsExt PROPERTIES
INTERFACE_INCLUDE_DIRECTORIES “${EBROOTCUDA}/include/nvtx3”)

STEP 4 – Find Torch
After the CUDA/NVTX targets exist, CMake calls:
find_package(Torch REQUIRED)
This imports all needed Torch C++ libraries such as torch, torch_cpu, c10, and others.

STEP 5 – Bake the Data Path
The path to ../data is computed and compiled into the binary as DATA_DIR:
target_compile_definitions(torch_ir_mem PRIVATE DATA_DIR=”${DATA_DIR}”)
This allows main.cpp to access the model directly:
torch::jit::load(std::string(DATA_DIR) + “/mlp_scripted.pt”);

STEP 6 – Embed RPATH
To eliminate the need for LD_LIBRARY_PATH, CMake embeds the search path
to the Torch shared libraries inside the executable itself:
set_target_properties(torch_ir_mem PROPERTIES
BUILD_RPATH “${_torch_lib}”
INSTALL_RPATH “${_torch_lib}”
INSTALL_RPATH_USE_LINK_PATH TRUE)


6.	USAGE ON GRACE HPC

(1) Load required modules
module purge
module load PyTorch/2.7.0-foss-2023b
module load CUDA/12.9.0

(2) Configure and build
cd 2_EstimateMem
rm -rf build && mkdir build && cd build
cmake ..
cmake –build . -j

(3) Run analyzer
./torch_ir_mem

Example output:
Parameter bytes:        10.79 KB
Peak activations:       2.00 KB
Total peak (inference): 12.79 KB


7.	KEY POINTS FOR THE ADVISOR

• The build system is self-contained; no manual paths or flags are needed.
• Fully compatible with Grace’s EasyBuild modules.
• Handles CUDA 12.9’s nvtx3 header-only design automatically.
• Embeds data and library paths to ensure reproducible runs.
• Serves as a reusable CMake template for future TorchScript-based analysis.

8.	TROUBLESHOOTING

Problem: TorchConfig.cmake not found
Cause: PyTorch module not loaded
Fix: module load PyTorch/2.7.0-foss-2023b

Problem: CUDA::nvToolsExt not found
Cause: Missing NVTX library
Fix: Handled automatically via header-only target

Problem: Runtime cannot find libtorch.so
Cause: RPATH not embedded
Fix: Delete build/ and rerun cmake ..

Problem: Compilation fails on another PyTorch version
Cause: API mismatch
Fix: Update include headers for that version’s torch::jit API.

9.	FUTURE IMPROVEMENTS

• Add CPU-only configuration for systems without CUDA.
• Convert the analyzer into a standalone CMake subproject for reuse.
• Dynamically detect Python version instead of assuming python3.11.

SUMMARY

This build system connects the Python-exported TorchScript model (mlp_scripted.pt)
with the C++ static analyzer on Grace HPC. It automates all environment discovery
and library linking for PyTorch and CUDA, ensuring reproducibility and ease of use.
Anyone loading the same modules can build and run the analyzer without modifying
any paths or CMake options.

Pytorch version on grace was based on CUDA so CPU version was not available. => So had to CUDA version