cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(torch_ir_mem CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------
# Expect you loaded:
#   module load PyTorch/2.7.0-foss-2023b
#   module load CUDA/12.9.0
# ----------------------------
if(NOT DEFINED ENV{EBROOTPYTORCH})
  message(FATAL_ERROR "EBROOTPYTORCH not set. Load the PyTorch module first.")
endif()
if(NOT DEFINED ENV{EBROOTCUDA})
  message(FATAL_ERROR "EBROOTCUDA not set. Load the CUDA module first.")
endif()
set(EBROOTPYTORCH "$ENV{EBROOTPYTORCH}")
set(EBROOTCUDA    "$ENV{EBROOTCUDA}")
message(STATUS "EBROOTPYTORCH = ${EBROOTPYTORCH}")
message(STATUS "EBROOTCUDA    = ${EBROOTCUDA}")

# Python ABI subdir under site-packages (override if needed)
if(NOT DEFINED PYTHON_ABI_SUBDIR)
  set(PYTHON_ABI_SUBDIR "python3.11")
endif()
message(STATUS "PYTHON_ABI_SUBDIR = ${PYTHON_ABI_SUBDIR}")

# ----------------------------
# Torch_DIR from EBROOTPYTORCH
# ----------------------------
set(_candidate_torch_dir
  "${EBROOTPYTORCH}/lib/${PYTHON_ABI_SUBDIR}/site-packages/torch/share/cmake/Torch")
if(EXISTS "${_candidate_torch_dir}/TorchConfig.cmake")
  set(Torch_DIR "${_candidate_torch_dir}")
  message(STATUS "Using Torch_DIR (EB exact) = ${Torch_DIR}")
else()
  file(GLOB _torch_dir_glob
       "${EBROOTPYTORCH}/lib/python3.*/site-packages/torch/share/cmake/Torch")
  list(LENGTH _torch_dir_glob _n)
  if(_n EQUAL 0)
    message(FATAL_ERROR "TorchConfig.cmake not found under EBROOTPYTORCH.")
  endif()
  list(GET _torch_dir_glob 0 Torch_DIR)
  message(STATUS "Using Torch_DIR (glob) = ${Torch_DIR}")
endif()

# ----------------------------
# CUDA hints + NVTX handling
# ----------------------------
# Hint CUDAToolkit location (helps find CUDA compilers/headers if needed)
set(CUDAToolkit_ROOT "${EBROOTCUDA}")

# Try to find legacy nvToolsExt library; if missing, fall back to nvtx3 headers
find_library(NVTOOLSEXT_LIB
  NAMES nvToolsExt
  PATHS "${EBROOTCUDA}/lib64" "${EBROOTCUDA}/lib"
  NO_DEFAULT_PATH)
find_path(NVTOOLSEXT_INC_LEGACY
  NAMES nvToolsExt.h
  PATHS "${EBROOTCUDA}/include"
  NO_DEFAULT_PATH)

# nvtx3 header-only (modern CUDA)
find_path(NVTX3_INC
  NAMES nvtx3/nvtx3.hpp
  PATHS "${EBROOTCUDA}/include"
  NO_DEFAULT_PATH)

# Define CUDA::nvToolsExt target one way or the other
if(NVTOOLSEXT_LIB AND NVTOOLSEXT_INC_LEGACY)
  message(STATUS "Found legacy NVTX: ${NVTOOLSEXT_LIB}")
  add_library(CUDA::nvToolsExt SHARED IMPORTED)
  set_target_properties(CUDA::nvToolsExt PROPERTIES
    IMPORTED_LOCATION "${NVTOOLSEXT_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${NVTOOLSEXT_INC_LEGACY}")
elseif(NVTX3_INC)
  message(STATUS "Using NVTX3 headers at: ${NVTX3_INC}")
  # Header-only shim target to satisfy Torch's link interface expectation
  add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
  set_target_properties(CUDA::nvToolsExt PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${NVTX3_INC}")
else()
  message(FATAL_ERROR
    "Neither legacy nvToolsExt nor nvtx3 headers were found under ${EBROOTCUDA}.")
endif()

# ----------------------------
# Find Torch after NVTX target is defined
# ----------------------------
find_package(Torch REQUIRED)  # uses Torch_DIR set above
message(STATUS "Torch_DIR       = ${Torch_DIR}")
message(STATUS "TORCH_LIBRARIES = ${TORCH_LIBRARIES}")

# ----------------------------
# Torch lib dir for RPATH
# ----------------------------
set(_torch_lib_from_env
  "${EBROOTPYTORCH}/lib/${PYTHON_ABI_SUBDIR}/site-packages/torch/lib")
if(EXISTS "${_torch_lib_from_env}")
  set(_torch_lib "${_torch_lib_from_env}")
else()
  get_filename_component(_share "${Torch_DIR}" DIRECTORY)
  get_filename_component(_root  "${_share}" DIRECTORY)
  set(_torch_lib "${_root}/lib")
endif()
message(STATUS "Torch lib dir = ${_torch_lib}")

# ----------------------------
# Data path baked into binary
# ----------------------------
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
set(DATA_DIR "${REPO_ROOT}/data")
message(STATUS "DATA_DIR = ${DATA_DIR}")

# ----------------------------
# Target + link + RPATH
# ----------------------------
add_executable(torch_ir_mem main.cpp)
target_link_libraries(torch_ir_mem
  "${TORCH_LIBRARIES}"
  CUDA::nvToolsExt)  # ensure the target exists for Torch's interface
target_compile_definitions(torch_ir_mem PRIVATE DATA_DIR="${DATA_DIR}")

set_target_properties(torch_ir_mem PROPERTIES
  BUILD_RPATH "${_torch_lib}"
  INSTALL_RPATH "${_torch_lib}"
  INSTALL_RPATH_USE_LINK_PATH TRUE
)
