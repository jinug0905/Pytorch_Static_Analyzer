# cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
# project(torch_ir_mem STATIC)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_PREFIX_PATH "/path/to/libtorch") # Set this to your LibTorch root

# find_package(Torch REQUIRED)

# add_executable(torch_ir_mem main.cpp)
# target_link_libraries(torch_ir_mem "${TORCH_LIBRARIES}")
# set_property(TARGET torch_ir_mem PROPERTY CXX_STANDARD 17)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
# project(torch_ir_mem CXX)
# set(CMAKE_CXX_STANDARD 17)

# # >>> Set your LibTorch root here (or pass -DCMAKE_PREFIX_PATH=... to cmake)
# set(CMAKE_PREFIX_PATH "$ENV{EBROOTPYTORCH}")

# find_package(Torch REQUIRED)

# # Compute repo root = parent of this CMakeLists.txt directory
# get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
# set(DATA_DIR "${REPO_ROOT}/data")

# add_executable(torch_ir_mem main.cpp)
# target_link_libraries(torch_ir_mem "${TORCH_LIBRARIES}")

# # Bake the absolute data path into the binary as DATA_DIR
# target_compile_definitions(torch_ir_mem PRIVATE DATA_DIR="${DATA_DIR}")

# message(STATUS "CMAKE_PREFIX_PATH = ${CMAKE_PREFIX_PATH}")
# message(STATUS "Torch_FOUND = ${Torch_FOUND}")
# message(STATUS "Torch_DIR = ${Torch_DIR}")
# message(STATUS "DATA_DIR = ${DATA_DIR}")

# -=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-
# cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
# project(torch_ir_mem CXX)
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # ----------------------------
# # 1) Locate TorchConfig.cmake
# # ----------------------------
# # Priority:
# #  A) User-provided Torch_DIR (-DTorch_DIR=/path/to/.../cmake/Torch)
# #  B) ENV{Torch_DIR}
# #  C) EasyBuild envs: EBROOTLIBTORCH / EBROOTPYTORCH
# #  D) Query active Python: import torch; print(path/to/share/cmake/Torch)

# # Helper: check if a candidate contains TorchConfig.cmake
# function(_torch_try_dir cand outvar)
#   if(EXISTS "${cand}/TorchConfig.cmake")
#     set(${outvar} "${cand}" PARENT_SCOPE)
#   endif()
# endfunction()

# # A) CLI-provided Torch_DIR wins automatically (CMake will use it)
# if(DEFINED Torch_DIR)
#   message(STATUS "Using Torch_DIR from CLI: ${Torch_DIR}")
# else()
#   # B) ENV{Torch_DIR}
#   if(DEFINED ENV{Torch_DIR})
#     _torch_try_dir("$ENV{Torch_DIR}" _found_torch_dir)
#     if(_found_torch_dir)
#       set(Torch_DIR "${_found_torch_dir}")
#       message(STATUS "Using Torch_DIR from environment: ${Torch_DIR}")
#     endif()
#   endif()

#   # C) EasyBuild roots
#   if(NOT Torch_DIR)
#     foreach(rootvar EBROOTLIBTORCH EBROOTPYTORCH)
#       if(DEFINED ENV{${rootvar}})
#         set(_root "$ENV{${rootvar}}")
#         foreach(sub "share/cmake/Torch" "lib/cmake/Torch")
#           _torch_try_dir("${_root}/${sub}" _found_torch_dir)
#           if(_found_torch_dir)
#             set(Torch_DIR "${_found_torch_dir}")
#             message(STATUS "Using Torch_DIR from ${rootvar}: ${Torch_DIR}")
#             break()
#           endif()
#         endforeach()
#       endif()
#       if(Torch_DIR)
#         break()
#       endif()
#     endforeach()
#   endif()

#   # D) Ask Python
#   if(NOT Torch_DIR)
#     find_program(PYTHON_EXECUTABLE NAMES python3 python)
#     if(PYTHON_EXECUTABLE)
#       execute_process(
#         COMMAND "${PYTHON_EXECUTABLE}" -c
#                 "import torch, pathlib; p=pathlib.Path(torch.__file__).parent; print(p/'share/cmake/Torch')"
#         OUTPUT_VARIABLE _py_torch_cmake
#         OUTPUT_STRIP_TRAILING_WHITESPACE
#         ERROR_QUIET
#       )
#       if(_py_torch_cmake)
#         _torch_try_dir("${_py_torch_cmake}" _found_torch_dir)
#         if(_found_torch_dir)
#           set(Torch_DIR "${_found_torch_dir}")
#           message(STATUS "Using Torch_DIR discovered via Python: ${Torch_DIR}")
#         endif()
#       endif()
#     endif()
#   endif()
# endif()

# if(NOT Torch_DIR)
#   message(FATAL_ERROR
#     "Could not locate TorchConfig.cmake.\n"
#     "Try one of:\n"
#     "  - Load your PyTorch/LibTorch module, then re-run cmake\n"
#     "  - Pass -DTorch_DIR=/sw/eb/.../site-packages/torch/share/cmake/Torch\n"
#     "  - Or set ENV Torch_DIR to that path"
#   )
# endif()

# # ----------------------------
# # 2) Find Torch and show info
# # ----------------------------
# find_package(Torch REQUIRED)   # Uses Torch_DIR above

# message(STATUS "Torch_DIR          = ${Torch_DIR}")
# message(STATUS "Torch version hint = ${TORCH_VERSION}")
# message(STATUS "Torch libraries    = ${TORCH_LIBRARIES}")

# # ----------------------------
# # 3) Data path baked into binary
# # ----------------------------
# # repo root = parent of this folder (2_EstimateMem -> repo/)
# get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
# set(DATA_DIR "${REPO_ROOT}/data")

# # ----------------------------
# # 4) Build target
# # ----------------------------
# add_executable(torch_ir_mem main.cpp)
# target_link_libraries(torch_ir_mem "${TORCH_LIBRARIES}")
# target_compile_definitions(torch_ir_mem PRIVATE DATA_DIR="${DATA_DIR}")

# # ----------------------------
# # 5) RPATH so runtime finds libtorch*.so without LD_LIBRARY_PATH
# #    Torch_DIR = .../site-packages/torch/share/cmake/Torch
# #    Torch libs are usually at .../site-packages/torch/lib
# # ----------------------------
# get_filename_component(_torch_share "${Torch_DIR}" DIRECTORY)     # .../share/cmake
# get_filename_component(_torch_root  "${_torch_share}" DIRECTORY)  # .../site-packages/torch
# set(_torch_lib "${_torch_root}/lib")

# # Only add if it exists (it should, but be graceful)
# if(EXISTS "${_torch_lib}")
#   message(STATUS "Embedding RPATH to torch lib dir: ${_torch_lib}")
#   set_target_properties(torch_ir_mem PROPERTIES
#     BUILD_RPATH "${_torch_lib}"
#     INSTALL_RPATH "${_torch_lib}"
#     INSTALL_RPATH_USE_LINK_PATH TRUE
#   )
# else()
#   message(WARNING "Torch lib dir not found at ${_torch_lib}. "
#                   "You may need to set LD_LIBRARY_PATH at runtime.")
# endif()

# # Optional: If you ever need to force the C++11 ABI (usually not necessary)
# # add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

#============================================================
# cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
# project(torch_ir_mem CXX)
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # ----------------------------
# # 0) Require EBROOTPYTORCH
# # ----------------------------
# if(NOT DEFINED ENV{EBROOTPYTORCH})
#   message(FATAL_ERROR
#     "EBROOTPYTORCH is not set. Load your PyTorch module first, e.g.:\n"
#     "  module load PyTorch/2.7.0-foss-2023b\n"
#     "Then rerun cmake.")
# endif()
# set(EBROOTPYTORCH "$ENV{EBROOTPYTORCH}")
# message(STATUS "EBROOTPYTORCH = ${EBROOTPYTORCH}")

# # ----------------------------
# # 1) Build Torch_DIR from EBROOTPYTORCH
# #    Default to python3.11, allow override via -DPYTHON_ABI_SUBDIR=python3.11
# #    Fallback: glob python3.* if the exact subdir doesn't exist.
# # ----------------------------
# if(NOT DEFINED PYTHON_ABI_SUBDIR)
#   set(PYTHON_ABI_SUBDIR "python3.11")
# endif()
# message(STATUS "PYTHON_ABI_SUBDIR (override with -DPYTHON_ABI_SUBDIR=...) = ${PYTHON_ABI_SUBDIR}")

# set(_candidate_torch_dir
#     "${EBROOTPYTORCH}/lib/${PYTHON_ABI_SUBDIR}/site-packages/torch/share/cmake/Torch")

# if(EXISTS "${_candidate_torch_dir}/TorchConfig.cmake")
#   set(Torch_DIR "${_candidate_torch_dir}")
#   message(STATUS "Using Torch_DIR (from EBROOTPYTORCH exact path): ${Torch_DIR}")
# else()
#   # Fallback: try to find a python3.* that contains TorchConfig.cmake
#   file(GLOB _torch_dir_glob
#        "${EBROOTPYTORCH}/lib/python3.*/site-packages/torch/share/cmake/Torch")
#   set(_found "")
#   foreach(p ${_torch_dir_glob})
#     if(EXISTS "${p}/TorchConfig.cmake")
#       set(_found "${p}")
#       break()
#     endif()
#   endforeach()

#   if(_found)
#     set(Torch_DIR "${_found}")
#     message(STATUS "Using Torch_DIR (glob fallback): ${Torch_DIR}")
#   else()
#     message(FATAL_ERROR
#       "Could not locate TorchConfig.cmake under EBROOTPYTORCH.\n"
#       "Tried:\n  ${_candidate_torch_dir}\n"
#       "and glob:\n  ${EBROOTPYTORCH}/lib/python3.*/site-packages/torch/share/cmake/Torch\n"
#       "Check your PyTorch module, or override with -DPYTHON_ABI_SUBDIR=pythonX.Y")
#   endif()
# endif()

# # ----------------------------
# # 2) Find Torch using the resolved Torch_DIR
# # ----------------------------
# find_package(Torch REQUIRED)
# message(STATUS "Torch_DIR          = ${Torch_DIR}")
# message(STATUS "TORCH_VERSION hint = ${TORCH_VERSION}")
# message(STATUS "TORCH_LIBRARIES    = ${TORCH_LIBRARIES}")

# # ----------------------------
# # 3) Compute torch lib dir from EBROOTPYTORCH (preferred),
# #    fallback to deriving from Torch_DIR.
# # ----------------------------
# set(_torch_lib_from_env
#     "${EBROOTPYTORCH}/lib/${PYTHON_ABI_SUBDIR}/site-packages/torch/lib")
# if(EXISTS "${_torch_lib_from_env}")
#   set(_torch_lib "${_torch_lib_from_env}")
#   message(STATUS "Torch lib dir (from EBROOTPYTORCH) = ${_torch_lib}")
# else()
#   # Derive: Torch_DIR = .../site-packages/torch/share/cmake/Torch
#   get_filename_component(_torch_share "${Torch_DIR}" DIRECTORY)    # .../share/cmake
#   get_filename_component(_torch_root  "${_torch_share}" DIRECTORY) # .../site-packages/torch
#   set(_torch_lib "${_torch_root}/lib")
#   message(WARNING
#     "Torch lib dir not found at ${_torch_lib_from_env}; "
#     "falling back to ${_torch_lib}")
# endif()

# if(NOT EXISTS "${_torch_lib}")
#   message(WARNING
#     "Torch lib directory ${_torch_lib} does not exist. "
#     "You may need to set LD_LIBRARY_PATH at runtime to the actual torch/lib.")
# endif()

# # ----------------------------
# # 4) Data path baked into binary (../data from this directory)
# # ----------------------------
# get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
# set(DATA_DIR "${REPO_ROOT}/data")
# message(STATUS "DATA_DIR (baked) = ${DATA_DIR}")

# # ----------------------------
# # 5) Target, link, and RPATH
# # ----------------------------
# add_executable(torch_ir_mem main.cpp)
# target_link_libraries(torch_ir_mem "${TORCH_LIBRARIES}")
# target_compile_definitions(torch_ir_mem PRIVATE DATA_DIR="${DATA_DIR}")

# # Embed RPATH so the binary can find libtorch*.so without LD_LIBRARY_PATH
# set_target_properties(torch_ir_mem PROPERTIES
#   BUILD_RPATH "${_torch_lib}"
#   INSTALL_RPATH "${_torch_lib}"
#   INSTALL_RPATH_USE_LINK_PATH TRUE
# )

# # Optional: if you ever need to force C++11 ABI (usually not necessary on modern stacks)
# # add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

# -=-=-=-=-=-=-=-=-=--=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-

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