# PyTorch Static Analyzer

>  This repository presents a public overview of PyTorch Static Analyzer project
> **"PyTorch Static Analyzer"**, which estimates GPU memory usage of deep learning models
> using TorchScript Intermediate Representation (IR).  

---

## 🔍 Overview

- **Goal:** Estimate memory consumption (weights, activations) statically without running inference.
- **Core Components:**
  - `1_ExportingIR/` – Exports model TorchScript IR.
  - `2_EstimateMem/` – C++ analyzer for IR traversal and size estimation.

---

## 🧾 Note
For access to the full version (e.g., detailed HPC scripts or datasets),
please contact:
**Jinug Lee** (jinuglee@tamu.edu)
