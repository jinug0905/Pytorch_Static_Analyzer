// main.cpp
#include "StaticAnalyzer.h"
#include <iostream>
#include <string>

#ifndef DATA_DIR
#error "DATA_DIR not defined"
#endif

int main() {
    try {
        const std::string ts_path = std::string(DATA_DIR) + "/resnet50b1_script.pt";

        auto inf = analyze_inference_model(ts_path);
        std::cout << "[Inference]\n";
        std::cout << "  params (MB):  " << bytes_to_mb(inf.parameter_bytes) << "\n";
        std::cout << "  acts   (MB):  " << bytes_to_mb(inf.peak_activation_bytes) << "\n";
        std::cout << "  total  (MB):  " << bytes_to_mb(inf.total_peak_bytes) << "\n\n";

        auto tr = analyze_training_model(ts_path);
        std::cout << "[Training L2]\n";
        std::cout << "  params (MB):        " << bytes_to_mb(tr.parameter_bytes) << "\n";
        std::cout << "  fwd+saved (MB):     " << bytes_to_mb(tr.peak_forward_bytes) << "\n";
        std::cout << "  saved only (MB):    " << bytes_to_mb(tr.saved_for_backward) << "\n";
        std::cout << "  param grads (MB):   " << bytes_to_mb(tr.param_grad_bytes) << "\n";
        std::cout << "  TOTAL peak (MB):    " << bytes_to_mb(tr.total_peak_bytes) << "\n";

        return 0;
    } catch (const c10::Error& e) {
        std::cerr << "Torch error: " << e.msg() << "\n";
        return 1;
    }
}