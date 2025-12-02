#include "numerical_type.h"
#include "curvature_database.h"
#include "precision_analyzer.h"
#include "operations.h"

// This file exists to ensure the library compiles
// Most functionality is header-only for template/inline usage

namespace hnf {

// Version information
const char* version() {
    return "HNF Precision-Aware AD v1.0.0";
}

// Utility function to convert hardware model to string
std::string hardware_to_string(HardwareModel H) {
    switch(H) {
        case HardwareModel::FLOAT16: return "float16";
        case HardwareModel::BFLOAT16: return "bfloat16";
        case HardwareModel::FLOAT32: return "float32";
        case HardwareModel::FLOAT64: return "float64";
        case HardwareModel::FLOAT128: return "float128";
        default: return "unknown";
    }
}

} // namespace hnf
