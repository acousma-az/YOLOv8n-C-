#pragma once
#include <vector>

// 插值模式
enum class ResizeMode {
    NEAREST,
    LINEAR,
    CUBIC
};

// 坐标变换模式
enum class CoordinateTransformMode {
    ASYMMETRIC,
    ALIGN_CORNERS,
    PYTORCH_HALF_PIXEL
};

// 最近邻模式
enum class NearestMode {
    ROUND_PREFER_FLOOR,
    ROUND_PREFER_CEIL,
    FLOOR,
    CEIL
};

// 4D张量resize
std::vector<std::vector<std::vector<std::vector<float>>>> resize(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input,
    int out_height,
    int out_width,
    ResizeMode mode = ResizeMode::NEAREST,
    CoordinateTransformMode coord_mode = CoordinateTransformMode::ASYMMETRIC,
    NearestMode nearest_mode = NearestMode::FLOOR,
    float cubic_coeff_a = -0.75f
);
