#include <cstdlib>

/**
 * @brief 图像缩放并归一化到 CHW 格式
 * 
 * @param image_data 原始图像数据（HWC，RGB，unsigned char*）
 * @param orig_w 原始宽度
 * @param orig_h 原始高度
 * @param target_w 目标宽度
 * @param target_h 目标高度
 * @return float* 返回归一化后的 float 数组，CHW 格式，需手动 free
 */
float* resize_and_normalize(unsigned char* image_data, int orig_w, int orig_h, int target_w, int target_h) {
    float* output = (float*)malloc(3 * target_w * target_h * sizeof(float));
    if (!output) return nullptr;

    for (int y = 0; y < target_h; y++) {
        for (int x = 0; x < target_w; x++) {
            int src_x = x * orig_w / target_w;
            int src_y = y * orig_h / target_h;
            unsigned char* pixel = image_data + (src_y * orig_w + src_x) * 3;

            int idx = y * target_w + x;
            // 注意: OpenCV读取的是BGR格式，需要转换为RGB
            output[idx] = pixel[2] / 255.0f;                       // R (原来的B)
            output[target_w * target_h + idx] = pixel[1] / 255.0f; // G (保持不变)
            output[2 * target_w * target_h + idx] = pixel[0] / 255.0f; // B (原来的R)
        }
    }
    return output;  // CHW 格式 float*，RGB顺序
}
