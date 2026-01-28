#include "image_resize.h"
#include "../yolov8/yolov8.h" 
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>  // 为memcpy添加头文件
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// 保存三维张量到文件 (batch, height, width) 或 (channels, height, width)
void save_3d_tensor(const std::vector<std::vector<std::vector<float>>>& tensor, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cout << "无法创建输出文件: " << filename << std::endl;
        return;
    }
    // 先计算统计信息（min/max/mean）
    double sum = 0.0;
    size_t count = 0;
    float minv = 0.0f, maxv = 0.0f;
    bool first = true;
    for (size_t i = 0; i < tensor.size(); ++i) {
        for (size_t j = 0; j < tensor[i].size(); ++j) {
            for (size_t k = 0; k < tensor[i][j].size(); ++k) {
                float v = tensor[i][j][k];
                if (first) { minv = maxv = v; first = false; }
                if (v < minv) minv = v;
                if (v > maxv) maxv = v;
                sum += v;
                ++count;
            }
        }
    }
    double meanv = count ? (sum / count) : 0.0;

    file << "# 3D Tensor Shape: [" << tensor.size() << ", " 
         << (tensor.empty() ? 0 : tensor[0].size()) << ", " 
         << (tensor.empty() || tensor[0].empty() ? 0 : tensor[0][0].size()) << "]\n";
    file << "# Stats: min=" << minv << ", max=" << maxv << ", mean=" << meanv << "\n";

    // 同时打印到控制台
    std::cout << "Saving 3D tensor to: " << filename << std::endl;
    std::cout << "  Stats -> min: " << minv << ", max: " << maxv << ", mean: " << meanv << std::endl;
    
    for (size_t i = 0; i < tensor.size(); ++i) {
        file << "# Dimension 0, Index " << i << "\n";
        for (size_t j = 0; j < tensor[i].size(); ++j) {
            for (size_t k = 0; k < tensor[i][j].size(); ++k) {
                file << tensor[i][j][k];
                if (k < tensor[i][j].size() - 1) file << " ";
            }
            file << "\n";
        }
        if (i < tensor.size() - 1) file << "\n";
    }
    
    file.close();
    std::cout << "三维张量已保存到: " << filename << std::endl;
}

// 保存四维张量到文件 (batch, channels, height, width)
void save_4d_tensor(const std::vector<std::vector<std::vector<std::vector<float>>>>& tensor, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cout << "无法创建输出文件: " << filename << std::endl;
        return;
    }
    // 先计算统计信息（min/max/mean）
    double sum = 0.0;
    size_t count = 0;
    float minv = 0.0f, maxv = 0.0f;
    bool first = true;
    for (size_t b = 0; b < tensor.size(); ++b) {
        for (size_t c = 0; c < tensor[b].size(); ++c) {
            for (size_t h = 0; h < tensor[b][c].size(); ++h) {
                for (size_t w = 0; w < tensor[b][c][h].size(); ++w) {
                    float v = tensor[b][c][h][w];
                    if (first) { minv = maxv = v; first = false; }
                    if (v < minv) minv = v;
                    if (v > maxv) maxv = v;
                    sum += v;
                    ++count;
                }
            }
        }
    }
    double meanv = count ? (sum / count) : 0.0;

    file << "# 4D Tensor Shape: [" << tensor.size() << ", " 
         << (tensor.empty() ? 0 : tensor[0].size()) << ", " 
         << (tensor.empty() || tensor[0].empty() ? 0 : tensor[0][0].size()) << ", "
         << (tensor.empty() || tensor[0].empty() || tensor[0][0].empty() ? 0 : tensor[0][0][0].size()) << "]\n";
    file << "# Stats: min=" << minv << ", max=" << maxv << ", mean=" << meanv << "\n";

    // 同时打印到控制台
    std::cout << "Saving 4D tensor to: " << filename << std::endl;
    std::cout << "  Stats -> min: " << minv << ", max: " << maxv << ", mean: " << meanv << std::endl;
    
    for (size_t i = 0; i < tensor.size(); ++i) {
        file << "# Batch " << i << "\n";
        for (size_t j = 0; j < tensor[i].size(); ++j) {
            file << "## Channel " << j << "\n";
            for (size_t k = 0; k < tensor[i][j].size(); ++k) {
                for (size_t l = 0; l < tensor[i][j][k].size(); ++l) {
                    file << tensor[i][j][k][l];
                    if (l < tensor[i][j][k].size() - 1) file << " ";
                }
                file << "\n";
            }
            if (j < tensor[i].size() - 1) file << "\n";
        }
        if (i < tensor.size() - 1) file << "\n";
    }
    
    file.close();
    std::cout << "四维张量已保存到: " << filename << std::endl;
}

// 实现load_image函数
unsigned char* load_image(const char* filename, int* width, int* height, int* channels, int desired_channels) {
    unsigned char* data = stbi_load(filename, width, height, channels, desired_channels);
    if (!data) {
        std::cout << "stb_image加载失败: " << stbi_failure_reason() << std::endl;
        return nullptr;
    }
    return data;
}

int main() {
    const char* filename = "end_640x640.jpg";  // 使用当前目录下的图像文件
    int width, height, channels;
    unsigned char* data = load_image(filename, &width, &height, &channels, 3);

    if (!data) {
        std::cout << "无法加载图像: " << filename << std::endl;
        return 1;
    }

    std::cout << "成功加载图像: " << filename << std::endl;
    std::cout << "原始尺寸: " << width << "x" << height << ", 通道: " << channels << std::endl;
    std::cout << "原始数据格式: " << (channels == 3 ? "RGB (stb_image默认)" : "灰度") << std::endl;

    // 检查尺寸是否已经符合要求
    if (width == 640 && height == 640) {
        std::cout << "图片尺寸已经是640x640，跳过resize处理" << std::endl;
    } else {
        std::cout << "警告: 图片尺寸不是640x640，YOLOv8可能需要640x640输入" << std::endl;
    }

    std::cout << "\n开始YOLO预处理步骤:" << std::endl;
    
    // 步骤1: 数据已经是RGB格式（stb_image默认），无需转换
    unsigned char* rgb_data = new unsigned char[width * height * channels];
    if (channels == 3) {
        std::cout << "步骤1: 数据已经是RGB格式，直接使用" << std::endl;
        
        // 打印原始RGB像素值 (0,0)
        std::cout << "  原始RGB (0,0): [" << (int)data[0] << ", " << (int)data[1] << ", " << (int)data[2] << "]" << std::endl;
        
        // stb_image加载的已经是RGB格式，直接复制
        memcpy(rgb_data, data, width * height * channels * sizeof(unsigned char));
        
        // 打印确认RGB像素值 (0,0)
        std::cout << "  确认RGB (0,0): [" << (int)rgb_data[0] << ", " << (int)rgb_data[1] << ", " << (int)rgb_data[2] << "]" << std::endl;
    } else {
        // 如果不是3通道，直接复制
        memcpy(rgb_data, data, width * height * channels * sizeof(unsigned char));
    }
    
    // 步骤2: 归一化 (0-255 -> 0-1)
    std::cout << "步骤2: 像素值归一化 (0-255 -> 0-1)" << std::endl;
    float* normalized_data = new float[width * height * channels];
    for (int i = 0; i < width * height * channels; ++i) {
        normalized_data[i] = (float)rgb_data[i] / 255.0f;
    }
    
    // 打印归一化后的像素值 (0,0)
    std::cout << "  归一化RGB (0,0): [" << normalized_data[0] << ", " << normalized_data[1] << ", " << normalized_data[2] << "]" << std::endl;
    
    // 步骤3: HWC -> CHW 维度转换
    std::cout << "步骤3: HWC -> CHW 维度转换" << std::endl;
    std::cout << "转换前: [" << height << ", " << width << ", " << channels << "] (HWC)" << std::endl;
    std::cout << "转换后: [" << channels << ", " << height << ", " << width << "] (CHW)" << std::endl;
    
    float* chw_data = new float[channels * height * width];
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                // CHW格式: chw_data[c * height * width + h * width + w]
                // HWC格式: normalized_data[h * width * channels + w * channels + c]
                chw_data[c * height * width + h * width + w] = 
                    normalized_data[h * width * channels + w * channels + c];
            }
        }
    }
    
    // 打印CHW转换后的像素值 (0,0)
    std::cout << "  CHW格式 (0,0): R=" << chw_data[0] << ", G=" << chw_data[height * width] << ", B=" << chw_data[2 * height * width] << std::endl;
    
    // 打印一些统计信息
    float min_val = chw_data[0], max_val = chw_data[0];
    double sum = 0.0;
    for (int i = 0; i < channels * height * width; ++i) {
        if (chw_data[i] < min_val) min_val = chw_data[i];
        if (chw_data[i] > max_val) max_val = chw_data[i];
        sum += chw_data[i];
    }
    double mean_val = sum / (channels * height * width);
    
    std::cout << "预处理完成统计信息:" << std::endl;
    std::cout << "  - 数据范围: [" << min_val << ", " << max_val << "]" << std::endl;
    std::cout << "  - 平均值: " << mean_val << std::endl;
    std::cout << "  - 总元素数: " << channels * height * width << std::endl;
    
    // 各通道统计信息
    std::cout << "\n各通道统计信息:" << std::endl;
    for (int c = 0; c < channels; ++c) {
        float channel_min = chw_data[c * height * width];
        float channel_max = chw_data[c * height * width];
        double channel_sum = 0.0;
        
        for (int i = 0; i < height * width; ++i) {
            float val = chw_data[c * height * width + i];
            if (val < channel_min) channel_min = val;
            if (val > channel_max) channel_max = val;
            channel_sum += val;
        }
        double channel_mean = channel_sum / (height * width);
        
        std::cout << "  通道 " << c << ": 均值=" << channel_mean 
                  << ", 范围=[" << channel_min << ", " << channel_max << "]" << std::endl;
    }
    

    
    // 使用预处理后的数据
    float* input_data = chw_data;
    
    // 保存预处理结果为numpy格式，方便与Python对比
    std::ofstream cpp_result("cpp_preprocessing_result.txt");
    if (cpp_result.is_open()) {
        cpp_result << "# C++ 预处理结果\n";
        cpp_result << "# Shape: [" << channels << ", " << height << ", " << width << "]\n";
        cpp_result << "# 前10个像素值 (CHW格式):\n";
        for (int i = 0; i < std::min(10, channels * height * width); ++i) {
            cpp_result << input_data[i] << "\n";
        }
        cpp_result.close();
        std::cout << "C++预处理结果已保存到: cpp_preprocessing_result.txt" << std::endl;
    }

    auto result = yolov8(input_data);
    
    // 根据推理结果设置返回值
    if (result.empty()) {
        std::cout << "YOLOv8推理失败" << std::endl;
        return 1;
    }
    
    std::cout << "YOLOv8推理成功" << std::endl;
    std::cout << "输出张量维度: [" << result.size() << ", " 
              << (result.empty() ? 0 : result[0].size()) << ", " 
              << (result.empty() || result[0].empty() ? 0 : result[0][0].size()) << "]" << std::endl;
    std::cout << "预期格式: [batch=1, channels=7, detections=8400]" << std::endl;
    
    // result是三维张量，使用三维保存函数
    save_3d_tensor(result, "yolo_3d_tensor_output.txt");

    
    // 如果需要保存为四维张量，使用四维保存函数
    //save_4d_tensor(result, "yolo_4d_tensor_output.txt");
    

    // 清理分配的内存
    delete[] rgb_data;       // 清理BGR->RGB转换后的数据
    delete[] normalized_data; // 清理归一化后的数据
    delete[] chw_data;       // 清理CHW格式的数据 (input_data指向这里)
    stbi_image_free(data);   // 释放stb_image分配的内存
    
    std::cout << "C++ 图像预处理完成，结果已保存用于对比" << std::endl;
    
    return 0;
}