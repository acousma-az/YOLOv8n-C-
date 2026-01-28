#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cmath>
#include "add_3d.h"

// 生成随机3D张量
std::vector<std::vector<std::vector<float>>> generate_random_3d_tensor(
    int depth, int height, int width, float min_val = -10.0f, float max_val = 10.0f) {
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    std::vector<std::vector<std::vector<float>>> tensor(
        depth, 
        std::vector<std::vector<float>>(
            height, 
            std::vector<float>(width)
        )
    );
    
    for (int d = 0; d < depth; d++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                tensor[d][h][w] = dis(gen);
            }
        }
    }
    
    return tensor;
}

// 打印3D张量的部分内容（用于调试）
void print_3d_tensor_sample(const std::vector<std::vector<std::vector<float>>>& tensor, 
                           const std::string& name, int sample_size = 3) {
    std::cout << name << " shape: [" << tensor.size() << ", " 
              << (tensor.size() > 0 ? tensor[0].size() : 0) << ", " 
              << (tensor.size() > 0 && tensor[0].size() > 0 ? tensor[0][0].size() : 0) << "]" << std::endl;
    
    std::cout << name << " sample values:" << std::endl;
    int max_d = std::min(sample_size, (int)tensor.size());
    int max_h = tensor.size() > 0 ? std::min(sample_size, (int)tensor[0].size()) : 0;
    int max_w = tensor.size() > 0 && tensor[0].size() > 0 ? 
                std::min(sample_size, (int)tensor[0][0].size()) : 0;
    
    for (int d = 0; d < max_d; d++) {
        std::cout << "  Channel " << d << ":" << std::endl;
        for (int h = 0; h < max_h; h++) {
            std::cout << "    ";
            for (int w = 0; w < max_w; w++) {
                std::cout << std::fixed << std::setprecision(3) << tensor[d][h][w] << " ";
            }
            if (max_w < tensor[0][0].size()) std::cout << "...";
            std::cout << std::endl;
        }
        if (max_h < tensor[0].size()) std::cout << "    ..." << std::endl;
    }
    if (max_d < tensor.size()) std::cout << "  ..." << std::endl;
    std::cout << std::endl;
}

// 验证结果是否正确（简单检查）
bool verify_addition(const std::vector<std::vector<std::vector<float>>>& input1,
                    const std::vector<std::vector<std::vector<float>>>& input2,
                    const std::vector<std::vector<std::vector<float>>>& result) {
    
    // 获取维度信息
    int d1 = input1.size();
    int h1 = (d1 > 0) ? input1[0].size() : 0;
    int w1 = (h1 > 0 && d1 > 0) ? input1[0][0].size() : 0;
    
    int d2 = input2.size();
    int h2 = (d2 > 0) ? input2[0].size() : 0;
    int w2 = (h2 > 0 && d2 > 0) ? input2[0][0].size() : 0;
    
    int d_out = result.size();
    int h_out = (d_out > 0) ? result[0].size() : 0;
    int w_out = (h_out > 0 && d_out > 0) ? result[0][0].size() : 0;
    
    // 检查几个随机点
    const int check_points = 10;
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (int i = 0; i < check_points && d_out > 0 && h_out > 0 && w_out > 0; i++) {
        std::uniform_int_distribution<int> d_dis(0, d_out - 1);
        std::uniform_int_distribution<int> h_dis(0, h_out - 1);
        std::uniform_int_distribution<int> w_dis(0, w_out - 1);
        
        int d = d_dis(gen);
        int h = h_dis(gen);
        int w = w_dis(gen);
        
        // 计算广播索引
        int d1_idx = (d1 == 1) ? 0 : (d1 > d) ? d : d % d1;
        int h1_idx = (h1 == 1) ? 0 : (h1 > h) ? h : h % h1;
        int w1_idx = (w1 == 1) ? 0 : (w1 > w) ? w : w % w1;
        
        int d2_idx = (d2 == 1) ? 0 : (d2 > d) ? d : d % d2;
        int h2_idx = (h2 == 1) ? 0 : (h2 > h) ? h : h % h2;
        int w2_idx = (w2 == 1) ? 0 : (w2 > w) ? w : w % w2;
        
        float expected = input1[d1_idx][h1_idx][w1_idx] + input2[d2_idx][h2_idx][w2_idx];
        float actual = result[d][h][w];
        
        if (std::abs(expected - actual) > 1e-5) {
            std::cout << "Verification failed at [" << d << "," << h << "," << w << "]" << std::endl;
            std::cout << "Expected: " << expected << ", Got: " << actual << std::endl;
            return false;
        }
    }
    
    return true;
}

int main() {
    std::cout << "=== 三维矩阵加法测试 ===" << std::endl;
    std::cout << std::endl;
    
    // 测试案例1: 大规模相同维度张量
    std::cout << "测试案例1: 大规模相同维度张量 [64, 128, 128]" << std::endl;
    auto input1 = generate_random_3d_tensor(64, 128, 128);
    auto input2 = generate_random_3d_tensor(64, 128, 128);
    
    print_3d_tensor_sample(input1, "Input1");
    print_3d_tensor_sample(input2, "Input2");
    
    auto start = std::chrono::high_resolution_clock::now();
    auto result1 = add_3d(input1, input2);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    print_3d_tensor_sample(result1, "Result1");
    std::cout << "计算时间: " << duration.count() << " ms" << std::endl;
    std::cout << "验证结果: " << (verify_addition(input1, input2, result1) ? "通过" : "失败") << std::endl;
    std::cout << std::endl;
    
    // 测试案例2: 广播 - 标量与张量
    std::cout << "测试案例2: 广播测试 [32, 64, 64] + [1, 1, 1]" << std::endl;
    auto input3 = generate_random_3d_tensor(32, 64, 64);
    auto input4 = generate_random_3d_tensor(1, 1, 1);  // 标量广播
    
    print_3d_tensor_sample(input3, "Input3");
    print_3d_tensor_sample(input4, "Input4 (scalar)");
    
    start = std::chrono::high_resolution_clock::now();
    auto result2 = add_3d(input3, input4);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    print_3d_tensor_sample(result2, "Result2");
    std::cout << "计算时间: " << duration.count() << " ms" << std::endl;
    std::cout << "验证结果: " << (verify_addition(input3, input4, result2) ? "通过" : "失败") << std::endl;
    std::cout << std::endl;
    
    // 测试案例3: 不同维度广播
    std::cout << "测试案例3: 不同维度广播 [16, 32, 64] + [1, 32, 1]" << std::endl;
    auto input5 = generate_random_3d_tensor(16, 32, 64);
    auto input6 = generate_random_3d_tensor(1, 32, 1);
    
    print_3d_tensor_sample(input5, "Input5");
    print_3d_tensor_sample(input6, "Input6");
    
    start = std::chrono::high_resolution_clock::now();
    auto result3 = add_3d(input5, input6);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    print_3d_tensor_sample(result3, "Result3");
    std::cout << "计算时间: " << duration.count() << " ms" << std::endl;
    std::cout << "验证结果: " << (verify_addition(input5, input6, result3) ? "通过" : "失败") << std::endl;
    std::cout << std::endl;
    
    // 测试案例4: 更大规模张量
    std::cout << "测试案例4: 更大规模张量 [128, 256, 256]" << std::endl;
    auto input7 = generate_random_3d_tensor(128, 256, 256);
    auto input8 = generate_random_3d_tensor(128, 256, 256);
    
    std::cout << "Input7 shape: [128, 256, 256]" << std::endl;
    std::cout << "Input8 shape: [128, 256, 256]" << std::endl;
    
    start = std::chrono::high_resolution_clock::now();
    auto result4 = add_3d(input7, input8);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Result4 shape: [" << result4.size() << ", " 
              << result4[0].size() << ", " << result4[0][0].size() << "]" << std::endl;
    std::cout << "计算时间: " << duration.count() << " ms" << std::endl;
    std::cout << "验证结果: " << (verify_addition(input7, input8, result4) ? "通过" : "失败") << std::endl;
    
    std::cout << std::endl << "=== 所有测试完成 ===" << std::endl;
    
    return 0;
}
