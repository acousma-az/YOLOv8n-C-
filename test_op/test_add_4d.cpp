#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cmath>
#include "add_4d.h"

// 生成随机4D张量
std::vector<std::vector<std::vector<std::vector<float>>>> generate_random_4d_tensor(
    int batch, int channels, int height, int width, float min_val = -10.0f, float max_val = 10.0f) {
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    std::vector<std::vector<std::vector<std::vector<float>>>> tensor(
        batch,
        std::vector<std::vector<std::vector<float>>>(
            channels,
            std::vector<std::vector<float>>(
                height, 
                std::vector<float>(width)
            )
        )
    );
    
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    tensor[b][c][h][w] = dis(gen);
                }
            }
        }
    }
    
    return tensor;
}

// 打印4D张量的部分内容（用于调试）
void print_4d_tensor_sample(const std::vector<std::vector<std::vector<std::vector<float>>>>& tensor, 
                           const std::string& name, int sample_size = 2) {
    std::cout << name << " shape: [" << tensor.size() << ", " 
              << (tensor.size() > 0 ? tensor[0].size() : 0) << ", " 
              << (tensor.size() > 0 && tensor[0].size() > 0 ? tensor[0][0].size() : 0) << ", "
              << (tensor.size() > 0 && tensor[0].size() > 0 && tensor[0][0].size() > 0 ? tensor[0][0][0].size() : 0) << "]" << std::endl;
    
    std::cout << name << " sample values:" << std::endl;
    int max_b = std::min(sample_size, (int)tensor.size());
    int max_c = tensor.size() > 0 ? std::min(sample_size, (int)tensor[0].size()) : 0;
    int max_h = tensor.size() > 0 && tensor[0].size() > 0 ? 
                std::min(sample_size, (int)tensor[0][0].size()) : 0;
    int max_w = tensor.size() > 0 && tensor[0].size() > 0 && tensor[0][0].size() > 0 ? 
                std::min(sample_size, (int)tensor[0][0][0].size()) : 0;
    
    for (int b = 0; b < max_b; b++) {
        std::cout << "  Batch " << b << ":" << std::endl;
        for (int c = 0; c < max_c; c++) {
            std::cout << "    Channel " << c << ":" << std::endl;
            for (int h = 0; h < max_h; h++) {
                std::cout << "      ";
                for (int w = 0; w < max_w; w++) {
                    std::cout << std::fixed << std::setprecision(3) << tensor[b][c][h][w] << " ";
                }
                if (max_w < tensor[0][0][0].size()) std::cout << "...";
                std::cout << std::endl;
            }
            if (max_h < tensor[0][0].size()) std::cout << "      ..." << std::endl;
        }
        if (max_c < tensor[0].size()) std::cout << "    ..." << std::endl;
    }
    if (max_b < tensor.size()) std::cout << "  ..." << std::endl;
    std::cout << std::endl;
}

// 验证结果是否正确（简单检查）
bool verify_addition_4d(const std::vector<std::vector<std::vector<std::vector<float>>>>& input1,
                        const std::vector<std::vector<std::vector<std::vector<float>>>>& input2,
                        const std::vector<std::vector<std::vector<std::vector<float>>>>& result) {
    
    // 获取维度信息
    int b1 = input1.size();
    int c1 = (b1 > 0) ? input1[0].size() : 0;
    int h1 = (c1 > 0 && b1 > 0) ? input1[0][0].size() : 0;
    int w1 = (h1 > 0 && c1 > 0 && b1 > 0) ? input1[0][0][0].size() : 0;
    
    int b2 = input2.size();
    int c2 = (b2 > 0) ? input2[0].size() : 0;
    int h2 = (c2 > 0 && b2 > 0) ? input2[0][0].size() : 0;
    int w2 = (h2 > 0 && c2 > 0 && b2 > 0) ? input2[0][0][0].size() : 0;
    
    int b_out = result.size();
    int c_out = (b_out > 0) ? result[0].size() : 0;
    int h_out = (c_out > 0 && b_out > 0) ? result[0][0].size() : 0;
    int w_out = (h_out > 0 && c_out > 0 && b_out > 0) ? result[0][0][0].size() : 0;
    
    // 检查几个随机点
    const int check_points = 20;
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (int i = 0; i < check_points && b_out > 0 && c_out > 0 && h_out > 0 && w_out > 0; i++) {
        std::uniform_int_distribution<int> b_dis(0, b_out - 1);
        std::uniform_int_distribution<int> c_dis(0, c_out - 1);
        std::uniform_int_distribution<int> h_dis(0, h_out - 1);
        std::uniform_int_distribution<int> w_dis(0, w_out - 1);
        
        int b = b_dis(gen);
        int c = c_dis(gen);
        int h = h_dis(gen);
        int w = w_dis(gen);
        
        // 计算广播索引
        int b1_idx = (b1 == 1) ? 0 : (b1 > b) ? b : b % b1;
        int c1_idx = (c1 == 1) ? 0 : (c1 > c) ? c : c % c1;
        int h1_idx = (h1 == 1) ? 0 : (h1 > h) ? h : h % h1;
        int w1_idx = (w1 == 1) ? 0 : (w1 > w) ? w : w % w1;
        
        int b2_idx = (b2 == 1) ? 0 : (b2 > b) ? b : b % b2;
        int c2_idx = (c2 == 1) ? 0 : (c2 > c) ? c : c % c2;
        int h2_idx = (h2 == 1) ? 0 : (h2 > h) ? h : h % h2;
        int w2_idx = (w2 == 1) ? 0 : (w2 > w) ? w : w % w2;
        
        float expected = input1[b1_idx][c1_idx][h1_idx][w1_idx] + 
                        input2[b2_idx][c2_idx][h2_idx][w2_idx];
        float actual = result[b][c][h][w];
        
        if (std::abs(expected - actual) > 1e-5) {
            std::cout << "Verification failed at [" << b << "," << c << "," << h << "," << w << "]" << std::endl;
            std::cout << "Expected: " << expected << ", Got: " << actual << std::endl;
            return false;
        }
    }
    
    return true;
}

int main() {
    std::cout << "=== 四维矩阵加法测试 ===" << std::endl;
    std::cout << std::endl;
    
    // 测试案例1: 中等规模相同维度张量
    std::cout << "测试案例1: 中等规模相同维度张量 [8, 32, 64, 64]" << std::endl;
    auto input1 = generate_random_4d_tensor(8, 32, 64, 64);
    auto input2 = generate_random_4d_tensor(8, 32, 64, 64);
    
    print_4d_tensor_sample(input1, "Input1");
    print_4d_tensor_sample(input2, "Input2");
    
    auto start = std::chrono::high_resolution_clock::now();
    auto result1 = add_4d(input1, input2);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    print_4d_tensor_sample(result1, "Result1");
    std::cout << "计算时间: " << duration.count() << " ms" << std::endl;
    std::cout << "验证结果: " << (verify_addition_4d(input1, input2, result1) ? "通过" : "失败") << std::endl;
    std::cout << std::endl;
    
    // 测试案例2: 广播 - 不同batch大小
    std::cout << "测试案例2: Batch广播测试 [4, 16, 32, 32] + [1, 16, 32, 32]" << std::endl;
    auto input3 = generate_random_4d_tensor(4, 16, 32, 32);
    auto input4 = generate_random_4d_tensor(1, 16, 32, 32);  // batch维度广播
    
    print_4d_tensor_sample(input3, "Input3");
    print_4d_tensor_sample(input4, "Input4 (batch broadcast)");
    
    start = std::chrono::high_resolution_clock::now();
    auto result2 = add_4d(input3, input4);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    print_4d_tensor_sample(result2, "Result2");
    std::cout << "计算时间: " << duration.count() << " ms" << std::endl;
    std::cout << "验证结果: " << (verify_addition_4d(input3, input4, result2) ? "通过" : "失败") << std::endl;
    std::cout << std::endl;
    
    // 测试案例3: 多维度广播
    std::cout << "测试案例3: 多维度广播 [2, 8, 16, 16] + [1, 1, 1, 16]" << std::endl;
    auto input5 = generate_random_4d_tensor(2, 8, 16, 16);
    auto input6 = generate_random_4d_tensor(1, 1, 1, 16);
    
    print_4d_tensor_sample(input5, "Input5");
    print_4d_tensor_sample(input6, "Input6 (multi-dim broadcast)");
    
    start = std::chrono::high_resolution_clock::now();
    auto result3 = add_4d(input5, input6);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    print_4d_tensor_sample(result3, "Result3");
    std::cout << "计算时间: " << duration.count() << " ms" << std::endl;
    std::cout << "验证结果: " << (verify_addition_4d(input5, input6, result3) ? "通过" : "失败") << std::endl;
    std::cout << std::endl;
    
    // 测试案例4: 大规模张量
    std::cout << "测试案例4: 大规模张量 [16, 64, 128, 128]" << std::endl;
    auto input7 = generate_random_4d_tensor(16, 64, 128, 128);
    auto input8 = generate_random_4d_tensor(16, 64, 128, 128);
    
    std::cout << "Input7 shape: [16, 64, 128, 128]" << std::endl;
    std::cout << "Input8 shape: [16, 64, 128, 128]" << std::endl;
    
    start = std::chrono::high_resolution_clock::now();
    auto result4 = add_4d(input7, input8);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Result4 shape: [" << result4.size() << ", " 
              << result4[0].size() << ", " << result4[0][0].size() << ", " 
              << result4[0][0][0].size() << "]" << std::endl;
    std::cout << "计算时间: " << duration.count() << " ms" << std::endl;
    std::cout << "验证结果: " << (verify_addition_4d(input7, input8, result4) ? "通过" : "失败") << std::endl;
    std::cout << std::endl;
    
    // 测试案例5: 超大规模张量
    std::cout << "测试案例5: 超大规模张量 [32, 128, 256, 256]" << std::endl;
    auto input9 = generate_random_4d_tensor(32, 128, 256, 256);
    auto input10 = generate_random_4d_tensor(32, 128, 256, 256);
    
    std::cout << "Input9 shape: [32, 128, 256, 256]" << std::endl;
    std::cout << "Input10 shape: [32, 128, 256, 256]" << std::endl;
    
    start = std::chrono::high_resolution_clock::now();
    auto result5 = add_4d(input9, input10);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    long long total_elements = 32LL * 128 * 256 * 256;
    double throughput = (duration.count() > 0) ? (double)total_elements / (double)duration.count() / 1000.0 : 0.0; // M元素/秒
    
    std::cout << "Result5 shape: [" << result5.size() << ", " 
              << result5[0].size() << ", " << result5[0][0].size() << ", " 
              << result5[0][0][0].size() << "]" << std::endl;
    std::cout << "计算时间: " << duration.count() << " ms" << std::endl;
    std::cout << "总元素数: " << total_elements << std::endl;
    std::cout << "吞吐量: " << std::fixed << std::setprecision(1) << throughput << " M元素/秒" << std::endl;
    std::cout << "验证结果: " << (verify_addition_4d(input9, input10, result5) ? "通过" : "失败") << std::endl;
    
    std::cout << std::endl << "=== 所有测试完成 ===" << std::endl;
    
    return 0;
}
