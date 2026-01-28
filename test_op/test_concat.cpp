#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include "concat.h"

// 生成随机4D张量
std::vector<std::vector<std::vector<std::vector<float>>>> generate_random_4d_tensor(
    int batch, int channels, int height, int width, float min_val = -5.0f, float max_val = 5.0f) {
    
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

// 打印4D张量的形状和前几个值
void print_tensor_info(const std::vector<std::vector<std::vector<std::vector<float>>>>& tensor, 
                       const std::string& name) {
    std::cout << name << " shape: [" << tensor.size() << ", " 
              << (tensor.size() > 0 ? tensor[0].size() : 0) << ", " 
              << (tensor.size() > 0 && tensor[0].size() > 0 ? tensor[0][0].size() : 0) << ", "
              << (tensor.size() > 0 && tensor[0].size() > 0 && tensor[0][0].size() > 0 ? tensor[0][0][0].size() : 0) << "]" << std::endl;
    
    // 打印前几个值
    if (tensor.size() > 0 && tensor[0].size() > 0 && tensor[0][0].size() > 0 && tensor[0][0][0].size() > 0) {
        std::cout << name << " first values: ";
        for (int i = 0; i < std::min(5, (int)tensor[0][0][0].size()); i++) {
            std::cout << std::fixed << std::setprecision(2) << tensor[0][0][0][i] << " ";
        }
        std::cout << "..." << std::endl;
    }
}

// 验证concat结果
bool verify_concat_4d(const std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>& inputs,
                      const std::vector<std::vector<std::vector<std::vector<float>>>>& result) {
    if (inputs.empty()) return false;
    
    int batch = inputs[0].size();
    int height = inputs[0][0][0].size();
    int width = inputs[0][0][0][0].size();
    
    // 计算期望的总通道数
    int expected_channels = 0;
    for (const auto& tensor : inputs) {
        if (!tensor.empty() && !tensor[0].empty()) {
            expected_channels += tensor[0].size();
        }
    }
    
    // 检查结果形状
    if (result.size() != batch || 
        result[0].size() != expected_channels ||
        result[0][0].size() != height ||
        result[0][0][0].size() != width) {
        return false;
    }
    
    // 验证通道数据
    int channel_offset = 0;
    for (const auto& tensor : inputs) {
        if (tensor.empty() || tensor[0].empty()) continue;
        
        int tensor_channels = tensor[0].size();
        
        // 检查几个随机位置
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> b_dis(0, batch - 1);
        std::uniform_int_distribution<int> h_dis(0, height - 1);
        std::uniform_int_distribution<int> w_dis(0, width - 1);
        
        for (int test = 0; test < 5; test++) {
            int b = b_dis(gen);
            int h = h_dis(gen);
            int w = w_dis(gen);
            
            for (int c = 0; c < tensor_channels; c++) {
                if (std::abs(result[b][channel_offset + c][h][w] - tensor[b][c][h][w]) > 1e-5) {
                    return false;
                }
            }
        }
        
        channel_offset += tensor_channels;
    }
    
    return true;
}

int main() {
    std::cout << "=== 四维张量Concat测试 (沿通道维度) ===" << std::endl << std::endl;
    
    // 测试案例1: 基本通道连接
    std::cout << "测试案例1: 基本通道连接" << std::endl;
    auto tensor1 = generate_random_4d_tensor(2, 3, 4, 4);
    auto tensor2 = generate_random_4d_tensor(2, 2, 4, 4);
    auto tensor3 = generate_random_4d_tensor(2, 1, 4, 4);
    
    print_tensor_info(tensor1, "Tensor1");
    print_tensor_info(tensor2, "Tensor2");
    print_tensor_info(tensor3, "Tensor3");
    
    std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> inputs1 = {tensor1, tensor2, tensor3};
    auto result1 = concat(inputs1, 1);
    
    print_tensor_info(result1, "Result1");
    std::cout << "期望形状: [2, " << 3+2+1 << ", 4, 4] = [2, 6, 4, 4]" << std::endl;
    std::cout << "验证结果: " << (verify_concat_4d(inputs1, result1) ? "通过" : "失败") << std::endl;
    std::cout << std::endl;
    
    // 测试案例2: 不同通道数的张量
    std::cout << "测试案例2: 不同通道数的张量" << std::endl;
    auto tensor4 = generate_random_4d_tensor(1, 8, 8, 8);
    auto tensor5 = generate_random_4d_tensor(1, 4, 8, 8);
    auto tensor6 = generate_random_4d_tensor(1, 12, 8, 8);
    
    print_tensor_info(tensor4, "Tensor4");
    print_tensor_info(tensor5, "Tensor5");
    print_tensor_info(tensor6, "Tensor6");
    
    std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> inputs2 = {tensor4, tensor5, tensor6};
    auto result2 = concat(inputs2, 1);
    
    print_tensor_info(result2, "Result2");
    std::cout << "期望形状: [1, " << 8+4+12 << ", 8, 8] = [1, 24, 8, 8]" << std::endl;
    std::cout << "验证结果: " << (verify_concat_4d(inputs2, result2) ? "通过" : "失败") << std::endl;
    std::cout << std::endl;
    
    // 测试案例3: 大规模张量连接
    std::cout << "测试案例3: 大规模张量连接" << std::endl;
    auto tensor7 = generate_random_4d_tensor(4, 16, 32, 32);
    auto tensor8 = generate_random_4d_tensor(4, 8, 32, 32);
    auto tensor9 = generate_random_4d_tensor(4, 32, 32, 32);
    
    print_tensor_info(tensor7, "Tensor7");
    print_tensor_info(tensor8, "Tensor8");
    print_tensor_info(tensor9, "Tensor9");
    
    std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> inputs3 = {tensor7, tensor8, tensor9};
    auto result3 = concat(inputs3, 1);
    
    print_tensor_info(result3, "Result3");
    std::cout << "期望形状: [4, " << 16+8+32 << ", 32, 32] = [4, 56, 32, 32]" << std::endl;
    std::cout << "验证结果: " << (verify_concat_4d(inputs3, result3) ? "通过" : "失败") << std::endl;
    std::cout << std::endl;
    
    // 测试案例4: 错误情况测试 - 不同的高度
    std::cout << "测试案例4: 错误情况测试 - 不同的高度" << std::endl;
    try {
        auto tensor_wrong1 = generate_random_4d_tensor(2, 3, 4, 4);
        auto tensor_wrong2 = generate_random_4d_tensor(2, 2, 6, 4); // height不匹配
        
        std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> wrong_inputs = {tensor_wrong1, tensor_wrong2};
        auto wrong_result = concat(wrong_inputs, 1);
        
        std::cout << "错误：应该抛出异常但没有" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "正确捕获异常: " << e.what() << std::endl;
    }
    std::cout << std::endl;
    
    // 测试案例5: 错误情况测试 - 不支持的轴
    std::cout << "测试案例5: 错误情况测试 - 不支持的轴" << std::endl;
    try {
        auto tensor_a = generate_random_4d_tensor(2, 3, 4, 4);
        auto tensor_b = generate_random_4d_tensor(2, 2, 4, 4);
        
        std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> inputs = {tensor_a, tensor_b};
        auto wrong_result = concat(inputs, 2); // axis=2不支持
        
        std::cout << "错误：应该抛出异常但没有" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "正确捕获异常: " << e.what() << std::endl;
    }
    
    std::cout << std::endl << "=== 所有测试完成 ===" << std::endl;
    
    return 0;
}
