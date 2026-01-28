#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include "concat_3d.h"

// 生成随机3D张量
std::vector<std::vector<std::vector<float>>> generate_random_3d_tensor(
    int depth, int height, int width, float min_val = -5.0f, float max_val = 5.0f) {
    
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

// 打印3D张量的形状和前几个值
void print_tensor_info(const std::vector<std::vector<std::vector<float>>>& tensor, 
                       const std::string& name) {
    std::cout << name << " shape: [" << tensor.size() << ", " 
              << (tensor.size() > 0 ? tensor[0].size() : 0) << ", " 
              << (tensor.size() > 0 && tensor[0].size() > 0 ? tensor[0][0].size() : 0) << "]" << std::endl;
    
    // 打印前几个值
    if (tensor.size() > 0 && tensor[0].size() > 0 && tensor[0][0].size() > 0) {
        std::cout << name << " first values: ";
        for (int i = 0; i < std::min(5, (int)tensor[0][0].size()); i++) {
            std::cout << std::fixed << std::setprecision(2) << tensor[0][0][i] << " ";
        }
        std::cout << "..." << std::endl;
    }
}

// 验证concat结果
bool verify_concat_3d(const std::vector<std::vector<std::vector<std::vector<float>>>>& inputs,
                      const std::vector<std::vector<std::vector<float>>>& result,
                      int axis) {
    if (inputs.empty()) return false;
    
    int d = inputs[0].size();
    int h = inputs[0][0].size();
    int w = inputs[0][0][0].size();
    
    // 检查几个随机位置
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (int test = 0; test < 10; test++) {
        std::uniform_int_distribution<int> d_dis(0, d - 1);
        std::uniform_int_distribution<int> h_dis(0, h - 1);
        std::uniform_int_distribution<int> w_dis(0, w - 1);
        
        int test_d = d_dis(gen);
        int test_h = h_dis(gen);
        int test_w = w_dis(gen);
        
        if (axis == 1) {
            // 沿height维度连接
            int h_offset = 0;
            for (int i = 0; i < inputs.size(); i++) {
                int tensor_h = inputs[i][0].size();
                if (test_h >= h_offset && test_h < h_offset + tensor_h) {
                    int local_h = test_h - h_offset;
                    if (std::abs(result[test_d][test_h][test_w] - 
                                inputs[i][test_d][local_h][test_w]) > 1e-5) {
                        return false;
                    }
                    break;
                }
                h_offset += tensor_h;
            }
        } else { // axis == 2
            // 沿width维度连接
            int w_offset = 0;
            for (int i = 0; i < inputs.size(); i++) {
                int tensor_w = inputs[i][0][0].size();
                if (test_w >= w_offset && test_w < w_offset + tensor_w) {
                    int local_w = test_w - w_offset;
                    if (std::abs(result[test_d][test_h][test_w] - 
                                inputs[i][test_d][test_h][local_w]) > 1e-5) {
                        return false;
                    }
                    break;
                }
                w_offset += tensor_w;
            }
        }
    }
    
    return true;
}

int main() {
    std::cout << "=== 三维张量Concat测试 ===" << std::endl << std::endl;
    
    // 测试案例1: 沿height维度连接 (axis=1)
    std::cout << "测试案例1: 沿height维度连接 (axis=1)" << std::endl;
    auto tensor1 = generate_random_3d_tensor(4, 3, 5);
    auto tensor2 = generate_random_3d_tensor(4, 2, 5);
    auto tensor3 = generate_random_3d_tensor(4, 4, 5);
    
    print_tensor_info(tensor1, "Tensor1");
    print_tensor_info(tensor2, "Tensor2");
    print_tensor_info(tensor3, "Tensor3");
    
    std::vector<std::vector<std::vector<std::vector<float>>>> inputs1 = {tensor1, tensor2, tensor3};
    auto result1 = concat_3d(inputs1, 1);
    
    print_tensor_info(result1, "Result1");
    std::cout << "验证结果: " << (verify_concat_3d(inputs1, result1, 1) ? "通过" : "失败") << std::endl;
    std::cout << std::endl;
    
    // 测试案例2: 沿width维度连接 (axis=2)
    std::cout << "测试案例2: 沿width维度连接 (axis=2)" << std::endl;
    auto tensor4 = generate_random_3d_tensor(3, 4, 2);
    auto tensor5 = generate_random_3d_tensor(3, 4, 3);
    auto tensor6 = generate_random_3d_tensor(3, 4, 1);
    
    print_tensor_info(tensor4, "Tensor4");
    print_tensor_info(tensor5, "Tensor5");
    print_tensor_info(tensor6, "Tensor6");
    
    std::vector<std::vector<std::vector<std::vector<float>>>> inputs2 = {tensor4, tensor5, tensor6};
    auto result2 = concat_3d(inputs2, 2);
    
    print_tensor_info(result2, "Result2");
    std::cout << "验证结果: " << (verify_concat_3d(inputs2, result2, 2) ? "通过" : "失败") << std::endl;
    std::cout << std::endl;
    
    // 测试案例3: 大规模张量连接
    std::cout << "测试案例3: 大规模张量连接" << std::endl;
    auto tensor7 = generate_random_3d_tensor(16, 32, 64);
    auto tensor8 = generate_random_3d_tensor(16, 16, 64);
    auto tensor9 = generate_random_3d_tensor(16, 8, 64);
    
    print_tensor_info(tensor7, "Tensor7");
    print_tensor_info(tensor8, "Tensor8");
    print_tensor_info(tensor9, "Tensor9");
    
    std::vector<std::vector<std::vector<std::vector<float>>>> inputs3 = {tensor7, tensor8, tensor9};
    auto result3 = concat_3d(inputs3, 1);
    
    print_tensor_info(result3, "Result3");
    std::cout << "验证结果: " << (verify_concat_3d(inputs3, result3, 1) ? "通过" : "失败") << std::endl;
    std::cout << std::endl;
    
    // 测试案例4: 错误情况测试
    std::cout << "测试案例4: 错误情况测试" << std::endl;
    try {
        auto tensor_wrong1 = generate_random_3d_tensor(4, 3, 5);
        auto tensor_wrong2 = generate_random_3d_tensor(4, 2, 3); // width不匹配
        
        std::vector<std::vector<std::vector<std::vector<float>>>> wrong_inputs = {tensor_wrong1, tensor_wrong2};
        auto wrong_result = concat_3d(wrong_inputs, 1); // 应该抛出异常
        
        std::cout << "错误：应该抛出异常但没有" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "正确捕获异常: " << e.what() << std::endl;
    }
    
    std::cout << std::endl << "=== 所有测试完成 ===" << std::endl;
    
    return 0;
}
