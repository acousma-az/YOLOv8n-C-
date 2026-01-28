#include <iostream>
#include <vector>
#include <cmath>

// 复制比较函数来演示
bool compare_3d_tensors_demo(const std::vector<std::vector<std::vector<float>>>& tensor1,
                             const std::vector<std::vector<std::vector<float>>>& tensor2,
                             float tolerance = 1e-5) {
    if (tensor1.size() != tensor2.size()) {
        std::cout << "维度0不同: " << tensor1.size() << " vs " << tensor2.size() << std::endl;
        return false;
    }
    
    for (int d = 0; d < tensor1.size(); d++) {
        if (tensor1[d].size() != tensor2[d].size()) {
            std::cout << "维度1不同: " << tensor1[d].size() << " vs " << tensor2[d].size() << std::endl;
            return false;
        }
        
        for (int h = 0; h < tensor1[d].size(); h++) {
            if (tensor1[d][h].size() != tensor2[d][h].size()) {
                std::cout << "维度2不同: " << tensor1[d][h].size() << " vs " << tensor2[d][h].size() << std::endl;
                return false;
            }
            
            for (int w = 0; w < tensor1[d][h].size(); w++) {
                if (std::abs(tensor1[d][h][w] - tensor2[d][h][w]) > tolerance) {
                    std::cout << "数值差异位置 [" << d << "," << h << "," << w << "]: "
                              << tensor1[d][h][w] << " vs " << tensor2[d][h][w] 
                              << " (差值: " << std::abs(tensor1[d][h][w] - tensor2[d][h][w]) << ")" << std::endl;
                    return false;
                }
            }
        }
    }
    return true;
}

int main() {
    std::cout << "=== 演示数值比较测试 ===" << std::endl << std::endl;
    
    // 创建两个相同的张量
    std::vector<std::vector<std::vector<float>>> tensor1 = {
        {{1.0f, 2.0f}, {3.0f, 4.0f}},
        {{5.0f, 6.0f}, {7.0f, 8.0f}}
    };
    
    std::vector<std::vector<std::vector<float>>> tensor2 = {
        {{1.0f, 2.0f}, {3.0f, 4.0f}},
        {{5.0f, 6.0f}, {7.0f, 8.0f}}
    };
    
    std::cout << "测试1: 相同张量" << std::endl;
    bool result1 = compare_3d_tensors_demo(tensor1, tensor2);
    std::cout << "结果: " << (result1 ? "✅ 通过" : "❌ 失败") << std::endl << std::endl;
    
    // 修改一个值，制造数值差异
    tensor2[1][0][1] = 6.1f;  // 从6.0改为6.1
    
    std::cout << "测试2: 有数值差异的张量" << std::endl;
    bool result2 = compare_3d_tensors_demo(tensor1, tensor2);
    std::cout << "结果: " << (result2 ? "✅ 通过" : "❌ 失败") << std::endl << std::endl;
    
    // 制造维度差异
    std::vector<std::vector<std::vector<float>>> tensor3 = {
        {{1.0f, 2.0f, 9.0f}, {3.0f, 4.0f, 10.0f}},  // 多一个元素
        {{5.0f, 6.0f, 11.0f}, {7.0f, 8.0f, 12.0f}}
    };
    
    std::cout << "测试3: 有维度差异的张量" << std::endl;
    bool result3 = compare_3d_tensors_demo(tensor1, tensor3);
    std::cout << "结果: " << (result3 ? "✅ 通过" : "❌ 失败") << std::endl << std::endl;
    
    std::cout << "=== 结论 ===" << std::endl;
    std::cout << "比较函数会检查:" << std::endl;
    std::cout << "1. 所有维度是否相同" << std::endl;
    std::cout << "2. 每个位置的数值差异是否在容忍度(1e-5)内" << std::endl;
    std::cout << "3. 如果发现差异，会显示具体位置和数值" << std::endl;
    
    return 0;
}
