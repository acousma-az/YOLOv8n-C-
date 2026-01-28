import torch
import torch.nn.functional as F
import numpy as np

def save_to_npy(tensor, filename):
    """保存张量到numpy文件"""
    np.save(filename, tensor.numpy())
    print(f"保存 {filename}: {tensor.shape}")

def test_sigmoid_4d():
    print("=== 生成 Sigmoid 4D 对比测试数据 ===\n")
    
    torch.manual_seed(42)  # 保证可重现性
    
    # 测试案例1: 小尺寸正常值
    print("生成测试案例1: 小尺寸正常值")
    input1 = torch.randn(1, 3, 4, 4)  # [1, 3, 4, 4]
    result1 = torch.sigmoid(input1)
    
    print(f"输入形状: {input1.shape}")
    print(f"输入范围: [{input1.min().item():.6f}, {input1.max().item():.6f}]")
    print(f"结果形状: {result1.shape}")
    print(f"结果范围: [{result1.min().item():.6f}, {result1.max().item():.6f}]")
    
    save_to_npy(input1, "sigmoid_case1_input.npy")
    save_to_npy(result1, "sigmoid_case1_result.npy")
    print("保存到: sigmoid_case1_*.npy\n")
    
    # 测试案例2: 多batch测试
    print("生成测试案例2: 多batch测试")
    input2 = torch.randn(2, 8, 6, 6) * 2  # [2, 8, 6, 6] 扩大范围
    result2 = torch.sigmoid(input2)
    
    print(f"输入形状: {input2.shape}")
    print(f"输入范围: [{input2.min().item():.6f}, {input2.max().item():.6f}]")
    print(f"结果形状: {result2.shape}")
    print(f"结果范围: [{result2.min().item():.6f}, {result2.max().item():.6f}]")
    
    save_to_npy(input2, "sigmoid_case2_input.npy")
    save_to_npy(result2, "sigmoid_case2_result.npy")
    print("保存到: sigmoid_case2_*.npy\n")
    
    # 测试案例3: 极值测试
    print("生成测试案例3: 极值测试")
    input3 = torch.tensor([[
        [[-15.0, -10.0, -5.0, -2.0], 
         [-1.0, -0.5, 0.0, 0.5],
         [1.0, 2.0, 5.0, 10.0],
         [15.0, 20.0, -20.0, 25.0]]
    ]])  # [1, 1, 4, 4] 包含极值
    result3 = torch.sigmoid(input3)
    
    print(f"输入形状: {input3.shape}")
    print(f"输入范围: [{input3.min().item():.6f}, {input3.max().item():.6f}]")
    print(f"结果形状: {result3.shape}")
    print(f"结果范围: [{result3.min().item():.6f}, {result3.max().item():.6f}]")
    
    save_to_npy(input3, "sigmoid_case3_input.npy")
    save_to_npy(result3, "sigmoid_case3_result.npy")
    print("保存到: sigmoid_case3_*.npy\n")
    
    # 测试案例4: 零值和微小值
    print("生成测试案例4: 零值和微小值")
    input4 = torch.tensor([[
        [[0.0, 0.001, -0.001, 0.01], 
         [-0.01, 0.1, -0.1, 0.0001]],
        [[-0.0001, 1e-6, -1e-6, 0.0],
         [1e-5, -1e-5, 1e-4, -1e-4]]
    ]])  # [1, 2, 2, 4] 零值和微小值
    result4 = torch.sigmoid(input4)
    
    print(f"输入形状: {input4.shape}")
    print(f"输入范围: [{input4.min().item():.6f}, {input4.max().item():.6f}]")
    print(f"结果形状: {result4.shape}")
    print(f"结果范围: [{result4.min().item():.6f}, {result4.max().item():.6f}]")
    
    save_to_npy(input4, "sigmoid_case4_input.npy")
    save_to_npy(result4, "sigmoid_case4_result.npy")
    print("保存到: sigmoid_case4_*.npy\n")
    
    # 测试案例5: 大尺寸随机值
    print("生成测试案例5: 大尺寸随机值")
    input5 = torch.randn(4, 16, 8, 8) * 3  # [4, 16, 8, 8] 更大范围
    result5 = torch.sigmoid(input5)
    
    print(f"输入形状: {input5.shape}")
    print(f"输入范围: [{input5.min().item():.6f}, {input5.max().item():.6f}]")
    print(f"结果形状: {result5.shape}")
    print(f"结果范围: [{result5.min().item():.6f}, {result5.max().item():.6f}]")
    
    save_to_npy(input5, "sigmoid_case5_input.npy")
    save_to_npy(result5, "sigmoid_case5_result.npy")
    print("保存到: sigmoid_case5_*.npy\n")

if __name__ == "__main__":
    test_sigmoid_4d()
    print("=== Sigmoid 4D 测试数据生成完成 ===")
