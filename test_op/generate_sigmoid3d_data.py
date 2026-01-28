import torch
import torch.nn.functional as F
import numpy as np

def save_to_npy(tensor, filename):
    """保存张量到numpy文件"""
    np.save(filename, tensor.numpy())
    print(f"保存 {filename}: {tensor.shape}")

def test_sigmoid_3d():
    print("=== 生成 Sigmoid 3D 对比测试数据 ===\n")
    
    torch.manual_seed(42)  # 保证可重现性
    
    # 测试案例1: 小尺寸正常值
    print("生成测试案例1: 小尺寸正常值")
    input1 = torch.randn(2, 3, 4)  # [2, 3, 4]
    result1 = torch.sigmoid(input1)
    
    print(f"输入形状: {input1.shape}")
    print(f"输入范围: [{input1.min().item():.6f}, {input1.max().item():.6f}]")
    print(f"结果形状: {result1.shape}")
    print(f"结果范围: [{result1.min().item():.6f}, {result1.max().item():.6f}]")
    
    save_to_npy(input1, "sigmoid3d_case1_input.npy")
    save_to_npy(result1, "sigmoid3d_case1_result.npy")
    print("保存到: sigmoid3d_case1_*.npy\n")
    
    # 测试案例2: 极值测试
    print("生成测试案例2: 极值测试")
    input2 = torch.tensor([
        [[-10.0, -5.0, -1.0], [0.0, 1.0, 5.0], [10.0, 15.0, 20.0]],
        [[-20.0, -15.0, -2.0], [0.5, 2.0, 8.0], [12.0, 18.0, 25.0]]
    ])  # [2, 3, 3] 包含极大和极小值
    result2 = torch.sigmoid(input2)
    
    print(f"输入形状: {input2.shape}")
    print(f"输入范围: [{input2.min().item():.6f}, {input2.max().item():.6f}]")
    print(f"结果形状: {result2.shape}")
    print(f"结果范围: [{result2.min().item():.6f}, {result2.max().item():.6f}]")
    
    save_to_npy(input2, "sigmoid3d_case2_input.npy")
    save_to_npy(result2, "sigmoid3d_case2_result.npy")
    print("保存到: sigmoid3d_case2_*.npy\n")
    
    # 测试案例3: 零值和接近零的值
    print("生成测试案例3: 零值和接近零的值")
    input3 = torch.tensor([
        [[0.0, 0.001, -0.001], [0.1, -0.1, 0.01], [-0.01, 0.0001, -0.0001]]
    ])  # [1, 3, 3] 零值和微小值
    result3 = torch.sigmoid(input3)
    
    print(f"输入形状: {input3.shape}")
    print(f"输入范围: [{input3.min().item():.6f}, {input3.max().item():.6f}]")
    print(f"结果形状: {result3.shape}")
    print(f"结果范围: [{result3.min().item():.6f}, {result3.max().item():.6f}]")
    
    save_to_npy(input3, "sigmoid3d_case3_input.npy")
    save_to_npy(result3, "sigmoid3d_case3_result.npy")
    print("保存到: sigmoid3d_case3_*.npy\n")
    
    # 测试案例4: 大尺寸随机值
    print("生成测试案例4: 大尺寸随机值")
    input4 = torch.randn(8, 12, 16) * 3  # [8, 12, 16] 扩大范围
    result4 = torch.sigmoid(input4)
    
    print(f"输入形状: {input4.shape}")
    print(f"输入范围: [{input4.min().item():.6f}, {input4.max().item():.6f}]")
    print(f"结果形状: {result4.shape}")
    print(f"结果范围: [{result4.min().item():.6f}, {result4.max().item():.6f}]")
    
    save_to_npy(input4, "sigmoid3d_case4_input.npy")
    save_to_npy(result4, "sigmoid3d_case4_result.npy")
    print("保存到: sigmoid3d_case4_*.npy\n")
    
    # 测试案例5: 特殊数值稳定性测试
    print("生成测试案例5: 特殊数值稳定性测试")
    input5 = torch.tensor([
        [
            [-100.0, -50.0, -30.0, -20.0], 
            [-10.0, -5.0, -2.0, -1.0],
            [0.0, 1.0, 2.0, 5.0],
            [10.0, 20.0, 30.0, 50.0]
        ],
        [
            [-88.0, -44.0, -22.0, -11.0],
            [-7.0, -3.5, -1.5, -0.5],
            [0.5, 1.5, 3.5, 7.0],
            [11.0, 22.0, 44.0, 88.0]
        ]
    ])  # [2, 4, 4] 测试数值稳定性
    result5 = torch.sigmoid(input5)
    
    print(f"输入形状: {input5.shape}")
    print(f"输入范围: [{input5.min().item():.6f}, {input5.max().item():.6f}]")
    print(f"结果形状: {result5.shape}")
    print(f"结果范围: [{result5.min().item():.6f}, {result5.max().item():.6f}]")
    
    save_to_npy(input5, "sigmoid3d_case5_input.npy")
    save_to_npy(result5, "sigmoid3d_case5_result.npy")
    print("保存到: sigmoid3d_case5_*.npy\n")

if __name__ == "__main__":
    test_sigmoid_3d()
    print("=== Sigmoid 3D 测试数据生成完成 ===")
