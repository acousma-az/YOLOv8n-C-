import torch
import torch.nn.functional as F
import numpy as np

def save_4d_tensor(tensor, filename):
    """保存4D张量到npy文件"""
    numpy_data = tensor.detach().cpu().numpy()
    np.save(filename, numpy_data)
    print(f"保存 {filename}: {tensor.shape}")

def test_softmax():
    """生成softmax测试数据"""
    print("=== 生成 Softmax 对比测试数据 ===\n")
    
    # 测试案例1: 沿axis=0的softmax (batch维度)
    print("生成测试案例1: 沿axis=0的softmax")
    input1 = torch.randn(4, 3, 2, 2)
    axis1 = 0
    result1 = F.softmax(input1, dim=axis1)
    
    print(f"输入形状: {input1.shape}")
    print(f"axis: {axis1}")
    print(f"结果形状: {result1.shape}")
    print(f"输入范围: [{input1.min():.6f}, {input1.max():.6f}]")
    print(f"结果范围: [{result1.min():.6f}, {result1.max():.6f}]")
    print(f"结果总和(沿axis={axis1}): {result1.sum(dim=axis1).unique()}")
    
    save_4d_tensor(input1, "softmax_case1_input.npy")
    save_4d_tensor(result1, "softmax_case1_result.npy")
    
    # 保存axis参数
    axis_param1 = np.array([axis1])
    np.save("softmax_case1_params.npy", axis_param1)
    print(f"保存参数: axis={axis1}")
    print("保存到: softmax_case1_*.npy\n")
    
    # 测试案例2: 沿axis=1的softmax (通道维度)
    print("生成测试案例2: 沿axis=1的softmax")
    input2 = torch.randn(2, 8, 4, 4)
    axis2 = 1
    result2 = F.softmax(input2, dim=axis2)
    
    print(f"输入形状: {input2.shape}")
    print(f"axis: {axis2}")
    print(f"结果形状: {result2.shape}")
    print(f"输入范围: [{input2.min():.6f}, {input2.max():.6f}]")
    print(f"结果范围: [{result2.min():.6f}, {result2.max():.6f}]")
    print(f"结果总和(沿axis={axis2}): {result2.sum(dim=axis2).unique()}")
    
    save_4d_tensor(input2, "softmax_case2_input.npy")
    save_4d_tensor(result2, "softmax_case2_result.npy")
    
    axis_param2 = np.array([axis2])
    np.save("softmax_case2_params.npy", axis_param2)
    print(f"保存参数: axis={axis2}")
    print("保存到: softmax_case2_*.npy\n")
    
    # 测试案例3: 沿axis=2的softmax (高度维度)
    print("生成测试案例3: 沿axis=2的softmax")
    input3 = torch.randn(2, 4, 6, 3)
    axis3 = 2
    result3 = F.softmax(input3, dim=axis3)
    
    print(f"输入形状: {input3.shape}")
    print(f"axis: {axis3}")
    print(f"结果形状: {result3.shape}")
    print(f"输入范围: [{input3.min():.6f}, {input3.max():.6f}]")
    print(f"结果范围: [{result3.min():.6f}, {result3.max():.6f}]")
    print(f"结果总和(沿axis={axis3}): {result3.sum(dim=axis3).unique()}")
    
    save_4d_tensor(input3, "softmax_case3_input.npy")
    save_4d_tensor(result3, "softmax_case3_result.npy")
    
    axis_param3 = np.array([axis3])
    np.save("softmax_case3_params.npy", axis_param3)
    print(f"保存参数: axis={axis3}")
    print("保存到: softmax_case3_*.npy\n")
    
    # 测试案例4: 沿axis=3的softmax (宽度维度)
    print("生成测试案例4: 沿axis=3的softmax")
    input4 = torch.randn(2, 3, 3, 8)
    axis4 = 3
    result4 = F.softmax(input4, dim=axis4)
    
    print(f"输入形状: {input4.shape}")
    print(f"axis: {axis4}")
    print(f"结果形状: {result4.shape}")
    print(f"输入范围: [{input4.min():.6f}, {input4.max():.6f}]")
    print(f"结果范围: [{result4.min():.6f}, {result4.max():.6f}]")
    print(f"结果总和(沿axis={axis4}): {result4.sum(dim=axis4).unique()}")
    
    save_4d_tensor(input4, "softmax_case4_input.npy")
    save_4d_tensor(result4, "softmax_case4_result.npy")
    
    axis_param4 = np.array([axis4])
    np.save("softmax_case4_params.npy", axis_param4)
    print(f"保存参数: axis={axis4}")
    print("保存到: softmax_case4_*.npy\n")
    
    # 测试案例5: 数值稳定性测试 (大值)
    print("生成测试案例5: 数值稳定性测试")
    input5 = torch.randn(1, 4, 2, 2) * 10  # 放大输入值
    axis5 = 1
    result5 = F.softmax(input5, dim=axis5)
    
    print(f"输入形状: {input5.shape}")
    print(f"axis: {axis5}")
    print(f"结果形状: {result5.shape}")
    print(f"输入范围: [{input5.min():.6f}, {input5.max():.6f}]")
    print(f"结果范围: [{result5.min():.6f}, {result5.max():.6f}]")
    print(f"结果总和(沿axis={axis5}): {result5.sum(dim=axis5).unique()}")
    
    save_4d_tensor(input5, "softmax_case5_input.npy")
    save_4d_tensor(result5, "softmax_case5_result.npy")
    
    axis_param5 = np.array([axis5])
    np.save("softmax_case5_params.npy", axis_param5)
    print(f"保存参数: axis={axis5}")
    print("保存到: softmax_case5_*.npy\n")
    
    print("=== Softmax 测试数据生成完成 ===")

if __name__ == "__main__":
    test_softmax()
