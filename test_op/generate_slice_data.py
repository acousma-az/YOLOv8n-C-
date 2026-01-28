import torch
import numpy as np

def save_3d_tensor(tensor, filename):
    """保存3D张量到npy文件"""
    numpy_data = tensor.detach().cpu().numpy()
    np.save(filename, numpy_data)
    print(f"保存 {filename}: {tensor.shape}")

def test_slice():
    """生成slice测试数据"""
    print("=== 生成 Slice 对比测试数据 ===\n")
    
    # 测试案例1: 沿axis=0切片 (基础测试)
    print("生成测试案例1: 沿axis=0切片")
    input1 = torch.randn(6, 4, 3)
    start1, end1, axis1 = 1, 4, 0
    result1 = input1[start1:end1, :, :]
    
    print(f"输入形状: {input1.shape}")
    print(f"切片参数: start={start1}, end={end1}, axis={axis1}")
    print(f"结果形状: {result1.shape}")
    print(f"输入范围: [{input1.min():.6f}, {input1.max():.6f}]")
    print(f"结果范围: [{result1.min():.6f}, {result1.max():.6f}]")
    
    save_3d_tensor(input1, "slice_case1_input.npy")
    save_3d_tensor(result1, "slice_case1_result.npy")
    
    # 保存切片参数
    params1 = np.array([start1, end1, axis1])
    np.save("slice_case1_params.npy", params1)
    print(f"保存参数: start={start1}, end={end1}, axis={axis1}")
    print("保存到: slice_case1_*.npy\n")
    
    # 测试案例2: 沿axis=1切片
    print("生成测试案例2: 沿axis=1切片")
    input2 = torch.randn(3, 8, 5)
    start2, end2, axis2 = 2, 6, 1
    result2 = input2[:, start2:end2, :]
    
    print(f"输入形状: {input2.shape}")
    print(f"切片参数: start={start2}, end={end2}, axis={axis2}")
    print(f"结果形状: {result2.shape}")
    print(f"输入范围: [{input2.min():.6f}, {input2.max():.6f}]")
    print(f"结果范围: [{result2.min():.6f}, {result2.max():.6f}]")
    
    save_3d_tensor(input2, "slice_case2_input.npy")
    save_3d_tensor(result2, "slice_case2_result.npy")
    
    params2 = np.array([start2, end2, axis2])
    np.save("slice_case2_params.npy", params2)
    print(f"保存参数: start={start2}, end={end2}, axis={axis2}")
    print("保存到: slice_case2_*.npy\n")
    
    # 测试案例3: 沿axis=2切片
    print("生成测试案例3: 沿axis=2切片")
    input3 = torch.randn(4, 3, 10)
    start3, end3, axis3 = 3, 7, 2
    result3 = input3[:, :, start3:end3]
    
    print(f"输入形状: {input3.shape}")
    print(f"切片参数: start={start3}, end={end3}, axis={axis3}")
    print(f"结果形状: {result3.shape}")
    print(f"输入范围: [{input3.min():.6f}, {input3.max():.6f}]")
    print(f"结果范围: [{result3.min():.6f}, {result3.max():.6f}]")
    
    save_3d_tensor(input3, "slice_case3_input.npy")
    save_3d_tensor(result3, "slice_case3_result.npy")
    
    params3 = np.array([start3, end3, axis3])
    np.save("slice_case3_params.npy", params3)
    print(f"保存参数: start={start3}, end={end3}, axis={axis3}")
    print("保存到: slice_case3_*.npy\n")
    
    # 测试案例4: 边界切片 (取开头部分)
    print("生成测试案例4: 边界切片(开头)")
    input4 = torch.randn(5, 6, 4)
    start4, end4, axis4 = 0, 2, 0
    result4 = input4[start4:end4, :, :]
    
    print(f"输入形状: {input4.shape}")
    print(f"切片参数: start={start4}, end={end4}, axis={axis4}")
    print(f"结果形状: {result4.shape}")
    print(f"输入范围: [{input4.min():.6f}, {input4.max():.6f}]")
    print(f"结果范围: [{result4.min():.6f}, {result4.max():.6f}]")
    
    save_3d_tensor(input4, "slice_case4_input.npy")
    save_3d_tensor(result4, "slice_case4_result.npy")
    
    params4 = np.array([start4, end4, axis4])
    np.save("slice_case4_params.npy", params4)
    print(f"保存参数: start={start4}, end={end4}, axis={axis4}")
    print("保存到: slice_case4_*.npy\n")
    
    # 测试案例5: 边界切片 (取结尾部分)
    print("生成测试案例5: 边界切片(结尾)")
    input5 = torch.randn(2, 7, 8)
    start5, end5, axis5 = 4, 7, 1
    result5 = input5[:, start5:end5, :]
    
    print(f"输入形状: {input5.shape}")
    print(f"切片参数: start={start5}, end={end5}, axis={axis5}")
    print(f"结果形状: {result5.shape}")
    print(f"输入范围: [{input5.min():.6f}, {input5.max():.6f}]")
    print(f"结果范围: [{result5.min():.6f}, {result5.max():.6f}]")
    
    save_3d_tensor(input5, "slice_case5_input.npy")
    save_3d_tensor(result5, "slice_case5_result.npy")
    
    params5 = np.array([start5, end5, axis5])
    np.save("slice_case5_params.npy", params5)
    print(f"保存参数: start={start5}, end={end5}, axis={axis5}")
    print("保存到: slice_case5_*.npy\n")
    
    print("=== Slice 测试数据生成完成 ===")

if __name__ == "__main__":
    test_slice()
