import torch
import numpy as np

def save_3d_tensor(tensor, filename):
    """保存3D张量到npy文件"""
    numpy_data = tensor.detach().cpu().numpy()
    np.save(filename, numpy_data)
    print(f"保存 {filename}: {tensor.shape}")

def test_split_3d():
    """生成split_3d测试数据"""
    print("=== 生成 Split_3D 对比测试数据 ===\n")
    
    # 测试案例1: 沿axis=0均等分割 (4 -> 2+2)
    print("生成测试案例1: 沿axis=0均等分割")
    input1 = torch.randn(6, 4, 3)
    axis1 = 0
    num_splits1 = 3  # 分成3份，每份2个
    results1 = torch.split(input1, 2, dim=axis1)
    
    print(f"输入形状: {input1.shape}")
    print(f"axis: {axis1}, num_splits: {num_splits1}")
    print(f"分割结果数量: {len(results1)}")
    for i, result in enumerate(results1):
        print(f"  结果{i+1}形状: {result.shape}")
    print(f"输入范围: [{input1.min():.6f}, {input1.max():.6f}]")
    
    save_3d_tensor(input1, "split3d_case1_input.npy")
    for i, result in enumerate(results1):
        save_3d_tensor(result, f"split3d_case1_output{i+1}.npy")
    
    # 保存参数
    params1 = np.array([axis1, num_splits1, 2])  # axis, num_splits, split_size
    np.save("split3d_case1_params.npy", params1)
    print(f"保存参数: axis={axis1}, num_splits={num_splits1}, split_size=2")
    print("保存到: split3d_case1_*.npy\n")
    
    # 测试案例2: 沿axis=1均等分割 (8 -> 2+2+2+2)
    print("生成测试案例2: 沿axis=1均等分割")
    input2 = torch.randn(3, 8, 5)
    axis2 = 1
    num_splits2 = 4  # 分成4份，每份2个
    results2 = torch.split(input2, 2, dim=axis2)
    
    print(f"输入形状: {input2.shape}")
    print(f"axis: {axis2}, num_splits: {num_splits2}")
    print(f"分割结果数量: {len(results2)}")
    for i, result in enumerate(results2):
        print(f"  结果{i+1}形状: {result.shape}")
    print(f"输入范围: [{input2.min():.6f}, {input2.max():.6f}]")
    
    save_3d_tensor(input2, "split3d_case2_input.npy")
    for i, result in enumerate(results2):
        save_3d_tensor(result, f"split3d_case2_output{i+1}.npy")
    
    params2 = np.array([axis2, num_splits2, 2])
    np.save("split3d_case2_params.npy", params2)
    print(f"保存参数: axis={axis2}, num_splits={num_splits2}, split_size=2")
    print("保存到: split3d_case2_*.npy\n")
    
    # 测试案例3: 沿axis=2均等分割 (9 -> 3+3+3)
    print("生成测试案例3: 沿axis=2均等分割")
    input3 = torch.randn(2, 4, 9)
    axis3 = 2
    num_splits3 = 3  # 分成3份，每份3个
    results3 = torch.split(input3, 3, dim=axis3)
    
    print(f"输入形状: {input3.shape}")
    print(f"axis: {axis3}, num_splits: {num_splits3}")
    print(f"分割结果数量: {len(results3)}")
    for i, result in enumerate(results3):
        print(f"  结果{i+1}形状: {result.shape}")
    print(f"输入范围: [{input3.min():.6f}, {input3.max():.6f}]")
    
    save_3d_tensor(input3, "split3d_case3_input.npy")
    for i, result in enumerate(results3):
        save_3d_tensor(result, f"split3d_case3_output{i+1}.npy")
    
    params3 = np.array([axis3, num_splits3, 3])
    np.save("split3d_case3_params.npy", params3)
    print(f"保存参数: axis={axis3}, num_splits={num_splits3}, split_size=3")
    print("保存到: split3d_case3_*.npy\n")
    
    # 测试案例4: 不均等分割 (10 -> 3+4+3)
    print("生成测试案例4: 沿axis=1不均等分割")
    input4 = torch.randn(2, 10, 4)
    axis4 = 1
    split_sizes4 = [3, 4, 3]
    results4 = torch.split(input4, split_sizes4, dim=axis4)
    
    print(f"输入形状: {input4.shape}")
    print(f"axis: {axis4}, split_sizes: {split_sizes4}")
    print(f"分割结果数量: {len(results4)}")
    for i, result in enumerate(results4):
        print(f"  结果{i+1}形状: {result.shape}")
    print(f"输入范围: [{input4.min():.6f}, {input4.max():.6f}]")
    
    save_3d_tensor(input4, "split3d_case4_input.npy")
    for i, result in enumerate(results4):
        save_3d_tensor(result, f"split3d_case4_output{i+1}.npy")
    
    # 保存不均等分割的参数: axis, 然后是split_sizes
    params4 = np.array([axis4] + split_sizes4)
    np.save("split3d_case4_params.npy", params4)
    print(f"保存参数: axis={axis4}, split_sizes={split_sizes4}")
    print("保存到: split3d_case4_*.npy\n")
    
    # 测试案例5: 边界测试 (只分成2份)
    print("生成测试案例5: 沿axis=0分成2份")
    input5 = torch.randn(4, 3, 6)
    axis5 = 0
    num_splits5 = 2  # 分成2份，每份2个
    results5 = torch.split(input5, 2, dim=axis5)
    
    print(f"输入形状: {input5.shape}")
    print(f"axis: {axis5}, num_splits: {num_splits5}")
    print(f"分割结果数量: {len(results5)}")
    for i, result in enumerate(results5):
        print(f"  结果{i+1}形状: {result.shape}")
    print(f"输入范围: [{input5.min():.6f}, {input5.max():.6f}]")
    
    save_3d_tensor(input5, "split3d_case5_input.npy")
    for i, result in enumerate(results5):
        save_3d_tensor(result, f"split3d_case5_output{i+1}.npy")
    
    params5 = np.array([axis5, num_splits5, 2])
    np.save("split3d_case5_params.npy", params5)
    print(f"保存参数: axis={axis5}, num_splits={num_splits5}, split_size=2")
    print("保存到: split3d_case5_*.npy\n")
    
    print("=== Split_3D 测试数据生成完成 ===")

if __name__ == "__main__":
    test_split_3d()
