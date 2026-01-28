import torch
import numpy as np

def save_4d_tensor(tensor, filename):
    """保存4D张量到npy文件"""
    numpy_data = tensor.detach().cpu().numpy()
    np.save(filename, numpy_data)
    print(f"保存 {filename}: {tensor.shape}")

def test_split():
    """生成split测试数据"""
    print("=== 生成 Split (4D) 对比测试数据 ===\n")
    
    # 测试案例1: 沿axis=1均等分割 (8通道 -> 2份，每份4通道)
    print("生成测试案例1: 沿axis=1均等分割 (8->4+4)")
    input1 = torch.randn(2, 8, 4, 4)
    axis1 = 1
    split_size1 = 4
    results1 = torch.split(input1, split_size1, dim=axis1)
    
    print(f"输入形状: {input1.shape}")
    print(f"axis: {axis1}, split_size: {split_size1}")
    print(f"分割结果数量: {len(results1)}")
    for i, result in enumerate(results1):
        print(f"  结果{i+1}形状: {result.shape}")
    print(f"输入范围: [{input1.min():.6f}, {input1.max():.6f}]")
    
    save_4d_tensor(input1, "split_case1_input.npy")
    for i, result in enumerate(results1):
        save_4d_tensor(result, f"split_case1_output{i+1}.npy")
    
    # 保存参数: axis, num_splits, split_size
    params1 = np.array([axis1, len(results1), split_size1])
    np.save("split_case1_params.npy", params1)
    print(f"保存参数: axis={axis1}, num_splits={len(results1)}, split_size={split_size1}")
    print("保存到: split_case1_*.npy\n")
    
    # 测试案例2: 沿axis=1均等分割 (12通道 -> 3份，每份4通道)
    print("生成测试案例2: 沿axis=1均等分割 (12->4+4+4)")
    input2 = torch.randn(1, 12, 6, 6)
    axis2 = 1
    split_size2 = 4
    results2 = torch.split(input2, split_size2, dim=axis2)
    
    print(f"输入形状: {input2.shape}")
    print(f"axis: {axis2}, split_size: {split_size2}")
    print(f"分割结果数量: {len(results2)}")
    for i, result in enumerate(results2):
        print(f"  结果{i+1}形状: {result.shape}")
    print(f"输入范围: [{input2.min():.6f}, {input2.max():.6f}]")
    
    save_4d_tensor(input2, "split_case2_input.npy")
    for i, result in enumerate(results2):
        save_4d_tensor(result, f"split_case2_output{i+1}.npy")
    
    params2 = np.array([axis2, len(results2), split_size2])
    np.save("split_case2_params.npy", params2)
    print(f"保存参数: axis={axis2}, num_splits={len(results2)}, split_size={split_size2}")
    print("保存到: split_case2_*.npy\n")
    
    # 测试案例3: 沿axis=1均等分割 (16通道 -> 4份，每份4通道)
    print("生成测试案例3: 沿axis=1均等分割 (16->4+4+4+4)")
    input3 = torch.randn(2, 16, 3, 3)
    axis3 = 1
    split_size3 = 4
    results3 = torch.split(input3, split_size3, dim=axis3)
    
    print(f"输入形状: {input3.shape}")
    print(f"axis: {axis3}, split_size: {split_size3}")
    print(f"分割结果数量: {len(results3)}")
    for i, result in enumerate(results3):
        print(f"  结果{i+1}形状: {result.shape}")
    print(f"输入范围: [{input3.min():.6f}, {input3.max():.6f}]")
    
    save_4d_tensor(input3, "split_case3_input.npy")
    for i, result in enumerate(results3):
        save_4d_tensor(result, f"split_case3_output{i+1}.npy")
    
    params3 = np.array([axis3, len(results3), split_size3])
    np.save("split_case3_params.npy", params3)
    print(f"保存参数: axis={axis3}, num_splits={len(results3)}, split_size={split_size3}")
    print("保存到: split_case3_*.npy\n")
    
    # 测试案例4: 不均等分割 (10通道 -> 3+4+3)
    print("生成测试案例4: 沿axis=1不均等分割 (10->3+4+3)")
    input4 = torch.randn(1, 10, 5, 5)
    axis4 = 1
    split_sizes4 = [3, 4, 3]
    results4 = torch.split(input4, split_sizes4, dim=axis4)
    
    print(f"输入形状: {input4.shape}")
    print(f"axis: {axis4}, split_sizes: {split_sizes4}")
    print(f"分割结果数量: {len(results4)}")
    for i, result in enumerate(results4):
        print(f"  结果{i+1}形状: {result.shape}")
    print(f"输入范围: [{input4.min():.6f}, {input4.max():.6f}]")
    
    save_4d_tensor(input4, "split_case4_input.npy")
    for i, result in enumerate(results4):
        save_4d_tensor(result, f"split_case4_output{i+1}.npy")
    
    # 保存不均等分割的参数: axis, 然后是split_sizes
    params4 = np.array([axis4] + split_sizes4)
    np.save("split_case4_params.npy", params4)
    print(f"保存参数: axis={axis4}, split_sizes={split_sizes4}")
    print("保存到: split_case4_*.npy\n")
    
    # 测试案例5: 边界测试 (6通道 -> 2份，每份3通道)
    print("生成测试案例5: 沿axis=1分成2份 (6->3+3)")
    input5 = torch.randn(3, 6, 2, 2)
    axis5 = 1
    split_size5 = 3
    results5 = torch.split(input5, split_size5, dim=axis5)
    
    print(f"输入形状: {input5.shape}")
    print(f"axis: {axis5}, split_size: {split_size5}")
    print(f"分割结果数量: {len(results5)}")
    for i, result in enumerate(results5):
        print(f"  结果{i+1}形状: {result.shape}")
    print(f"输入范围: [{input5.min():.6f}, {input5.max():.6f}]")
    
    save_4d_tensor(input5, "split_case5_input.npy")
    for i, result in enumerate(results5):
        save_4d_tensor(result, f"split_case5_output{i+1}.npy")
    
    params5 = np.array([axis5, len(results5), split_size5])
    np.save("split_case5_params.npy", params5)
    print(f"保存参数: axis={axis5}, num_splits={len(results5)}, split_size={split_size5}")
    print("保存到: split_case5_*.npy\n")
    
    print("=== Split (4D) 测试数据生成完成 ===")

if __name__ == "__main__":
    test_split()
