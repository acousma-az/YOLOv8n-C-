import torch
import numpy as np

def save_to_npy(tensor, filename):
    """保存张量到numpy文件"""
    np.save(filename, tensor.numpy())
    print(f"保存 {filename}: {tensor.shape}")

def test_reshape_3d_to_4d():
    print("=== 生成 Reshape 3D to 4D 对比测试数据 ===\n")
    
    torch.manual_seed(42)  # 保证可重现性
    
    # 测试案例1: 默认reshape (添加batch维度)
    print("生成测试案例1: 默认reshape (添加batch维度)")
    input1 = torch.randn(3, 4, 5)  # [3, 4, 5]
    # 默认行为: [1, 3, 4, 5]
    result1 = input1.unsqueeze(0)  # 添加batch维度
    
    print(f"输入形状: {input1.shape}")
    print(f"结果形状: {result1.shape}")
    
    save_to_npy(input1, "reshape_case1_input.npy")
    save_to_npy(result1, "reshape_case1_result.npy")
    # 保存目标形状信息（空，表示默认行为）
    np.save("reshape_case1_target_shape.npy", np.array([], dtype=np.int32))
    print("保存到: reshape_case1_*.npy\n")
    
    # 测试案例2: 指定4D形状 - 常规reshape
    print("生成测试案例2: 指定4D形状 - 常规reshape")
    input2 = torch.randn(2, 3, 8)  # [2, 3, 8] = 48 elements
    target_shape2 = [1, 6, 2, 4]  # 1*6*2*4 = 48 elements
    result2 = input2.view(target_shape2)
    
    print(f"输入形状: {input2.shape}")
    print(f"目标形状: {target_shape2}")
    print(f"结果形状: {result2.shape}")
    
    save_to_npy(input2, "reshape_case2_input.npy")
    save_to_npy(result2, "reshape_case2_result.npy")
    np.save("reshape_case2_target_shape.npy", np.array(target_shape2, dtype=np.int32))
    print("保存到: reshape_case2_*.npy\n")
    
    # 测试案例3: 包含-1的自动推断
    print("生成测试案例3: 包含-1的自动推断")
    input3 = torch.randn(4, 6, 5)  # [4, 6, 5] = 120 elements
    target_shape3 = [2, -1, 3, 4]  # 2*?*3*4 = 120, ? = 5
    result3 = input3.view(2, 5, 3, 4)
    
    print(f"输入形状: {input3.shape}")
    print(f"目标形状: {target_shape3}")
    print(f"结果形状: {result3.shape}")
    
    save_to_npy(input3, "reshape_case3_input.npy")
    save_to_npy(result3, "reshape_case3_result.npy")
    np.save("reshape_case3_target_shape.npy", np.array(target_shape3, dtype=np.int32))
    print("保存到: reshape_case3_*.npy\n")
    
    # 测试案例4: 大尺寸测试
    print("生成测试案例4: 大尺寸测试")
    input4 = torch.randn(8, 16, 9)  # [8, 16, 9] = 1152 elements
    target_shape4 = [3, 8, 6, 8]  # 3*8*6*8 = 1152 elements
    result4 = input4.view(target_shape4)
    
    print(f"输入形状: {input4.shape}")
    print(f"目标形状: {target_shape4}")
    print(f"结果形状: {result4.shape}")
    
    save_to_npy(input4, "reshape_case4_input.npy")
    save_to_npy(result4, "reshape_case4_result.npy")
    np.save("reshape_case4_target_shape.npy", np.array(target_shape4, dtype=np.int32))
    print("保存到: reshape_case4_*.npy\n")

if __name__ == "__main__":
    test_reshape_3d_to_4d()
    print("=== Reshape 3D to 4D 测试数据生成完成 ===")
