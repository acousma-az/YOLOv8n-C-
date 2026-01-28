import torch
import numpy as np

def save_to_npy(tensor, filename):
    """保存张量到numpy文件"""
    np.save(filename, tensor.numpy())
    print(f"保存 {filename}: {tensor.shape}")

def test_reshape_4d_to_3d():
    print("=== 生成 Reshape 4D to 3D 对比测试数据 ===\n")
    
    torch.manual_seed(42)  # 保证可重现性
    
    # 测试案例1: 默认reshape (合并height和width)
    print("生成测试案例1: 默认reshape (合并height和width)")
    input1 = torch.randn(2, 3, 4, 5)  # [2, 3, 4, 5]
    # 默认行为: [2, 3, 20] (height*width = 4*5 = 20)
    result1 = input1.view(2, 3, 20)
    
    print(f"输入形状: {input1.shape}")
    print(f"结果形状: {result1.shape}")
    
    save_to_npy(input1, "reshape4d3d_case1_input.npy")
    save_to_npy(result1, "reshape4d3d_case1_result.npy")
    # 保存目标形状信息（空，表示默认行为）
    np.save("reshape4d3d_case1_target_shape.npy", np.array([], dtype=np.int32))
    print("保存到: reshape4d3d_case1_*.npy\n")
    
    # 测试案例2: 指定3D形状 - 常规reshape
    print("生成测试案例2: 指定3D形状 - 常规reshape")
    input2 = torch.randn(1, 8, 3, 4)  # [1, 8, 3, 4] = 96 elements
    target_shape2 = [6, 4, 4]  # 6*4*4 = 96 elements
    result2 = input2.view(target_shape2)
    
    print(f"输入形状: {input2.shape}")
    print(f"目标形状: {target_shape2}")
    print(f"结果形状: {result2.shape}")
    
    save_to_npy(input2, "reshape4d3d_case2_input.npy")
    save_to_npy(result2, "reshape4d3d_case2_result.npy")
    np.save("reshape4d3d_case2_target_shape.npy", np.array(target_shape2, dtype=np.int32))
    print("保存到: reshape4d3d_case2_*.npy\n")
    
    # 测试案例3: 包含-1的自动推断
    print("生成测试案例3: 包含-1的自动推断")
    input3 = torch.randn(2, 6, 4, 3)  # [2, 6, 4, 3] = 144 elements
    target_shape3 = [8, -1, 6]  # 8*?*6 = 144, ? = 3
    result3 = input3.view(8, 3, 6)
    
    print(f"输入形状: {input3.shape}")
    print(f"目标形状: {target_shape3}")
    print(f"结果形状: {result3.shape}")
    
    save_to_npy(input3, "reshape4d3d_case3_input.npy")
    save_to_npy(result3, "reshape4d3d_case3_result.npy")
    np.save("reshape4d3d_case3_target_shape.npy", np.array(target_shape3, dtype=np.int32))
    print("保存到: reshape4d3d_case3_*.npy\n")
    
    # 测试案例4: 大尺寸测试
    print("生成测试案例4: 大尺寸测试")
    input4 = torch.randn(4, 8, 6, 9)  # [4, 8, 6, 9] = 1728 elements
    target_shape4 = [12, 24, 6]  # 12*24*6 = 1728 elements
    result4 = input4.view(target_shape4)
    
    print(f"输入形状: {input4.shape}")
    print(f"目标形状: {target_shape4}")
    print(f"结果形状: {result4.shape}")
    
    save_to_npy(input4, "reshape4d3d_case4_input.npy")
    save_to_npy(result4, "reshape4d3d_case4_result.npy")
    np.save("reshape4d3d_case4_target_shape.npy", np.array(target_shape4, dtype=np.int32))
    print("保存到: reshape4d3d_case4_*.npy\n")
    
    # 测试案例5: batch维度合并
    print("生成测试案例5: batch维度合并")
    input5 = torch.randn(3, 4, 2, 8)  # [3, 4, 2, 8] = 192 elements
    target_shape5 = [12, 4, 4]  # 12*4*4 = 192 elements (batch*channels, height, width)
    result5 = input5.view(target_shape5)
    
    print(f"输入形状: {input5.shape}")
    print(f"目标形状: {target_shape5}")
    print(f"结果形状: {result5.shape}")
    
    save_to_npy(input5, "reshape4d3d_case5_input.npy")
    save_to_npy(result5, "reshape4d3d_case5_result.npy")
    np.save("reshape4d3d_case5_target_shape.npy", np.array(target_shape5, dtype=np.int32))
    print("保存到: reshape4d3d_case5_*.npy\n")

if __name__ == "__main__":
    test_reshape_4d_to_3d()
    print("=== Reshape 4D to 3D 测试数据生成完成 ===")
