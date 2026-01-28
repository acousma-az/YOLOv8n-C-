import torch
import torch.nn.functional as F
import numpy as np

def save_to_npy(tensor, filename):
    """保存张量到numpy文件"""
    np.save(filename, tensor.numpy())
    print(f"保存 {filename}: {tensor.shape}")

def test_resize():
    print("=== 生成 Resize 对比测试数据 ===\n")
    
    torch.manual_seed(42)  # 保证可重现性
    
    # 测试案例1: 最近邻插值上采样
    print("生成测试案例1: 最近邻插值上采样")
    input1 = torch.randn(1, 3, 4, 4)  # [1, 3, 4, 4]
    output_h1, output_w1 = 8, 8
    # 使用最近邻插值
    result1 = F.interpolate(input1, size=(output_h1, output_w1), mode='nearest')
    
    print(f"输入形状: {input1.shape}")
    print(f"输出尺寸: ({output_h1}, {output_w1})")
    print(f"结果形状: {result1.shape}")
    
    save_to_npy(input1, "resize_case1_input.npy")
    save_to_npy(result1, "resize_case1_result.npy")
    # 保存参数
    np.save("resize_case1_params.npy", np.array([output_h1, output_w1], dtype=np.int32))
    print("保存到: resize_case1_*.npy\n")
    
    # 测试案例2: 最近邻插值下采样
    print("生成测试案例2: 最近邻插值下采样")
    input2 = torch.randn(2, 8, 16, 16)  # [2, 8, 16, 16]
    output_h2, output_w2 = 8, 8
    result2 = F.interpolate(input2, size=(output_h2, output_w2), mode='nearest')
    
    print(f"输入形状: {input2.shape}")
    print(f"输出尺寸: ({output_h2}, {output_w2})")
    print(f"结果形状: {result2.shape}")
    
    save_to_npy(input2, "resize_case2_input.npy")
    save_to_npy(result2, "resize_case2_result.npy")
    np.save("resize_case2_params.npy", np.array([output_h2, output_w2], dtype=np.int32))
    print("保存到: resize_case2_*.npy\n")
    
    # 测试案例3: 不同比例的resize
    print("生成测试案例3: 不同比例的resize")
    input3 = torch.randn(1, 4, 6, 9)  # [1, 4, 6, 9]
    output_h3, output_w3 = 12, 6  # height x2, width /1.5
    result3 = F.interpolate(input3, size=(output_h3, output_w3), mode='nearest')
    
    print(f"输入形状: {input3.shape}")
    print(f"输出尺寸: ({output_h3}, {output_w3})")
    print(f"结果形状: {result3.shape}")
    
    save_to_npy(input3, "resize_case3_input.npy")
    save_to_npy(result3, "resize_case3_result.npy")
    np.save("resize_case3_params.npy", np.array([output_h3, output_w3], dtype=np.int32))
    print("保存到: resize_case3_*.npy\n")
    
    # 测试案例4: 极端尺寸变化
    print("生成测试案例4: 极端尺寸变化")
    input4 = torch.randn(1, 2, 32, 32)  # [1, 2, 32, 32]
    output_h4, output_w4 = 8, 16  # 大幅下采样，不同比例
    result4 = F.interpolate(input4, size=(output_h4, output_w4), mode='nearest')
    
    print(f"输入形状: {input4.shape}")
    print(f"输出尺寸: ({output_h4}, {output_w4})")
    print(f"结果形状: {result4.shape}")
    
    save_to_npy(input4, "resize_case4_input.npy")
    save_to_npy(result4, "resize_case4_result.npy")
    np.save("resize_case4_params.npy", np.array([output_h4, output_w4], dtype=np.int32))
    print("保存到: resize_case4_*.npy\n")
    
    # 测试案例5: 单像素输入
    print("生成测试案例5: 单像素输入")
    input5 = torch.randn(2, 6, 1, 1)  # [2, 6, 1, 1]
    output_h5, output_w5 = 4, 6
    result5 = F.interpolate(input5, size=(output_h5, output_w5), mode='nearest')
    
    print(f"输入形状: {input5.shape}")
    print(f"输出尺寸: ({output_h5}, {output_w5})")
    print(f"结果形状: {result5.shape}")
    
    save_to_npy(input5, "resize_case5_input.npy")
    save_to_npy(result5, "resize_case5_result.npy")
    np.save("resize_case5_params.npy", np.array([output_h5, output_w5], dtype=np.int32))
    print("保存到: resize_case5_*.npy\n")

if __name__ == "__main__":
    test_resize()
    print("=== Resize 测试数据生成完成 ===")
