import torch
import numpy as np

# 设置随机种子确保结果可重复
torch.manual_seed(12345)
np.random.seed(12345)

print("=== 生成add_3d对比测试数据 ===")

# 测试案例1: 基本加法
print("生成测试案例1: 基本3D张量加法")
input1 = torch.randn(3, 4, 5)
input2 = torch.randn(3, 4, 5)
result = input1 + input2

print(f"输入1形状: {input1.shape}")
print(f"输入2形状: {input2.shape}")
print(f"PyTorch结果形状: {result.shape}")

# 保存数据
np.save('add3d_test1_input1.npy', input1.numpy())
np.save('add3d_test1_input2.npy', input2.numpy())
np.save('add3d_test1_result_pytorch.npy', result.numpy())
print("保存到: add3d_test1_*.npy")

# 测试案例2: 广播测试
print("\n生成测试案例2: 广播操作")
input3 = torch.randn(3, 4, 5)
input4 = torch.randn(1, 4, 5)  # 第一维广播
result2 = input3 + input4

print(f"输入1形状: {input3.shape}")
print(f"输入2形状: {input4.shape}")
print(f"PyTorch结果形状: {result2.shape}")

np.save('add3d_test2_input1.npy', input3.numpy())
np.save('add3d_test2_input2.npy', input4.numpy())
np.save('add3d_test2_result_pytorch.npy', result2.numpy())
print("保存到: add3d_test2_*.npy")

# 测试案例3: 更复杂的广播
print("\n生成测试案例3: 复杂广播")
input5 = torch.randn(2, 3, 4)
input6 = torch.randn(2, 1, 4)  # 第二维广播
result3 = input5 + input6

print(f"输入1形状: {input5.shape}")
print(f"输入2形状: {input6.shape}")
print(f"PyTorch结果形状: {result3.shape}")

np.save('add3d_test3_input1.npy', input5.numpy())
np.save('add3d_test3_input2.npy', input6.numpy())
np.save('add3d_test3_result_pytorch.npy', result3.numpy())
print("保存到: add3d_test3_*.npy")

# 显示一些数值
print("\n测试案例1 - 部分数值:")
print(f"输入1[0,0,:3]: {input1[0,0,:3].numpy()}")
print(f"输入2[0,0,:3]: {input2[0,0,:3].numpy()}")
print(f"结果[0,0,:3]: {result[0,0,:3].numpy()}")

print("\nadd_3d数据生成完成！")
