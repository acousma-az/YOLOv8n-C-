#!/usr/bin/env python3
"""
最终数值验证脚本 - 精确对比C++和Python的预处理结果
"""

def read_preprocessing_results():
    """读取C++和Python的预处理结果"""
    
    # 读取C++结果
    cpp_values = []
    with open('cpp_preprocessing_result.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    cpp_values.append(float(line))
                except ValueError:
                    continue
    
    # 读取Python结果  
    python_values = []
    with open('python_preprocessing_result.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    python_values.append(float(line))
                except ValueError:
                    continue
    
    return cpp_values, python_values

def analyze_differences(cpp_values, python_values):
    """分析数值差异"""
    
    print("=" * 60)
    print("最终数值验证结果")
    print("=" * 60)
    
    if len(cpp_values) != len(python_values):
        print(f"⚠️  数组长度不同: C++={len(cpp_values)}, Python={len(python_values)}")
        min_len = min(len(cpp_values), len(python_values))
    else:
        min_len = len(cpp_values)
        print(f"✅ 数组长度一致: {min_len}")
    
    # 计算差异
    differences = []
    max_diff = 0
    max_diff_idx = 0
    
    print(f"\n前10个像素值对比:")
    for i in range(min(10, min_len)):
        diff = abs(cpp_values[i] - python_values[i])
        differences.append(diff)
        
        if diff > max_diff:
            max_diff = diff
            max_diff_idx = i
            
        print(f"像素 {i:2d}: C++={cpp_values[i]:.10f}, Python={python_values[i]:.10f}, 差异={diff:.2e}")
    
    # 计算所有差异
    for i in range(min_len):
        diff = abs(cpp_values[i] - python_values[i])
        if diff > max_diff:
            max_diff = diff
            max_diff_idx = i
    
    avg_diff = sum(abs(cpp_values[i] - python_values[i]) for i in range(min_len)) / min_len
    
    print(f"\n差异统计:")
    print(f"最大差异: {max_diff:.2e} (位置 {max_diff_idx})")
    print(f"平均差异: {avg_diff:.2e}")
    print(f"相对误差: {max_diff/max(cpp_values) * 100:.8f}%")
    
    # 分类评估
    if max_diff < 1e-6:
        print(f"✅ 结果几乎完全一致 (差异 < 1e-6)")
        verdict = "完全等效"
    elif max_diff < 1e-4:
        print(f"✅ 结果高度一致 (差异 < 1e-4)")  
        verdict = "高度等效"
    elif max_diff < 1e-2:
        print(f"⚠️  存在小差异 (差异 < 1e-2)")
        verdict = "基本等效"
    else:
        print(f"❌ 存在显著差异 (差异 >= 1e-2)")
        verdict = "不等效"
    
    return verdict, max_diff, avg_diff

def main():
    print("读取预处理结果文件...")
    
    try:
        cpp_values, python_values = read_preprocessing_results()
        print(f"C++结果: {len(cpp_values)} 个值")
        print(f"Python结果: {len(python_values)} 个值")
        
        if len(cpp_values) == 0 or len(python_values) == 0:
            print("❌ 无法读取有效数据")
            return
            
        verdict, max_diff, avg_diff = analyze_differences(cpp_values, python_values)
        
        print("\n" + "=" * 60)
        print("最终结论")
        print("=" * 60)
        print(f"等效性评级: {verdict}")
        print(f"最大数值差异: {max_diff:.2e}")
        print(f"平均数值差异: {avg_diff:.2e}")
        
        if max_diff < 1e-6:
            print("\n🎯 C++和Python的图像预处理在数值上几乎完全一致")
            print("   可以安全地用于相同的机器学习模型")
            print("   微小差异来源于浮点精度和库实现细节")
        
    except FileNotFoundError as e:
        print(f"❌ 文件不存在: {e}")
    except Exception as e:
        print(f"❌ 处理错误: {e}")

if __name__ == "__main__":
    main()
