#!/bin/bash

echo "================================="
echo "    C++操作符完整对比测试套件"
echo "================================="
echo

# 检查是否存在必要的文件
echo "🔍 检查测试环境..."
if [ ! -f "concat.cpp" ] || [ ! -f "add_3d.cpp" ]; then
    echo "❌ 错误: 缺少必要的源文件"
    exit 1
fi

# 清理之前的编译结果
echo "🧹 清理环境..."
rm -f test_compare test_compare_add3d

# 生成测试数据
echo "📊 生成测试数据..."
echo "  - 生成concat操作符测试数据"
python generate_data.py
echo "  - 生成add_3d操作符测试数据" 
python generate_add3d_data.py

echo

# 编译测试程序
echo "🔨 编译测试程序..."
echo "  - 编译concat对比测试"
g++ -O2 -o test_compare test_compare.cpp npy_loader.cpp concat.cpp
if [ $? -ne 0 ]; then
    echo "❌ concat编译失败"
    exit 1
fi

echo "  - 编译add_3d对比测试"
g++ -O2 -o test_compare_add3d test_compare_add3d.cpp add_3d.cpp
if [ $? -ne 0 ]; then
    echo "❌ add_3d编译失败"
    exit 1
fi

echo "✅ 编译成功"
echo

# 运行测试
echo "🚀 运行对比测试..."
echo

echo "1️⃣ ===== CONCAT操作符测试 ====="
./test_compare
concat_result=$?

echo
echo "2️⃣ ===== ADD_3D操作符测试 ====="
./test_compare_add3d
add3d_result=$?

echo
echo "================================="
echo "📋 测试结果汇总"
echo "================================="

if [ $concat_result -eq 0 ]; then
    echo "✅ CONCAT操作符: 所有测试通过"
else
    echo "❌ CONCAT操作符: 测试失败"
fi

if [ $add3d_result -eq 0 ]; then
    echo "✅ ADD_3D操作符: 所有测试通过"
else
    echo "❌ ADD_3D操作符: 测试失败"
fi

echo
if [ $concat_result -eq 0 ] && [ $add3d_result -eq 0 ]; then
    echo "🎉 总体结果: 所有C++操作符与PyTorch完全匹配！"
    echo "📈 验证状态: C++实现正确性已确认"
else
    echo "⚠️  总体结果: 部分测试失败，需要检查实现"
fi

echo
echo "🔧 测试环境信息:"
echo "  - 编译器: $(g++ --version | head -1)"
echo "  - 优化级别: -O2"
echo "  - 测试框架: 直接数值对比"
echo "  - 容差: 1e-5"
echo
echo "测试完成！"
