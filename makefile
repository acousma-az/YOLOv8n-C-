# Makefile for YOLO Medical Imaging Project
# 包含三个主要功能：YOLO检测推理、测试和特征向量提取

# 定义项目路径
PROJECT_ROOT := /home/tpu1/project/heyang
YOLO_DIR := $(PROJECT_ROOT)/yolo
YOLO_MEDICAL_DIR := $(PROJECT_ROOT)/YOLOv8-Medical-Imaging-master
DETECT_SCRIPTS_DIR := $(YOLO_MEDICAL_DIR)/detect_scripts
DETECTION_DIR := $(YOLO_MEDICAL_DIR)/detection

# 定义编译器和Python解释器
CC := gcc
CXX := g++
PYTHON := python3
CFLAGS := -Wall -O2
CXXFLAGS := -Wall -O2 -std=c++11

# 默认目标
.PHONY: all help clean yolo-detect yolo-test extract-features build-yolo build-test

# 默认执行帮助
all: help

# 帮助信息
help:
	@echo "YOLO医学影像项目 Makefile"
	@echo "=========================="
	@echo "可用目标："
	@echo "  build-yolo      - 编译YOLO C/C++推理程序"
	@echo "  yolo-detect     - 运行YOLO检测推理 (C/C++)"
	@echo "  extract-features - 提取YOLOv8特征向量 (Python)"
	@echo "  build-all       - 编译所有C/C++程序"
	@echo "  help           - 显示此帮助信息"
	@echo ""
	@echo "示例用法："
	@echo "  make build-yolo && make yolo-detect"
	@echo "  make extract-features"

# 编译YOLO推理程序
build:
	@echo "======================"
	@echo "编译YOLO推理程序..."
	@echo "======================"
	@echo "检查YOLO目录..."
	@if [ ! -d "$(YOLO_DIR)" ]; then \
		echo "错误: YOLO目录不存在: $(YOLO_DIR)"; \
		exit 1; \
	fi
	@echo "切换到YOLO目录: $(YOLO_DIR)"
	cd $(YOLO_DIR) && \
	if [ -d "detection" ] && [ -d "yolov8" ] && [ -f "detection/main.cpp" ]; then \
		echo "找到detection/main.cpp和yolov8目录，编译推理程序..."; \
		$(CXX) $(CXXFLAGS) detection/*.cpp yolov8/*.cpp yolov8/complex_block/*.cpp yolov8/operator/*.cpp yolov8/swish_block/*.cpp yolov8/weight/*.cpp -o app; \
		echo "编译完成: app (detection推理程序)"; \
	else \
		echo "未找到完整的YOLOv8项目结构"; \
		echo "需要的文件:"; \
		echo "  - detection/main.cpp"; \
		echo "  - yolov8/ 目录及其子目录"; \
		ls -la detection/ yolov8/ 2>/dev/null || echo "  目录不存在"; \
	fi

#  运行YOLO检测推理 (C/C++)
runc: build-yolo
	@echo "===================="
	@echo "运行YOLO检测推理..."
	@echo "===================="
	@echo "检查YOLO目录..."
	@if [ ! -d "$(YOLO_DIR)" ]; then \
		echo "错误: YOLO目录不存在: $(YOLO_DIR)"; \
		exit 1; \
	fi
	@echo "切换到YOLO目录: $(YOLO_DIR)"
	cd $(YOLO_DIR) && \
	if [ -f "app" ]; then \
		echo "运行编译好的 app 程序..."; \
		./app; \
	elif [ -f "test_yolov8" ]; then \
		echo "运行编译好的 test_yolov8 程序..."; \
		./test_yolov8; \
	elif [ -f "detect" ]; then \
		echo "运行编译好的 detect 程序..."; \
		./detect; \
	elif [ -f "inference" ]; then \
		echo "运行编译好的 inference 程序..."; \
		./inference; \
	elif [ -f "yolo" ]; then \
		echo "运行编译好的 yolo 程序..."; \
		./yolo; \
	else \
		echo "未找到编译好的可执行文件"; \
		echo "可用的可执行文件:"; \
		find . -type f -executable | head -5; \
	fi


#  提取YOLOv8特征张量（py）
runpy:
	@echo "========================"
	@echo "提取YOLOv8特征向量..."
	@echo "========================"
	@echo "检查特征提取脚本..."
	@if [ ! -f "$(DETECT_SCRIPTS_DIR)/extract_feature_vectors.py" ]; then \
		echo "错误: 特征提取脚本不存在: $(DETECT_SCRIPTS_DIR)/extract_feature_vectors.py"; \
		exit 1; \
	fi
	@echo "切换到检测脚本目录: $(DETECT_SCRIPTS_DIR)"
	cd $(DETECT_SCRIPTS_DIR) && \
	echo "运行特征向量提取脚本..." && \
	$(PYTHON) extract_feature_vectors.py
	@echo "特征向量提取完成！"
	@echo "查看生成的文件："
	@ls -la $(DETECT_SCRIPTS_DIR)/embed_* $(DETECT_SCRIPTS_DIR)/feature_* 2>/dev/null || echo "暂无生成文件"

# 快捷方式别名
detect: yolo-detect
test: yolo-test
features: extract-features
build: build-all

# 编译所有C/C++程序
build-all: build-yolo build-test
	@echo "========================="
	@echo "所有C/C++程序编译完成！"
	@echo "========================="

# 组合命令
run-all: build-all extract-features
	@echo "先运行YOLO推理..."
	@make yolo-detect
	@echo "然后运行测试..."
	@make yolo-test
	@echo "最后提取特征向量..."
	@make extract-features
	@echo "========================="
	@echo "所有任务执行完成！"
	@echo "========================="

# 检查环境
check-env:
	@echo "========================="
	@echo "检查项目环境..."
	@echo "========================="
	@echo "编译器版本:"
	@$(CC) --version | head -1
	@$(CXX) --version | head -1
	@echo "Python版本:"
	@$(PYTHON) --version
	@echo ""
	@echo "项目目录结构:"
	@echo "YOLO目录: $(YOLO_DIR)"
	@ls -la $(YOLO_DIR) 2>/dev/null || echo "  目录不存在"
	@echo ""
	@echo "测试目录:"
	@echo "  test1: $(PROJECT_ROOT)/test1"
	@ls $(PROJECT_ROOT)/test1/*.c $(PROJECT_ROOT)/test1/*.cpp 2>/dev/null || echo "    无C/C++文件"
	@echo "  test2: $(PROJECT_ROOT)/test2" 
	@ls $(PROJECT_ROOT)/test2/*.c $(PROJECT_ROOT)/test2/*.cpp 2>/dev/null || echo "    无C/C++文件"
	@echo "  test_operator: $(PROJECT_ROOT)/test_operator"
	@ls $(PROJECT_ROOT)/test_operator/*.c $(PROJECT_ROOT)/test_operator/*.cpp 2>/dev/null || echo "    无C/C++文件"
	@echo ""
	@echo "YOLO医学影像目录: $(YOLO_MEDICAL_DIR)"
	@ls -la $(YOLO_MEDICAL_DIR) 2>/dev/null || echo "  目录不存在"
	@echo ""
	@echo "检测脚本目录: $(DETECT_SCRIPTS_DIR)"
	@ls -la $(DETECT_SCRIPTS_DIR)/extract_feature_vectors.py 2>/dev/null || echo "  Python脚本不存在"
	@echo ""
	@echo "检查Python依赖:"
	@$(PYTHON) -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "  PyTorch未安装"
	@$(PYTHON) -c "import ultralytics; print('Ultralytics YOLO: 已安装')" 2>/dev/null || echo "  Ultralytics未安装"
	@$(PYTHON) -c "import numpy; print(f'NumPy: {numpy.__version__}')" 2>/dev/null || echo "  NumPy未安装"

# 只编译test_operator中的特定文件
build-test-operator:
	@echo "======================="
	@echo "编译test_operator..."
	@echo "======================="
	@if [ -d "$(PROJECT_ROOT)/test_operator" ]; then \
		cd $(PROJECT_ROOT)/test_operator && \
		for file in demo_comparison.cpp float_precision_demo.cpp test_add_3d.cpp test_add_4d.cpp test_compare_add3d.cpp test_compare_add4d.cpp test_compare_concat3d.cpp test_compare.cpp test_concat_3d.cpp test_concat3d_detailed.cpp test_concat.cpp test_conv_detailed.cpp test_div3d_detailed.cpp test_maxpool_detailed.cpp test_mul_detailed.cpp test_reshape4d3d_detailed.cpp test_reshape_detailed.cpp test_resize_detailed.cpp test_sigmoid3d_detailed.cpp test_sigmoid_detailed.cpp test_slice_detailed.cpp test_softmax_detailed.cpp test_split3d_detailed.cpp test_split_detailed.cpp test_sub3d_detailed.cpp test_transpose_detailed.cpp; do \
			if [ -f "$$file" ]; then \
				base=$$(basename "$$file" .cpp); \
				echo "编译 $$file -> $$base"; \
				$(CXX) $(CXXFLAGS) "$$file" -o "$$base" 2>/dev/null || echo "  编译失败: $$file"; \
			fi; \
		done; \
		echo "test_operator编译完成"; \
	fi

# 运行特定的test_operator测试
run-test-operator: build-test-operator
	@echo "========================"
	@echo "运行test_operator测试..."
	@echo "========================"
	@if [ -d "$(PROJECT_ROOT)/test_operator" ]; then \
		cd $(PROJECT_ROOT)/test_operator && \
		for exe in demo_comparison float_precision_demo test_add_3d test_add_4d test_compare_add3d; do \
			if [ -f "$$exe" ] && [ -x "$$exe" ]; then \
				echo "运行 $$exe..."; \
				"./$$exe"; \
				echo ""; \
			fi; \
		done; \
	fi

# 详细的特征提取（带参数）
extract-features-detailed:
	@echo "========================="
	@echo "详细特征向量提取..."
	@echo "========================="
	cd $(DETECT_SCRIPTS_DIR) && \
	$(PYTHON) extract_feature_vectors.py && \
	echo "特征提取完成！"
	@echo "生成的文件列表："
	@find $(DETECT_SCRIPTS_DIR) -name "embed_*" -o -name "feature_*" | sort

# 查看最近的特征提取结果
show-features:
	@echo "========================="
	@echo "最近的特征提取结果..."
	@echo "========================="
	@if [ -f "$(DETECT_SCRIPTS_DIR)/embed_extraction_summary.txt" ]; then \
		echo "特征提取摘要:"; \
		cat $(DETECT_SCRIPTS_DIR)/embed_extraction_summary.txt; \
	else \
		echo "未找到特征提取摘要文件"; \
	fi
	@echo ""
	@echo "生成的文件："
	@ls -lah $(DETECT_SCRIPTS_DIR)/embed_* $(DETECT_SCRIPTS_DIR)/feature_* 2>/dev/null || echo "无特征文件"
