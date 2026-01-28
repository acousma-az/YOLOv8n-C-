#include <vector>
#include <stdexcept>

// Split tensor along a given axis
// input: [batch, channels, height, width]
// axis: 轴索引 (这里主要处理 axis=1 的通道分割)
// num_splits: 分割份数 (当指定该参数时忽略 split_sizes)
// split_sizes: 每份的大小列表
std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> split(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input,
    int axis,
    int num_splits = 0,
    const std::vector<int>& split_sizes = {}
) {
    // 获取输入张量维度
    size_t batch = input.size();
    if (batch == 0) return {};
    size_t channels = input[0].size();
    if (channels == 0) return {};
    size_t height = input[0][0].size();
    if (height == 0) return {};
    (void)input[0][0][0].size(); // suppress unused dimension warning if any
    
    // 确定实际分割方案
    std::vector<int> final_split_sizes;
    if (num_splits > 0) {
        // 均等分割模式
        if (channels % num_splits != 0) {
            throw std::invalid_argument("Channels must be divisible by num_splits");
        }
        int split_size = channels / num_splits;
        final_split_sizes = std::vector<int>(num_splits, split_size);
    } else if (!split_sizes.empty()) {
        // 自定义分割模式
    size_t total = 0;
    for (int size : split_sizes) {
            if (size <= 0) {
                throw std::invalid_argument("All split sizes must be positive");
            }
            total += size;
        }
    if (total != channels) {
            throw std::invalid_argument("Sum of split_sizes must equal channels");
        }
        final_split_sizes = split_sizes;
    } else {
        throw std::invalid_argument("Must specify either num_splits or split_sizes");
    }
    
    // 准备结果容器
    size_t num_outputs = final_split_sizes.size();
    std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> outputs;
    outputs.reserve(num_outputs);
    
    // 仅支持通道轴 (axis=1) 的分割
    if (axis != 1) {
        throw std::invalid_argument("Only axis=1 (channel) splitting is supported");
    }
    
    // 执行分割操作
    size_t start_channel = 0;
    for (size_t i = 0; i < num_outputs; i++) {
        size_t end_channel = start_channel + final_split_sizes[i];
        
        // 创建当前输出张量 [batch, split_size, height, width]
        std::vector<std::vector<std::vector<std::vector<float>>>> output_tensor;
        output_tensor.reserve(batch);
        
        for (size_t b = 0; b < batch; b++) {
            std::vector<std::vector<std::vector<float>>> batch_tensor;
            batch_tensor.reserve(final_split_sizes[i]);
            
            // 复制指定通道范围的数据
            for (size_t c = start_channel; c < end_channel; c++) {
                batch_tensor.push_back(input[b][c]);
            }
            output_tensor.push_back(std::move(batch_tensor));
        }
        outputs.push_back(std::move(output_tensor));
        start_channel = end_channel;
    }
    
    return outputs;
}