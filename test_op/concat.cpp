#include <vector>
#include <stdexcept>

// Concatenate tensors along a given axis
// tensors: vector of 4D tensors, each with shape [batch, channels, height, width]
// axis: concatenation axis (only axis=1 is supported for channel-wise concatenation)
std::vector<std::vector<std::vector<std::vector<float>>>> concat(
    const std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>& tensors,
    int axis = 1
) {
    if (tensors.empty()) {
        return {};
    }
    
    // Check for empty input tensors
    if (tensors[0].empty()) {
        throw std::invalid_argument("First tensor is empty");
    }
    
    // Get reference dimensions from the first tensor
    int batch_size = tensors[0].size();
    int ref_channels = tensors[0][0].size();
    int ref_height = (ref_channels > 0) ? tensors[0][0][0].size() : 0;
    int ref_width = (ref_height > 0 && ref_channels > 0) ? tensors[0][0][0][0].size() : 0;
    
    // Only channel axis (axis=1) concatenation is supported
    if (axis != 1) {
        throw std::invalid_argument("Only axis=1 (channel) concatenation is supported");
    }
    
    // Validate all tensors have the same batch, height, and width
    for (int i = 0; i < tensors.size(); i++) {
        const auto& tensor = tensors[i];
        
        if (tensor.size() != batch_size) {
            throw std::invalid_argument("All tensors must have the same batch size");
        }
        
        if (tensor.empty() || tensor[0].empty()) {
            continue; // Skip empty tensors
        }
        
        int tensor_channels = tensor[0].size();
        int tensor_height = tensor[0][0].size();
        int tensor_width = (tensor_height > 0) ? tensor[0][0][0].size() : 0;
        
        // Check dimensions consistency
        if (tensor_height != ref_height) {
            throw std::invalid_argument("All tensors must have the same height");
        }
        
        if (tensor_width != ref_width) {
            throw std::invalid_argument("All tensors must have the same width");
        }
        
        // Check internal consistency for each batch
        for (int b = 0; b < batch_size; b++) {
            if (tensor[b].size() != tensor_channels) {
                throw std::invalid_argument("Inconsistent channel count in tensor " + std::to_string(i));
            }
            
            for (int c = 0; c < tensor_channels; c++) {
                if (tensor[b][c].size() != tensor_height) {
                    throw std::invalid_argument("Inconsistent height in tensor " + std::to_string(i));
                }
                
                for (int h = 0; h < tensor_height; h++) {
                    if (tensor[b][c][h].size() != tensor_width) {
                        throw std::invalid_argument("Inconsistent width in tensor " + std::to_string(i));
                    }
                }
            }
        }
    }
    
    // Calculate total channels across all tensors
    int total_channels = 0;
    for (const auto& tensor : tensors) {
        if (!tensor.empty() && !tensor[0].empty()) {
            total_channels += tensor[0].size();
        }
    }
    
    // Handle case where all tensors are empty
    if (total_channels == 0) {
        return std::vector<std::vector<std::vector<std::vector<float>>>>(
            batch_size, 
            std::vector<std::vector<std::vector<float>>>()
        );
    }
    
    // Initialize output tensor [batch, total_channels, height, width]
    std::vector<std::vector<std::vector<std::vector<float>>>> output(
        batch_size, 
        std::vector<std::vector<std::vector<float>>>(
            total_channels,
            std::vector<std::vector<float>>(
                ref_height,
                std::vector<float>(ref_width, 0.0f)
            )
        )
    );
    
    // Perform concatenation
    int channel_offset = 0;
    for (const auto& tensor : tensors) {
        if (tensor.empty() || tensor[0].empty()) {
            continue; // Skip empty tensors
        }
        
        int tensor_channels = tensor[0].size();
        
        for (int b = 0; b < batch_size; b++) {
            // Ensure current tensor has the same number of channels for this batch
            if (tensor[b].size() != tensor_channels) {
                throw std::runtime_error("Inconsistent channel count during concatenation");
            }
            
            for (int c = 0; c < tensor_channels; c++) {
                // Copy channel data
                output[b][channel_offset + c] = tensor[b][c];
            }
        }
        
        channel_offset += tensor_channels;
    }
    
    return output;
}
