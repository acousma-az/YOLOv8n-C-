#include <vector>
#include <stdexcept>

// Transpose operation with permutation [0, 2, 1, 3]
// input: [batch, channels, height, width]
// output: [batch, height, channels, width]
std::vector<std::vector<std::vector<std::vector<float>>>> transpose(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input,
    const std::vector<int>& perm = {0, 2, 1, 3}
) {
    if (input.empty() || input[0].empty() || input[0][0].empty() || input[0][0][0].empty()) {
        return {};
    }
    
    // Get input dimensions
    int batch = input.size();
    int channels = input[0].size();
    int height = input[0][0].size();
    int width = input[0][0][0].size();
    
    // Validate permutation
    if (perm.size() != 4) {
        throw std::invalid_argument("Permutation must have 4 elements for 4D tensor");
    }
    
    // For perm [0, 2, 1, 3]: [batch, channels, height, width] -> [batch, height, channels, width]
    if (perm[0] != 0 || perm[1] != 2 || perm[2] != 1 || perm[3] != 3) {
        throw std::invalid_argument("Only permutation [0, 2, 1, 3] is supported");
    }
    
    // Initialize output tensor: [batch, height, channels, width]
    std::vector<std::vector<std::vector<std::vector<float>>>> output(
        batch, std::vector<std::vector<std::vector<float>>>(
            height, std::vector<std::vector<float>>(
                channels, std::vector<float>(width, 0.0f)
            )
        )
    );
    
    // Perform transpose operation
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    // input[b][c][h][w] -> output[b][h][c][w]
                    output[b][h][c][w] = input[b][c][h][w];
                }
            }
        }
    }
    
    return output;
}
