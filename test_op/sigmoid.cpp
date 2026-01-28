#include <vector>
#include <cmath>

// Sigmoid activation function
// input: [batch, channels, height, width]
std::vector<std::vector<std::vector<std::vector<float>>>> sigmoid(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input
) {
    int batch = input.size();
    int channels = input[0].size();
    int height = input[0][0].size();
    int width = input[0][0][0].size();
    
    // Initialize output tensor with same dimensions
    std::vector<std::vector<std::vector<std::vector<float>>>> output(
        batch, std::vector<std::vector<std::vector<float>>>(
            channels, std::vector<std::vector<float>>(
                height, std::vector<float>(width, 0.0f)
            )
        )
    );
    
    // Apply sigmoid to each element
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    output[b][c][h][w] = 1.0f / (1.0f + std::exp(-input[b][c][h][w]));
                }
            }
        }
    }
    
    return output;
}
