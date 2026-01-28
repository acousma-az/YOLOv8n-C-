#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cassert>
#include <algorithm>  // for std::replace
#include "onnx.pb.h"

void print_dim(const ::onnx::TensorShapeProto_Dimension &dim)
{
    switch (dim.value_case())
    {
    case onnx::TensorShapeProto_Dimension::ValueCase::kDimParam:
        std::cout << dim.dim_param();
        break;
    case onnx::TensorShapeProto_Dimension::ValueCase::kDimValue:
        std::cout << dim.dim_value();
        break;
    default:
        assert(false && "should never happen");
    }
}

void print_io_info(const ::google::protobuf::RepeatedPtrField<::onnx::ValueInfoProto> &info)
{
    for (auto input_data : info)
    {
        auto shape = input_data.type().tensor_type().shape();
        std::cout << "  " << input_data.name() << ":";
        std::cout << "[";
        if (shape.dim_size() != 0)
        {
            int size = shape.dim_size();
            for (int i = 0; i < size - 1; ++i)
            {
                print_dim(shape.dim(i));
                std::cout << ",";
            }
            print_dim(shape.dim(size - 1));
        }
        std::cout << "]\n";
    }
}

void print_initializer_info(const ::google::protobuf::RepeatedPtrField<::onnx::TensorProto>& info)
{
    for (auto input_data : info)
    {
        auto dims = input_data.dims();
        std::cout << "shapes: ";
        for (auto dim : dims)
            std::cout << dim << " ";
        std::cout << std::endl;

        auto raw_data = input_data.raw_data();
        float *data_r = (float*)raw_data.c_str();
        int k = raw_data.size() / 4; // float is 4 bytes

        int i = 0;
        while (i < k)
        {
            std::cout << *data_r << " ";
            data_r++;
            i++;
        }

        std::cout << "\nraw size: " << raw_data.size() << std::endl;
        std::cout << "  " << input_data.name() << "\n";
    }
}

void print_node_info(const ::google::protobuf::RepeatedPtrField<::onnx::NodeProto>& info)
{
    for (auto input_data : info)
    {
        auto op_type = input_data.op_type();
        auto shape = input_data.attribute();
        std::cout << op_type << " " << input_data.name() << ":";
        std::cout << "\nInputs: ";
        for (auto inp : input_data.input())
            std::cout << inp << " ";
        std::cout << "\nOutputs: ";
        for (auto outp : input_data.output())
            std::cout << outp << " ";
        std::cout << "\nAttributes: [";
        for (auto y : shape)
        {
            std::cout << y.name() << ": ";
            for (auto t : y.ints())
                std::cout << t << " ";
        }
        std::cout << "]\n";
    }
}

void export_weights_to_c(const ::google::protobuf::RepeatedPtrField<::onnx::TensorProto>& initializers,
                         const std::string& header_filename,
                         const std::string& source_filename)
{
    std::ofstream header_file(header_filename);
    std::ofstream source_file(source_filename);

    header_file << "#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n";
    header_file << "#include <stdint.h>\n\n";
    source_file << "#include \"" << header_filename << "\"\n\n";

    for (const auto& tensor : initializers) {
        std::string var_name = tensor.name();
        std::replace(var_name.begin(), var_name.end(), '.', '_');  

        auto dims = tensor.dims();
        int total_size = 1;
        for (auto dim : dims) {
            total_size *= dim;
        }

        header_file << "extern const float " << var_name << "[" << total_size << "];\n";
        source_file << "const float " << var_name << "[" << total_size << "] = {\n  ";

        const std::string& raw_data = tensor.raw_data();
        const float* data_ptr = reinterpret_cast<const float*>(raw_data.data());
        int num_elements = raw_data.size() / sizeof(float);

        for (int i = 0; i < num_elements; ++i) {
            source_file << std::fixed << std::setprecision(17) << data_ptr[i] << "f";
            if (i != num_elements - 1) source_file << ", ";
            if ((i + 1) % 8 == 0) source_file << "\n  ";
        }

        source_file << "\n};\n\n";
    }

    header_file << "\n#endif\n";
    header_file.close();
    source_file.close();

    std::cout << "Exported weights to " << header_filename << " and " << source_filename << std::endl;
}

int main(void)
{
    onnx::ModelProto model;
    std::ifstream input("RBC_WBC.onnx", std::ios::ate | std::ios::binary);
    if (!input) {
        std::cerr << "Failed to open ONNX file.\n";
        return -1;
    }

    std::streamsize size = input.tellg();
    input.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    input.read(buffer.data(), size);
    if (!model.ParseFromArray(buffer.data(), size)) {
        std::cerr << "Failed to parse ONNX model.\n";
        return -1;
    }

    auto graph = model.graph();
    std::cout << "Number of initializers (weights): " << graph.initializer_size() << std::endl;
    std::cout << "Graph inputs:\n";
    print_io_info(graph.input());
    std::cout << "Graph outputs:\n";
    print_io_info(graph.output());
    std::cout << "Graph initializer (weights):\n";
    print_initializer_info(graph.initializer());
    std::cout << "Graph nodes:\n";
    print_node_info(graph.node());

    export_weights_to_c(graph.initializer(), "weights.h", "weights.c");

    return 0;
}
