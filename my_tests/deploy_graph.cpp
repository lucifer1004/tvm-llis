#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <cuda_runtime.h>
#include <chrono>

class MyModule {
  public:
    MyModule(tvm::runtime::Module* mod_factory) {
        DLDevice ctx_gpu{kDLCUDA, 0};

        gmod_ = mod_factory->GetFunction("default")(ctx_gpu);
        get_input_ = gmod_.GetFunction("get_input");
        get_output_ = gmod_.GetFunction("get_output");
        run_ = gmod_.GetFunction("run");

        input_dev_ = get_input_(0);
        output_dev_ = get_output_(0);
    }

    void run(float* output, unsigned output_size, float* input, unsigned input_size) {
        cudaMemcpy(input_dev_->data, input, input_size, cudaMemcpyHostToDevice);
        run_();
        cudaMemcpy(output, output_dev_->data, output_size, cudaMemcpyDeviceToHost);
    }

    auto get_input_shape() const {
        return input_dev_.Shape();
    }

    auto get_output_shape() const {
        return output_dev_.Shape();
    }

  private:
    tvm::runtime::Module gmod_;

    tvm::runtime::PackedFunc get_input_;
    tvm::runtime::PackedFunc get_output_;
    tvm::runtime::PackedFunc run_;

    tvm::runtime::NDArray input_dev_;
    tvm::runtime::NDArray output_dev_;
};

int main(int argc, char** argv) {
    const char* path = argv[1];
    unsigned num_iters = atoi(argv[2]);

    DLDevice ctx_gpu{kDLCUDA, 0};
    DLDevice ctx_cpu{kDLCPU, 0};

    std::vector<tvm::runtime::Module> mod_factories;
    mod_factories.reserve(num_iters);
    mod_factories.push_back(tvm::runtime::Module::LoadFromFile(path));

    auto start_time_factory = std::chrono::steady_clock::now();

    for (unsigned i = 1; i < num_iters; ++i) {
        mod_factories.push_back(tvm::runtime::Module::LoadFromFile(path));
    }

    auto end_time_factory = std::chrono::steady_clock::now();
    printf("Factory time: %f us\n", std::chrono::duration<double, std::micro>(end_time_factory - start_time_factory).count() / (num_iters - 1));

    std::vector<MyModule> my_modules;
    my_modules.reserve(num_iters);
    my_modules.emplace_back(&mod_factories[0]);

    auto start_time_init = std::chrono::steady_clock::now();

    for (unsigned i = 1; i < num_iters; ++i) {
        my_modules.emplace_back(&mod_factories[i]);
    }

    auto end_time_init = std::chrono::steady_clock::now();
    printf("Init time: %f us\n", std::chrono::duration<double, std::micro>(end_time_init - start_time_init).count() / (num_iters - 1));

    unsigned input_size = 1;
    for (unsigned x : my_modules[0].get_input_shape()) {
        input_size *= x;
    }

    unsigned output_size = 1;
    for (unsigned x : my_modules[0].get_output_shape()) {
        output_size *= x;
    }

    float* input;
    cudaMallocHost(&input, input_size);
    float* output;
    cudaMallocHost(&output, output_size);

    my_modules[0].run(output, output_size, input, input_size);

    auto start_time_run = std::chrono::steady_clock::now();

    for (unsigned i = 1; i < num_iters; ++i) {
        my_modules[i].run(output, output_size, input, input_size);
    }

    auto end_time_run = std::chrono::steady_clock::now();
    printf("Run time: %f us\n", std::chrono::duration<double, std::micro>(end_time_run - start_time_run).count() / (num_iters - 1));
}

