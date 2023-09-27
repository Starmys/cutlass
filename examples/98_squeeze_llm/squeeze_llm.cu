#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <vector>
#include <unordered_map>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemv.h"
#include "cutlass/gemm/kernel/gemv.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"

#include "squeeze_llm_gemv_device.h"
#include "gemv_kernel.h"
#include "gemv_kernel_batched.h"
// #include "squeeze_llm_gemv_kernel_raw.h"
// #include "squeeze_llm_gemv_kernel_sync.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

template<typename ElementIn, typename ElementOut>
int load_array_from_file(ElementOut* arr, std::string filepath) {
  std::cout << "Loading: " << filepath << std::endl;
  std::ifstream infile(filepath);
  int length = 0;
  ElementIn current_number;
  while (infile >> current_number) {
    arr[length++] = current_number;
  }
  return length;
}

/// Result structure
struct Result {

  double diff;
  double runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  //
  // Methods
  //

  Result(
    double runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess
  ):
    runtime_ms(runtime_ms), gflops(gflops), status(status), error(error), passed(true) { }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  int batch, K, N;
  int iterations;

  //
  // Methods
  // 

  Options():
    help(false),
    batch(1),
    K(4096),
    N(4096),
    iterations(20)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    cmd.get_cmd_line_argument("batch", batch, 1);
    cmd.get_cmd_line_argument("K", K, 4096);
    cmd.get_cmd_line_argument("N", N, 4096);
    cmd.get_cmd_line_argument("iterations", iterations, 20);
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {

    // Number of real-valued multiply-adds 
    int64_t fmas = (int64_t)batch * (int64_t)K * (int64_t)N;

    // Two flops per multiply-add
    // TODO: calc by thread block
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "98_squeeze_llm\n\n"
      << "  This example profiles the performance of a SqueezeLLM GEMV kernel.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement.\n\n"
      << "  --batch=<int>               Sets the batch size.\n"
      << "  --N=<int>                   Sets the N dimension.\n"
      << "  --K=<int>                   Sets the K dimension.\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n";

    out << "\n\nExamples:\n\n"

      << "# Runs a 4096x4096 SqueezeLLM GEMV\n"
      << "$ ./examples/98_squeeze_llm/98_squeeze_llm --batch=1 --N=4096 --K=4096\n\n";

    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemv>
class Testbed {
public:

  //
  // Type definitions
  //

  using ElementA = typename Gemv::ElementA;
  using ElementB = typename Gemv::ElementB;
  using ElementC = typename Gemv::ElementC;
  using ElementAccumulator = typename Gemv::ElementAccumulator;

  using LayoutA = typename Gemv::LayoutA;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using MatrixCoord = typename LayoutC::TensorCoord;

private:

  //
  // Data members
  //

  Options options;

  cutlass::HostTensor<int, LayoutA> tensor_quant_weight;
  cutlass::HostTensor<ElementA, LayoutA> tensor_quant_lut;
  cutlass::HostTensor<ElementA, LayoutA> tensor_weight;
  cutlass::HostTensor<ElementB, LayoutB> tensor_activations;
  cutlass::HostTensor<ElementC, LayoutC> tensor_bias;
  cutlass::HostTensor<ElementC, LayoutC> tensor_output;

  cutlass::HostTensor<ElementC, LayoutC> reference_output;

public:

  Testbed(Options const &options_):
    options(options_) { }

private:

  /// Initializes data structures
  void initialize_() {
    tensor_quant_weight.resize({options.N, options.K / 8});
    tensor_quant_lut.resize({options.N, 16});
    tensor_weight.resize({options.N, options.K});
    tensor_activations.resize({options.batch, options.K});
    tensor_bias.resize({options.batch, options.N});
    tensor_output.resize({options.batch, options.N});

    reference_output.resize({options.batch, options.N});

    std::string data_folder = "examples/98_squeeze_llm/data/";
    load_array_from_file<int, int>(tensor_quant_weight.host_ref().data(), data_folder + "quant_weight.txt");
    load_array_from_file<float, float>(tensor_quant_lut.host_ref().data(), data_folder + "quant_lut.txt");
    // load_array_from_file<float, float>(tensor_weight.host_ref().data(), data_folder + "weight.txt");
    load_array_from_file<float, float>(tensor_activations.host_ref().data(), data_folder + "activations.txt");
    load_array_from_file<float, float>(reference_output.host_ref().data(), data_folder + "output.txt");
    cutlass::reference::host::TensorFill(tensor_output.host_view());

    tensor_quant_weight.sync_device();
    tensor_quant_lut.sync_device();
    tensor_weight.sync_device();
    tensor_activations.sync_device();
    tensor_bias.sync_device();
    tensor_output.sync_device();
  }

  /// Verifies the result is a GEMV
  bool verify_(double& diff) {

    bool passed = true;

    tensor_output.sync_host();

    // Reference check
    // passed = cutlass::reference::host::TensorEquals(tensor_output.host_view(), reference_output.host_view());
    diff = cutlass::reference::host::TensorSumSqDiff(tensor_output.host_view(), reference_output.host_view());
    passed = diff / options.batch / options.N < 1e-6;

    if (!passed) {
      std::cerr << "\n***\nError - problem failed the QA check\n***\n" << std::endl;

      std::stringstream fname;

      fname << std::filesystem::path(__FILE__).parent_path().string()
            << "/error_98_squeeze_llm_"
            << options.batch << "x"
            << options.N << "x"
            << options.K << ".txt";

      std::cout << fname.str() << std::endl;

      std::ofstream results(fname.str());

      results
        << "\nQuant Weight:\n" << tensor_quant_weight.host_view() << "\n"
        << "\nQuant LUT:\n" << tensor_quant_lut.host_view() << "\n"
        << "\nActivations:\n" << tensor_activations.host_view() << "\n"
        << "\nOutput Reference:\n" << reference_output.host_view() << "\n"
        << "\nOutput Computed:\n" << tensor_output.host_view() << "\n";
    }

    return passed;
  }

public:

  /// Returns the number of threadblocks to launch if the kernel can run on the target
  /// device. Otherwise, returns zero.
  bool sufficient() const {
    //
    // Determine SMEM requirements and waive if not satisfied
    //

    int smem_size = int(sizeof(typename Gemv::GemvKernel::SharedStorage));

    cudaDeviceProp properties;
    int device_idx;
    cudaError_t result = cudaGetDevice(&device_idx);

    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDevice() API call failed.");
    }

    result = cudaGetDeviceProperties(&properties, device_idx);

    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDeviceProperties() failed");
    }

    if (properties.sharedMemPerBlockOptin < smem_size) {
      return false;
    }

    return true;
  }

  /// Executes a SqueezeLLM GeMV kernel and measures runtime.
  Result profile() {

    Result result;

    // Early exit
    if (!sufficient()) {
      std::cout << "Active CUDA device lacks hardware resources to run SqueezeLLM GeMV kernel." << std::endl;
      return result;
    }

    result.passed = false;

    // Initialize the problem
    initialize_();

    // Configure GEMV arguments
    typename Gemv::Arguments args(
      {options.N, options.K},
      options.batch,
      // tensor_weight.device_ref(),
      tensor_quant_weight.device_ref(),
      tensor_quant_lut.device_data(),
      tensor_activations.device_data(),
      tensor_bias.device_data(),
      tensor_output.device_data(),
      options.K,               // batch_stride_B
      options.N,               // batch_stride_C
      options.N                // batch_stride_D
    );

    // Initialize the GEMV object
    Gemv gemv;

    result.status = gemv.initialize(args);

    if (result.status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to initialize SqueezeLLM GeMV kernel." << std::endl;
      return result;
    }

    // Run the SqueezeLLM GeMV object
    result.status = gemv.run();

    if (result.status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to run SqueezeLLM GeMV kernel." << std::endl;
      return result;
    }

    // Wait for completion
    result.error = cudaDeviceSynchronize();

    if (result.error != cudaSuccess)  {
      std::cerr << "Kernel execution error: " << cudaGetErrorString(result.error);
      return result;
    }

    //
    // Verify correctness
    //
    result.passed = verify_(result.diff);

    result.error = cudaDeviceSynchronize();
    if (result.error != cudaSuccess)  {
      std::cerr << "Kernel execution error: " << cudaGetErrorString(result.error);
      return result;
    }

    //
    // Warm-up run
    //
    result.status = gemv.run();

    if (result.status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to run SqueezeLLM GeMV kernel." << std::endl;
      return result;
    }

    result.error = cudaDeviceSynchronize();
    if (result.error != cudaSuccess)  {
      std::cerr << "Kernel execution error: " << cudaGetErrorString(result.error);
      return result;
    }

    //
    // Construct events
    //

    cudaEvent_t events[2];

    for (auto & event : events) {
      result.error = cudaEventCreate(&event);
      if (result.error != cudaSuccess) {
        std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result.error) << std::endl;
        return -1;
      }
    }

    // Record an event at the start of a series of GEMV operations
    result.error = cudaEventRecord(events[0]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    //
    // Run profiling loop
    //

    result.error = cudaDeviceSynchronize();
    if (result.error != cudaSuccess)  {
      std::cerr << "Kernel execution error: " << cudaGetErrorString(result.error);
      return result;
    }

    for (int iter = 0; iter < options.iterations; ++iter) {
      gemv();
    }

    result.error = cudaDeviceSynchronize();
    if (result.error != cudaSuccess)  {
      std::cerr << "Kernel execution error: " << cudaGetErrorString(result.error);
      return result;
    }

    //
    // Stop profiling loop
    //

    // Record an event when the GEMV operations have been launched.
    result.error = cudaEventRecord(events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Wait for work on the device to complete.
    result.error = cudaEventSynchronize(events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Measure elapsed runtime
    float runtime_ms = 0;
    result.error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Compute average runtime and GFLOPs.
    result.runtime_ms = double(runtime_ms) / double(options.iterations);
    result.gflops = options.gflops(result.runtime_ms / 1000.0);

    //
    // Cleanup
    //

    for (auto event : events) {
      (void)cudaEventDestroy(event);
    }

    std::cout << std::endl;
    std::cout << "SqueezeLLM GeMV (CUTLASS):\n"
      << "       NxK = " << options.N << "x" << options.K << "\n"
      << "====================================================" << std::endl;

    std::cout << std::endl;
    std::cout << "    " << "SumDiff: " << result.diff << std::endl;
    std::cout << "    " << "Runtime: " << result.runtime_ms << " ms" << std::endl;
    std::cout << "    " << " GFLOPs: " << result.gflops << std::endl;

    return result;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (__CUDACC_VER_MAJOR__ < 11 || props.major < 8) {
  
    //
    // This example requires an NVIDIA Ampere-architecture GPU.
    //

    std::cout 
      << "SqueezeLLM GeMV example requires a GPU of NVIDIA's Ampere Architecture or "
      << "later (compute capability 80 or greater).\n";

    return 0;
  }

  //
  // Parse options
  //

  Options options;
  
  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  using ElementA = float;
  using ElementB = float;
  using ElementOutput = float;
  using ElementAccumulator = float;

  using LayoutA = cutlass::layout::RowMajor;

  int const kElementsPerAccess = 4;
  // int const kThreadCount = 64; // reg, sync
  // int const kStride = 256;
  // int const kThreadCount = 128; // smem
  // int const kStride = 128;
  int const kThreadCount = 64;
  int const kStride = 256;

  assert(options.batch > 0);
  assert(options.batch <= 32);

  using GemvKernel = cutlass::gemm::kernel::SqueezeLLMGemv<ElementA,
                                                           LayoutA,
                                                           ElementB,
                                                           ElementOutput,
                                                           ElementAccumulator,
                                                           kElementsPerAccess,
                                                           kThreadCount,
                                                           kStride,
                                                           2>;
  using Gemv = cutlass::gemm::device::SqueezeLLMGemv<GemvKernel>;

  //
  // Profile it
  //

  Testbed<Gemv> testbed(options);

  if (!testbed.sufficient()) {
    std::cout << "The active CUDA device lacks sufficient hardware resources to execute this kernel.\n";
    return 0;
  }

  Result result = testbed.profile();
  if (!result.passed) {
    std::cout << "\nFailed tp profile SqueezeLLM GEMV.\n";
    return -1;
  }

  std::cout << "\nPassed\n";

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
