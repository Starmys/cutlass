#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"

#include "pit_gemm_device.h"

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
  int mode; // SDD == 0, DSD == 1 
  int M, K, N;
  int block_size_x;
  int block_size_y;
  float sparsity;
  int iterations;
  float alpha;
  float beta;

  //
  // Methods
  // 

  Options():
    help(false),
    mode(0),
    M(1024),
    K(1024),
    N(1024),
    block_size_x(8),
    block_size_y(1),
    sparsity(0.5f),
    iterations(20),
    alpha(1.0f),
    beta(0.0f)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    cmd.get_cmd_line_argument("mode", mode, 0);
    cmd.get_cmd_line_argument("M", M, 0);
    cmd.get_cmd_line_argument("K", K, 0);
    cmd.get_cmd_line_argument("N", N, 0);
    cmd.get_cmd_line_argument("bx", block_size_x, 8);
    cmd.get_cmd_line_argument("bt", block_size_y, 1);
    cmd.get_cmd_line_argument("iterations", iterations, 20);
    cmd.get_cmd_line_argument("alpha", alpha, 1.0f);
    cmd.get_cmd_line_argument("beta", beta, 0.0f);   
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {

    // Number of real-valued multiply-adds 
    int64_t fmas = (int64_t)M * (int64_t)K * (int64_t)N;

    // Two flops per multiply-add
    return 2.0 * double(fmas) * sparsity / double(1.0e9) / runtime_s;
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "99_pit_gemm\n\n"
      << "  This example profiles the performance of a PIT GEMM kernel.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement.\n\n"
      << "  --mode=<int>                Sets the GeMM mode: 0 for D=SxD and 1 for D=DxS.\n"
      << "  --M=<int>                   Sets the M dimension.\n"
      << "  --N=<int>                   Sets the N dimension.\n"
      << "  --K=<int>                   Sets the K dimension.\n"
      << "  --bx=<int>                  Sets the PIT block size x.\n"
      << "  --by=<int>                  Sets the PIT block size y.\n"
      << "  --sparsity=<f32>            Sets the sparse ratio.\n"
      << "  --alpha=<f32>               Epilogue scalar alpha (real part)\n"
      << "  --beta=<f32>                Epilogue scalar beta (real part)\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n";

    out << "\n\nExamples:\n\n"

      << "# Runs a 1024x1024x1024 PIT GEMM with 8x1 block size and sparse ratio of 0.5\n"
      << "$ ./examples/99_pit_gemm/99_pit_gemm --M=1024 --N=1024 --K=1024 --bx=8 --by=1 --sparsity=0.5\n\n";

    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemm>
class Testbed {
public:

  //
  // Type definitions
  //

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC = typename Gemm::ElementC;
  using ElementAccumulator = typename Gemm::ElementAccumulator;

  using EpilogueOutputOp = typename Gemm::GemmKernel::Epilogue::OutputOp;
  using ElementCompute = typename EpilogueOutputOp::ElementCompute;

  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;

  using MatrixCoord = typename LayoutC::TensorCoord;

private:

  //
  // Data members
  //

  Options options;

  cutlass::HostTensor<ElementA, LayoutA> tensor_a;
  cutlass::HostTensor<ElementB, LayoutB> tensor_b;
  cutlass::HostTensor<ElementC, LayoutC> tensor_c;
  cutlass::HostTensor<ElementC, LayoutC> tensor_d;

  cutlass::HostTensor<ElementC, LayoutC> reference_d;

  cutlass::HostTensor<int32_t, LayoutA> tensor_pit_idx;

public:

  Testbed(Options const &options_):
    options(options_) { }

private:

  /// Initializes data structures
  void initialize_() {
    tensor_a.resize(cutlass::make_Coord(options.M, options.K));
    tensor_b.resize(cutlass::make_Coord(options.K, options.N));
    tensor_c.resize(cutlass::make_Coord(options.M, options.N));
    tensor_d.resize(cutlass::make_Coord(options.M, options.N));

    reference_d.resize(cutlass::make_Coord(options.M, options.N));

    tensor_pit_idx.resize(cutlass::make_Coord(options.M / options.block_size_y,
                                              options.K / options.block_size_x));

    std::string data_folder = "examples/99_pit_gemm/data/";
    load_array_from_file<float, cutlass::half_t>(tensor_a.host_ref().data(), data_folder + "A.txt");
    load_array_from_file<float, cutlass::half_t>(tensor_b.host_ref().data(), data_folder + "B.txt");
    // load_array_from_file<float, cutlass::half_t>(tensor_c.host_ref().data(), data_folder + "C.txt");
    load_array_from_file<float, cutlass::half_t>(reference_d.host_ref().data(), data_folder + "C.txt");
    load_array_from_file<int, int>(tensor_pit_idx.host_ref().data(), data_folder + "idx.txt");
    // std::cout << int(tensor_pit_idx.host_ref().data()[0]) << " "
    //           << int(tensor_pit_idx.host_ref().data()[1]) << " "
    //           << int(tensor_pit_idx.host_ref().data()[2]) << " "
    //           << int(tensor_pit_idx.host_ref().data()[3]) << std::endl;

    tensor_a.sync_device();
    tensor_b.sync_device();
    tensor_c.sync_device();
    tensor_d.sync_device();
    tensor_pit_idx.sync_device();
  }

  /// Verifies the result is a GEMM
  bool verify_() {

    bool passed = true;

    tensor_d.sync_host();

    // Reference check
    passed = cutlass::reference::host::TensorEquals(tensor_d.host_view(), reference_d.host_view());

    if (!passed) {
      std::cerr << "\n***\nError - problem failed the QA check\n***\n" << std::endl;

      std::stringstream fname;

      fname << "error_99_pit_gemm_"
            << options.M << "x"
            << options.N << "x"
            << options.K << "_"
            << options.block_size_y << "x"
            << options.block_size_x << ".txt";

      std::cout << fname.str() << std::endl;

      std::ofstream results(fname.str());

      results
        << "\nA:\n" << tensor_a.host_view() << "\n"
        << "\nA Pit Index:\n" << tensor_pit_idx.host_view() << "\n"
        << "\nB:\n" << tensor_b.host_view() << "\n"
        << "\nC:\n" << tensor_c.host_view() << "\n"
        << "\nD reference:\n" << reference_d.host_view() << "\n"
        << "\nD computed:\n" << tensor_d.host_view() << "\n";
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

    int smem_size = int(sizeof(typename Gemm::GemmKernel::SharedStorage));

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

  /// Executes a PIT GeMM kernel and measures runtime.
  Result profile() {

    Result result;

    // Early exit
    if (!sufficient()) {
      std::cout << "Active CUDA device lacks hardware resources to run PIT GeMM kernel." << std::endl;
      return result;
    }

    result.passed = false;

    // Initialize the problem
    initialize_();

    // Configure the GEMM arguments
    typename EpilogueOutputOp::Params epilogue_op(options.alpha, options.beta);

    // Configure GEMM arguments
    typename Gemm::Arguments args(
      {options.M, options.N, options.K},
      tensor_a.device_ref(),
      tensor_b.device_ref(),
      tensor_c.device_ref(),
      tensor_d.device_ref(),
      tensor_pit_idx.device_data(),
      options.block_size_x,
      options.block_size_y,
      epilogue_op 
    );

    // Initialize the GEMM object
    Gemm gemm;

    result.status = gemm.initialize(args);

    if (result.status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to initialize PIT GeMM kernel." << std::endl;
      return result;
    }

    // Run the PIT GeMM object
    result.status = gemm.run();

    if (result.status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to run PIT GeMM kernel." << std::endl;
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
    result.passed = verify_();

    std::cout << "=========== Sync 1 ==========" << std::endl;
    result.error = cudaDeviceSynchronize();
    if (result.error != cudaSuccess)  {
      std::cerr << "Kernel execution error: " << cudaGetErrorString(result.error);
      return result;
    }
    std::cout << "=============================" << std::endl;

    //
    // Warm-up run
    //
    result.status = gemm.run();

    if (result.status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to run PIT GeMM kernel." << std::endl;
      return result;
    }

    std::cout << "=========== Sync 2 ==========" << std::endl;
    result.error = cudaDeviceSynchronize();
    if (result.error != cudaSuccess)  {
      std::cerr << "Kernel execution error: " << cudaGetErrorString(result.error);
      return result;
    }
    std::cout << "=============================" << std::endl;

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

    // Record an event at the start of a series of GEMM operations
    result.error = cudaEventRecord(events[0]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    //
    // Run profiling loop
    //

    std::cout << "=========== Sync 3 ==========" << std::endl;
    result.error = cudaDeviceSynchronize();
    if (result.error != cudaSuccess)  {
      std::cerr << "Kernel execution error: " << cudaGetErrorString(result.error);
      return result;
    }
    std::cout << "=============================" << std::endl;

    for (int iter = 0; iter < options.iterations; ++iter) {
      gemm();
    }

    std::cout << "=========== Sync 4 ==========" << std::endl;
    result.error = cudaDeviceSynchronize();
    if (result.error != cudaSuccess)  {
      std::cerr << "Kernel execution error: " << cudaGetErrorString(result.error);
      return result;
    }
    std::cout << "=============================" << std::endl;

    //
    // Stop profiling loop
    //

    // Record an event when the GEMM operations have been launched.
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
    std::cout << "PIT GeMM (CUTLASS):\n"
      << "====================================================" << std::endl;

    std::cout << std::endl;
    std::cout << "    " << "Runtime: " << result.runtime_ms << " ms" << std::endl;
    std::cout << "    " << " GFLOPs: " << result.gflops << std::endl;

    return result;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  //
  // This example uses mma.sync to directly access Tensor Cores to achieve peak performance.
  //

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
      << "PIT GeMM example requires a GPU of NVIDIA's Ampere Architecture or "
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

  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  constexpr int32_t kAlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  constexpr int32_t kAlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  constexpr int32_t kStages = 4;
  using Gemm = typename cutlass::gemm::device::PitGemm<
    ElementA, 
    LayoutA, 
    ElementB,
    LayoutB, 
    ElementOutput,
    LayoutC,
    ElementAccumulator, 
    cutlass::arch::OpClassTensorOp, 
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape, 
    InstructionShape,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 
    kStages, kAlignmentA, kAlignmentB>;

  //
  // Profile it
  //

  Testbed<Gemm> testbed(options);

  if (!testbed.sufficient()) {
    std::cout << "The active CUDA device lacks sufficient hardware resources to execute this kernel.\n";
    return 0;
  }

  Result result = testbed.profile();
  if (!result.passed) {
    std::cout << "\nFailed tp profile PIT GEMM.\n";
    return -1;
  }

  std::cout << "\nPassed\n";

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
