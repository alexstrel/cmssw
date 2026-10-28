#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>

#include <algorithm>
#include <ranges>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/VecArray.h"
#include "HeterogeneousCore/AlpakaInterface/interface/blas/alpakaReducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/blas/alpakaTransformer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/blas/alpakaTransformReduceKernel.h"

//#define USE_EXT_FLOAT
#define CHECK_ROUNDING_ERRORS

using namespace cms::alpakatools;
using namespace ALPAKA_ACCELERATOR_NAMESPACE;

int main() {
  // get the list of devices on the current platform
  auto const &devices = cms::alpakatools::devices<Platform>();

  if (devices.empty()) {
    std::cerr << "No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
      "the test will be skipped.\n";
    exit(EXIT_FAILURE);
  }

  static const Idx N = 3 * 524288;  //1572864

  static constexpr Idx nSrc = 2;

  std::cout << "N : " << N << std::endl;

  using DataType = float;

  for (auto const &device : devices) {
    std::cout << "Test nsrc kernels on " << alpaka::getName(device) << '\n';

    // Select specific devices
    auto const devHost = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0);

    auto computeQueue = Queue(
        device);  // TODO: cannot be const due to unsupported combination for memcpy (specialization does not exist in Alpaka?)

    if constexpr (std::is_same_v<decltype(device), alpaka::DevCpu>) {
      std::cout << "... skipped" << std::endl;
      return EXIT_SUCCESS;
    }

    // get device handle for this Acc
    auto const devAcc = alpaka::getDev(computeQueue);

    // Select queue
    using QueueProperty = alpaka::NonBlocking;
    using QueueAcc = std::remove_cvref_t<decltype(computeQueue)>;  //alpaka::Queue<Acc3D, QueueProperty>;

    // Define the 1D extent (dimensions : e.g., phys volume)
    Vec1D const extent(static_cast<Idx>(N));
    //
    using HostViewType = decltype(alpaka::createView(
        std::declval<decltype(devHost)>(), std::declval<DataType *>(), std::declval<Vec1D const &>()));
    // Input vector allocation and copy to device buffer
#ifndef CHECK_ROUNDING_ERRORS
    const DataType a = static_cast<DataType>(2);
#else
    const DataType a = static_cast<DataType>(0);
#endif
    std::vector<DataType> xv(extent.prod(), 0);
    //
    std::vector<DataType> yv(extent.prod(), 0);

    // Use increasing values as input
    //
    HostViewType xView = alpaka::createView(devHost, xv.data(), extent);  //create 2D view
    //
    HostViewType yView = alpaka::createView(devHost, yv.data(), extent);  //create 2D view

    // Input buffer at device
    using Buf_t = alpaka::Buf<Acc3D, DataType, Dim1D, Idx>;

    std::vector<Buf_t> xAcc;
    xAcc.reserve(nSrc);
    std::vector<Buf_t> yAcc;
    yAcc.reserve(nSrc);

    // Initialize random number generator
    constexpr std::uint32_t seed = 12345;
    std::mt19937 gen(seed);

    // Define a normal (Gaussian) distribution with mean 0 and standard deviation 1
    std::normal_distribution<double> distr(0.0, 1.0);
#ifndef CHECK_ROUNDING_ERRORS
    constexpr bool use_random_nums = true;
#else
    constexpr bool use_random_nums = false;
#endif

    for (Idx i = 0; i < nSrc; i++) {
      // populate the vector with random numbers
      if constexpr (use_random_nums) {
        std::generate(xv.begin(), xv.end(), [&]() { return distr(gen); });
        std::generate(yv.begin(), yv.end(), [&]() { return distr(gen); });
      } else {  // use int sequence (and shuffle)
#ifndef CHECK_ROUNDING_ERRORS
        std::iota(xv.begin(), xv.end(), 1.0f);
        std::iota(yv.begin(), yv.end(), 2.0f);

        std::shuffle(xv.begin(), xv.end(), gen);
        std::shuffle(yv.begin(), yv.end(), gen);
#else
        constexpr std::array triple_vals{1.0e8f, 1.0f, -1.0e8f};

        std::fill(xv.begin(), xv.end(), 0);

        std::ranges::transform(std::views::iota(std::size_t{0}, yv.size()), yv.begin(), [&](auto i) {
          return triple_vals[i % triple_vals.size()];
        });
#endif
      }

      xAcc.push_back(alpaka::allocBuf<DataType, Idx>(devAcc, extent));
      yAcc.push_back(alpaka::allocBuf<DataType, Idx>(devAcc, extent));

      auto const check_extent = alpaka::getExtents(xAcc[i]);
      printf("check extent for source %u: %u\n", i, check_extent.prod());

      alpaka::memcpy(computeQueue, xAcc[i], xView);
      alpaka::memcpy(computeQueue, yAcc[i], yView);

      double gres = 0.0;

      for (size_t j{0}; j < N; ++j) {
        yView[j] = a * xView[j] + yView[j];
#ifndef CHECK_ROUNDING_ERRORS
        auto const t = yView[j] * yView[j];
#else
        auto const t = yView[j];
#endif
        gres = gres + t;
      }

      std::cout << "Result on the host :: (nSrc " << i << " ) " << std::setprecision(16) << gres << std::endl;
    }
    std::cout << "Running alpaka transform_reduce kernel..." << std::endl;
#ifndef USE_EXT_FLOAT
    using Reduce_t = float;
#else
    using Reduce_t = extfloat::ExtendedFloat<float>;
#endif

    using reducer_t = cms::alpakatools::reduce::plus<Reduce_t>;
#ifndef CHECK_ROUNDING_ERRORS
    using transformer_t = cms::alpakatools::transform::axpynorm2_batched<Reduce_t, DataType, nSrc>;
#else
    using transformer_t = cms::alpakatools::transform::axpysum_batched<Reduce_t, DataType, nSrc>;
#endif
    using init_t = cms::alpakatools::reduce::Set<Reduce_t>;

    constexpr bool copy_to_host = true;

    auto policy = make_transform_reduce_policy<Acc3D, nSrc, copy_to_host, true, false>(computeQueue);

    std::cout << "Check policy " << policy.nSrc << std::endl;

    auto reduced_values = transform_reduce(policy, 0, N, xAcc, yAcc, init_t{}, reducer_t{}, transformer_t{a});

    bool is_correct = true;

    constexpr double tol = 5e-4;

    for (size_t i{0}; i < nSrc; ++i) {
      std::cout << "CHECK output : " << std::setprecision(16) << reduced_values[i] << std::endl;
      //
      alpaka::memcpy(computeQueue, yView, yAcc[i]);
      alpaka::wait(computeQueue);

      double host_nrm = 0.0;

      for (size_t j{0}; j < N; ++j) {  //dof
#ifndef CHECK_ROUNDING_ERRORS
        host_nrm += (yView[j] * yView[j]);
#else
        host_nrm += yView[j];
#endif
      }

      std::cout << "NORM on the host :: " << std::setprecision(16) << host_nrm << std::endl;
      if (abs(reduced_values[i] - host_nrm) / host_nrm > tol) {
        std::cout << "NORM diff :: " << std::setprecision(16) << abs(reduced_values[i] - host_nrm) / host_nrm
                  << std::endl;
        is_correct = false;
      }
    }
    //  Print results
    std::cout << "Multisrc reduction kernel.\n";
    std::cout << "Vector Size:" << N << "x" << ", src number:" << nSrc << "\n";

    if (is_correct)
      std::cout << "Sampled result checks passed.\n";
    else {
      std::cout << "Sampled result checks failed.\n";
      exit(EXIT_FAILURE);
    }
  }

  return EXIT_SUCCESS;
}
