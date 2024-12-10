#ifndef HeterogeneousCore_AlpakaInterface_interface_blas alpakaBlockReduction_h
#define HeterogeneousCore_AlpakaInterface_interface_blas_alpakaBlockReduction_h

#include "HeterogeneousCore/AlpakaInterface/interface/VecArray.h"

#include "HeterogeneousCore/AlpakaInterface/interface/blas/alpakaAtomicHelper.h"

namespace cms::alpakatools {
  namespace reduce {
    ALPAKA_FN_HOST_ACC inline unsigned int roundUpToPowerOf2(unsigned int N) {  //

      if (N <= 1) {
        return 2;
      }

      N--;

      N |= (N >> 1);
      N |= (N >> 2);
      N |= (N >> 4);
      N |= (N >> 8);
      N |= (N >> 16);

      return (N + 1);
    }

    ALPAKA_FN_HOST_ACC inline unsigned int getSubgoupMask(unsigned int subgroup_size) {  //

      unsigned int pw = (subgroup_size >> 1) & 0xF;

      if (1 & pw)
        return 0x3;
      else if (2 & pw)
        return 0xF;

      return 0xFF;
    }

    class WarpReducer {
    public:
#ifdef(__CUDA_ARCH__)
      static constexpr int default_warp_size = 32;  //has to be alpaka::warp::getSize(acc)
#elif defined(__HIP_DEVICE_COMPILE__)
      static constexpr int default_warp_size = 64;
#else
      static constexpr int default_warp_size = 1;
#endif

      WarpReducer() = default;

      template <typename TAcc, typename transform_reducer_t>
      ALPAKA_FN_ACC inline auto apply(TAcc const& acc,
                                      typename transform_reducer_t::reduce_t const& in,
                                      const transform_reducer_t f,
                                      bool all = true)
          -> std::enable_if_t<sizeof(typename transform_reducer_t::reduce_t) <= default_warp_size,
                              typename transform_reducer_t::reduce_t> {
        using reduce_t = typename transform_reducer_t::reduce_t;

        int const warpExtent = alpaka::warp::getSize(acc);

        auto r = f.get_reducer();

        reduce_t result = r.init(in);
        //
        for (unsigned int offset = warpExtent / 2; offset > 0; offset /= 2) {
          unsigned int const width = offset << 1;  //only part of warp is used
          result = r(result, alpaka::warp::shfl_down(acc, result, offset, width));
        }

        if (all)
          result = alpaka::warp::shfl(acc, result, 0, warpExtent);

        return result;
      }
      //
      template <typename TAcc, typename transform_reducer_t>
      ALPAKA_FN_ACC inline auto apply(TAcc const& acc,
                                      typename transform_reducer_t::reduce_t const& in,
                                      const transform_reducer_t f,
                                      bool all = true)
          -> std::enable_if_t<(sizeof(typename transform_reducer_t::reduce_t) > default_warp_size),
                              typename transform_reducer_t::reduce_t> {
        using reduce_t = typename transform_reducer_t::reduce_t;

        using atomic_t = typename cms::alpakatools::AtomicType<reduce_t>::type;

        constexpr std::size_t n = sizeof(reduce_t) / sizeof(atomic_t);

        atomic_t sum_tmp[n];

        memcpy(sum_tmp, &in, sizeof(reduce_t));

        CMS_UNROLL_LOOP
        for (std::size_t i = 0; i < n; i++)
          sum_tmp[i] = apply<TAcc, transform_reducer_t>(acc, sum_tmp[i], f, all);

        reduce_t result;

        memcpy(&result, sum_tmp, sizeof(reduce_t));

        return result;
      }
    };

    class BlockReducer {
    public:
      BlockReducer() = default;

      WarpReducer warp_reducer;

      template <typename TAcc, typename transform_reducer_t>
      ALPAKA_FN_ACC inline auto apply(TAcc const& acc,
                                      int const batch,
                                      typename transform_reducer_t::reduce_t const& in,
                                      const transform_reducer_t f,
                                      bool all = true) -> typename transform_reducer_t::reduce_t {
        using reduce_t = typename transform_reducer_t::reduce_t;
        int const warpExtent = alpaka::warp::getSize(acc);

        auto const blockExtent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);

        constexpr unsigned int max_w_items = 32;  //64

        int const w_items = std::min(static_cast<int>(blockExtent.prod() / warpExtent),
                                              static_cast<int>(max_w_items));
        //
        // Perform warp reduction using shuffle operations
        auto result = warp_reducer.template apply<TAcc, transform_reducer_t>(acc, in, f, all);

        if (all && w_items == 1)
          return result;  // short circuit for single warp CTA

        // we need to know block dimensionality to deduce the leading dimension
        constexpr auto nDim = alpaka::Dim<TAcc>::value;

        constexpr std::size_t leading_dim = nDim - 1;

        auto const threadIdx_x = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[leading_dim];
        auto const blockDim_x = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[leading_dim];

        int const warpIdx_x = threadIdx_x / warpExtent;
        int const laneIdx = threadIdx_x % warpExtent;

        int const w_items_x =
            std::min(static_cast<int>(blockDim_x / warpExtent), static_cast<int>(max_w_items));

        auto& sdata(alpaka::declareSharedVar<cms::alpakatools::VecArray<reduce_t, max_w_items>, __COUNTER__>(acc));

        // Write the reduced sum of each warp to shared memory
        if (laneIdx == 0)
          sdata[warpIdx_x + batch * w_items_x] = result;

        alpaka::syncBlockThreads(acc);

        // Reduce the results from all warps (assuming blockDim.x / warpSize warps per block)
        auto const w_items_t = roundUpToPowerOf2(w_items_x);
        //
        if (threadIdx_x < std::max(static_cast<uint32_t>(warpExtent), w_items_t)) {
          result = threadIdx_x < w_items_x ? sdata[threadIdx_x + batch * w_items_x] : f.init();

          result = warp_reducer.template apply<TAcc, transform_reducer_t>(acc, result, f, all);
        }

        alpaka::syncBlockThreads(acc);

        return result;
      }
    };

  }  // namespace reduce
}  // namespace cms::alpakatools

#endif
