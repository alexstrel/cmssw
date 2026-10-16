#ifndef HeterogeneousCore_AlpakaInterface_interface_blas_WarpIntrinsics_h
#define HeterogeneousCore_AlpakaInterface_interface_blas_WarpIntrinsics_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include <type_traits>
#include <concepts>

namespace cms::alpakatools::blas {
  namespace warp {

#ifdef __HIP_DEVICE_COMPILE__

#if !((HIP_VERSION_MAJOR >= 7) || (HIP_VERSION >= 60200000 && defined(HIP_ENABLE_WARP_SYNC_BUILTINS)))
#warning "HIP Version is not supported."
#endif

#endif

    // Thread local intrinsics:
    //! Computes x * y in round mode
    //! \tparam T The type of the args specializing add.
    //! \param x The first argument.
    //! \param y The second argument.
    //! Remark : the default rounding mode is RN, so use this intrinsics
    //!          for numerically controlled algorithms (like Kahan algo)
    constexpr auto mul_rn(float const x, float const y) -> float {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      return __fmul_rn(x, y);
#else
      return (x * y);
#endif
    }

    constexpr auto mul_rn(double const x, double const y) -> double {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      return __dmul_rn(x, y);
#else
      return (x * y);
#endif
    }

    // Thread local intrinsics:
    //! Computes x * y + z in round mode
    //! \tparam T The type of the args specializing add.
    //! \param x The first argument.
    //! \param y The second argument.
    //! \param z The third argument.
    //! Remark : the default rounding mode is RN, so use this intrinsics
    //!          for numerically controlled algorithms (like Kahan algo)
    constexpr auto fma_rn(float const x, float const y, float const z) -> float {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      return __fmaf_ieee_rn(x, y, z);
#else
      return std::fma(x, y, z);
#endif
    }

    constexpr auto fma_rn(double const x, double const y, double const z) -> double {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      return __fma_rn(x, y, z);
#else
      return std::fma(x, y, z);
#endif
    }

    // Thread local intrinsics:
    //! Computes x + y in round mode
    //! \tparam T The type of the args specializing add.
    //! \param x The first argument.
    //! \param y The second argument.
    //! Remark : the default rounding mode is RN, so use this intrinsics
    //!          for numerically controlled algorithms (like Kahan algo)
    constexpr auto add_rn(float const x, float const y) -> float {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      return __fadd_rn(x, y);
#else
      return (x + y);
#endif
    }

    constexpr auto add_rn(double const x, double const y) -> double {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      return __dadd_rn(x, y);
#else
      return (x + y);
#endif
    }

    //! Computes x - y in round mode
    //! \tparam T The type of the args specializing sub.
    //! \param x The first argument.
    //! \param y The second argument.
    //! Remark : the default rounding mode is RN, so use this intrinsics
    //!          for numerically controlled algorithms (like Kahan algo)
    constexpr auto sub_rn(float const x, float const y) -> float {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      return __fsub_rn(x, y);
#else
      return (x - y);
#endif
    }

    constexpr auto sub_rn(double const x, double const y) -> double {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      return __dsub_rn(x, y);
#else
      return (x - y);
#endif
    }

    /**
 * @brief Synchronize all threads within a subset of lanes in the warp
 *
 * @tparam TAcc Alpaka accelerator type.
 * 
 * @param acc   Alpaka accelerator instance.
 * @param mask Input mask.
 */

    template <alpaka::concepts::Acc TAcc, typename TMask = decltype(alpaka::warp::activemask(std::declval<TAcc&>()))>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void syncWarpThreads_mask(TAcc const& acc, TMask const mask) {
      if (mask == 0)
        return;  //early return for the trivial mask

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      // Alpaka CUDA/HIP backend
      __syncwarp(mask);  // Synchronize all threads within a subset of lanes in the warp
#endif
      // No-op for CPU accelerators
    }

    /**
 * @brief Warp-wide ballot of a predicate, restricted to a given active-lane mask.
 *
 * Computes a warp mask containing the lanes for which 'pred' is non-zero,
 * considering only lanes enabled in 'mask'. 
 *
 * @tparam TAcc Alpaka accelerator type. 
 *
 * @param acc  Alpaka accelerator instance.
 * @param mask Active-lane mask defining which lanes participate in the ballot.
 * @param pred Per-lane predicate value; non-zero counts as 'true'.
 *
 * @return A warp mask with bits set for participating lanes (as defined by 'mask')
 *         whose 'pred' evaluates to 'true'.
 */
    template <alpaka::concepts::Acc TAcc, typename TMask = decltype(alpaka::warp::activemask(std::declval<TAcc&>()))>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE auto ballot_mask(TAcc const& acc, TMask const mask, int pred) -> TMask {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      return __ballot_sync(mask, pred);
#else
      return pred == 0 ? static_mask<TMask>(0) : mask;
#endif
    }
    /**
 * @brief Masked warp shuffle from a source lane.
 *
 * @tparam TAcc Alpaka accelerator type. 
 * @tparam T    Value type to be shuffled.
 *
 * @param acc     Alpaka accelerator instance.
 * @param mask    Active-lane mask for the shuffle operation.
 * @param var     Per-lane input value.
 * @param srcLane Source lane index within the shuffle width.
 * @param width   Logical warp width for the shuffle. 
 *
 * @return The value of 'var' from lane 'srcLane', or an unspecified value if
 *         the source lane is inactive (i.e., not set in 'mask' ).
 */
    template <alpaka::concepts::Acc TAcc,
              typename T,
              typename TMask = decltype(alpaka::warp::activemask(std::declval<TAcc&>()))>
      requires std::is_arithmetic_v<T>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE auto shfl_mask(TAcc const& acc, TMask const mask, T var, int srcLane, int width)
        -> T {
      T res{};
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      res = __shfl_sync(mask, var, srcLane, width);  // Synchronize all threads within a warp
#else
      res = var;
#endif

      return res;
    }

    /**
 * @brief Masked warp shuffle-down operation.
 *
 * @tparam TAcc Alpaka accelerator type. 
 * @tparam T    Value type to be shuffled.
 *
 * @param acc     Alpaka accelerator instance.
 * @param mask    Active-lane mask for the shuffle operation.
 * @param var     Per-lane input value.
 * @param srcLane Lane offset (delta) below the calling lane.
 * @param width   Logical warp width for the shuffle.
 *
 * @return The value of 'var' from the source lane, or an unspecified value if
 *         the source lane is inactive or out of range.
 */

    template <alpaka::concepts::Acc TAcc,
              typename T,
              typename TMask = decltype(alpaka::warp::activemask(std::declval<TAcc&>()))>
      requires std::is_arithmetic_v<T>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE auto shfl_down_mask(TAcc const& acc, TMask const mask, T var, int srcLane, int width)
        -> T {
      T res{};
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      res = __shfl_down_sync(mask, var, srcLane, width);
#else
      res = var;
#endif
      return res;
    }

    /**
 * @brief Masked warp shuffle-up operation.
 *
 * @tparam TAcc Alpaka accelerator type. 
 * @tparam T    Value type to be shuffled.
 *
 * @param acc     Alpaka accelerator instance.
 * @param mask    Active-lane mask for the shuffle operation.
 * @param var     Per-lane input value.
 * @param srcLane Lane offset (delta) above the calling lane.
 * @param width   Logical warp width for the shuffle.
 *
 * @return The value of 'var' from the source lane, or an unspecified value if
 *         the source lane is inactive or out of range.
 */

    template <alpaka::concepts::Acc TAcc,
              typename T,
              typename TMask = decltype(alpaka::warp::activemask(std::declval<TAcc&>()))>
      requires std::is_arithmetic_v<T>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE auto shfl_up_mask(TAcc const& acc, TMask const mask, T var, int srcLane, int width)
        -> T {
      T res{};
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      res = __shfl_up_sync(mask, var, srcLane, width);
#else
      res = var;
#endif
      return res;
    }

    /**
 * @brief Masked warp-wide match-any operation.
 *
 * @tparam TAcc Alpaka accelerator type. 
 * @tparam T    Value type used for the match comparison.
 *
 * @param acc  Alpaka accelerator instance.
 * @param mask Active-lane mask for the match operation.
 * @param val  Per-lane value to be compared across the warp.
 *
 * @return A warp mask with bits set for lanes (enabled in 'mask') whose
 *         'val' equals the calling lane's value.
 */
    template <alpaka::concepts::Acc TAcc,
              typename T,
              typename TMask = decltype(alpaka::warp::activemask(std::declval<TAcc&>()))>
      requires std::is_arithmetic_v<T>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE auto match_any_mask(TAcc const& acc, TMask const mask, T val) -> TMask {
      TMask res{};
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)

#if __CUDA_ARCH__ >= 700 || ALPAKA_ACC_GPU_HIP_ENABLED
      res = __match_any_sync(mask, val);
#else
      constexpr unsigned int w_extent = alpaka::warp::getSizeCompileTime<TAcc>();
      unsigned int match = 0;
      for (int iter_lane_idx = 0; iter_lane_idx < w_extent; ++iter_lane_idx) {
        T iter_val = __shfl_sync(mask, val, iter_lane_idx, w_extent);
        const unsigned int iter_lane_mask = 1 << iter_lane_idx;
        if (iter_val == val)
          match |= iter_lane_mask;
      }
      res = match & mask;
#endif
#else
      res = mask;
#endif
      return res;
    }

  }  // namespace warp

  /**
 * @brief Reverse the bit order of a warp mask.
 *
 * @tparam TAcc Alpaka accelerator type. 
 *
 * @param acc  Alpaka accelerator instance.
 * @param mask Input warp mask whose bit order is to be reversed.
 *
 * @return A warp mask with 32/64 bits reversed.
 */

  template <alpaka::concepts::Acc TAcc, typename TMask = decltype(alpaka::warp::activemask(std::declval<TAcc&>()))>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE auto brev(TAcc const& acc, TMask const mask) -> TMask {
    TMask res{0};
#if defined(__CUDA_ARCH__)
    res = __brev(mask);
#elif defined(__HIP_DEVICE_COMPILE__)
    res = __brevll(mask);
#else
    res = mask;
#endif
    return res;
  }

  /**
 * @brief Count leading zeros in a warp mask.
 *
 * @tparam TAcc Alpaka accelerator type. 
 *
 * @param acc  Alpaka accelerator instance.
 * @param mask Input warp mask.
 *
 * @return The number of leading zero bits in the lower 32/64 bits of 'mask'.
 */
  template <alpaka::concepts::Acc TAcc, typename MaskType = decltype(alpaka::warp::activemask(std::declval<TAcc&>()))>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE auto clz(TAcc const& acc, MaskType const mask) -> uint32_t {
    uint32_t res{0};
#if defined(__CUDA_ARCH__)
    res = __clz(mask);
#elif defined(__HIP_DEVICE_COMPILE__)
    res = __clzll(mask);
#else
    res = mask == 0 ? 1 : 0;
#endif
    return res;
  }

}  // namespace cms::alpakatools::blas
#endif
