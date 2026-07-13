#ifndef HeterogeneousCore_AlpakaInterface_interface_blas_BlasHelpers_h
#define HeterogeneousCore_AlpakaInterface_interface_blas_BlasHelpers_h

/**
 * @file PFMultiDepthClusterizerHelper.h
 * @brief Warp-level utility functions for particle flow multi-depth clustering.
 * 
 * This header provides basic warp-synchronous operations used in clustering algorithms,
 * including bitwise manipulations (least/most significant set bits) and masked
 * warp-exclusive sum computations.
 */

#include "HeterogeneousCore/AlpakaInterface/interface/blas/WarpIntrinsics.h"

#include <concepts>
#include <type_traits>


namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;
  using namespace cms::alpakablasintrinsics;

  template <typename reducer_t, typename reduce_t, std::size_t N>
  concept VecArrayReducer = requires(reducer_t const& f,
                                     cms::alpakatools::VecArray<reduce_t, N> const& x,
                                     cms::alpakatools::VecArray<reduce_t, N> const& y) {
    { f(x, y) } -> std::same_as<cms::alpakatools::VecArray<reduce_t, N>>;
  };

  /**
 * @brief Compute warp size
 *
 * @param mask Input lane index in the warp
 * 
 * @return compute lane mask:
 */
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool is_full_warp(TAcc const& acc) {
    using warp_mask_t = decltype(alpaka::warp::activemask(acc));

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    constexpr std::uint32_t w_extent = alpaka::warp::getSizeCompileTime<TAcc>();

    warp_mask_t const mask = alpaka::warp::activemask(acc);

    std::uint32_t const nActiveLanes = alpaka::popcount(acc, mask);

    return nActiveLanes == w_extent;
#endif
    return true;
  }

  /**
 * @brief Compute warp size
 *
 * @param mask Input lane index in the warp
 * 
 * @return compute lane mask:
 */
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE constexpr std::int32_t get_warp_size() {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return alpaka::warp::getSizeCompileTime<TAcc>();
#else
    return 1;
#endif
  }

  /**
 * @brief Compute lane mask
 *
 * @param mask Input lane index in the warp
 * 
 * @return compute lane mask:
 */
  template <alpaka::concepts::Acc TAcc, typename TMask = decltype(alpaka::warp::activemask(std::declval<TAcc&>()))>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE constexpr auto get_lane_mask(const std::uint32_t lane_idx) {
    //using warp_mask_t = decltype(alpaka::warp::activemask(std::declval<TAcc&>()));
    return static_cast<TMask>(1U) << lane_idx;
  }

  /**
 * @brief Check that given lane is active in the custom lane mask
 *
 * @param mask Input mask.
 * @param mask Input lane index in the warp
 * 
 * @return True if active, otherwise false.
 */
  template <alpaka::concepts::Acc TAcc, typename TMask = decltype(alpaka::warp::activemask(std::declval<TAcc&>()))>
  constexpr auto is_work_lane(TMask const work_mask, std::uint32_t const lane_idx) -> bool {
    return ((work_mask >> lane_idx) & 1);
  }

  /**
 * @brief Returns the position of the least significant set bit in a mask.
 *
 * @tparam TAcc Alpaka accelerator type.
 * 
 * @param acc   Alpaka accelerator instance.
 * @param mask  Input mask.
 * 
 * @return Index of least significant 1 bit (0-based). (or warp size if x == 0).
 */
  template <alpaka::concepts::Acc TAcc, typename TMask = decltype(alpaka::warp::activemask(std::declval<TAcc&>()))>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE auto get_ls1b_idx(TAcc const& acc, TMask const mask) -> TMask {
    if (mask == 0)
      return static_cast<TMask>(get_warp_size<TAcc>());

    if constexpr (std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevCpu>)
      return 0;

    using signed_TMask = std::make_signed_t<TMask>;

    const auto pos = alpaka::ffs(acc, static_cast<signed_TMask>(mask));

    return static_cast<TMask>(pos - 1);
  }

  /**
 * @brief Returns true if a given lane is represented in the least significant bit in a mask.
 *
 * @tparam TAcc Alpaka accelerator type.
 * 
 * @param acc   Alpaka accelerator instance.
 * @param mask  Input mask.
 * @param lane_idx Current thread's lane index.
 * 
 * @return True if lane_idx is the least significant bit in a mask, otherwise faulse .
 */
  template <alpaka::concepts::Acc TAcc, typename TMask = decltype(alpaka::warp::activemask(std::declval<TAcc&>()))>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE auto is_ls1b_idx(TMask const mask, std::uint32_t const lane_idx) -> bool {
    if (mask == 0)
      return false;

    if constexpr (std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevCpu>)
      return true;

    const TMask lane_mask = get_lane_mask<TAcc>(lane_idx);

    // First check wether the lane is represented at all, otherwise check the trivial case:
    if ((mask & lane_mask) == 0)
      return false;
    else if (lane_idx == 0)
      return true;

    constexpr std::uint32_t w_extent = get_warp_size<TAcc>();

    const TMask inverted_mask = mask << (w_extent - lane_idx);

    return (inverted_mask == 0);
  }

  /**
 * @brief Performs warp-level exclusive prefix sum
 *
 * @tparam TAcc Alpaka accelerator type.
 * @tparam accum If true, broadcast total accumulated value to lowest active lane.
 * 
 * @param acc   Alpaka accelerator instance.
 * @param val   Value to include in the prefix sum.
 * @param lane_idx Current thread's lane index.
 * 
 * @return Exclusive prefix sum value for the current lane.
 * convention used here:
 * - lanes 1..(w_extent-1) receive the exclusive prefix sum (CSR offsets within the warp),
 * - lane 0 receives the total sum over the warp (used as the per-warp NNZ aggregate)
 */

  template <alpaka::concepts::Acc TAcc, bool all = true>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE auto warp_exclusive_sum(TAcc const& acc,
                                                         std::uint32_t const val,
                                                         std::uint32_t const lane_idx) -> std::uint32_t {
    if constexpr (std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevCpu>)
      return all ? val : 0;

    using warp_mask_t = decltype(alpaka::warp::activemask(acc));

    constexpr std::uint32_t w_extent = get_warp_size<TAcc>();

    std::uint32_t local_offset = val;

    // Do inclusive sum first:
    CMS_UNROLL_LOOP
    for (std::uint32_t step = 1; step < w_extent; step *= 2) {
      const auto res = alpaka::warp::shfl_up(acc, local_offset, step, w_extent);
      if (lane_idx >= step)
        local_offset += res;
    }

    if constexpr (all) {
      const std::uint32_t high_lane_idx = w_extent - 1;

      if (lane_idx == 0 || lane_idx == high_lane_idx) {
        // send last lane value (total tile offset) to lane idx = low_lane_idx:
        const warp_mask_t active_mask = static_cast<warp_mask_t>(1) | get_lane_mask<TAcc>(high_lane_idx);
        const std::uint32_t tmp = warp::shfl_mask(acc, active_mask, local_offset, high_lane_idx, w_extent);

        local_offset = tmp;  //lane 0 keeps full (inclusive for the last lane) sum, nop for high_lane_idx
      }
    }
    return lane_idx == 0 ? local_offset : local_offset - val;  //we return exclusive sum!
  }

  /**
 * @brief Returns logical index for a given physical lane index based on custom lane mask.
 *
 * @tparam TAcc Alpaka accelerator type.
 * 
 * @param acc Alpaka accelerator instance.
 * @param mask Input bitmask.
 * @param lane_idx imput phys. lane index
 * 
 * @return Index of the lane in the mask 
 */

  template <alpaka::concepts::Acc TAcc, typename TMask = decltype(alpaka::warp::activemask(std::declval<TAcc&>()))>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE auto get_logical_lane_idx(TAcc const& acc,
                                                           TMask const mask,
                                                           std::uint32_t const lane_idx) -> std::uint32_t {
    if (lane_idx == 0)
      return lane_idx;  // nothing to do, phys idx coincide with the logical one.
    const TMask lane_mask = mask & (get_lane_mask<TAcc>(lane_idx) - 1);
    return alpaka::popcount(acc, lane_mask);  // Count 1s below current lane
  }

  /**
 * @brief Returns physical lane index for a given logical lane index based on custom lane mask.
 *
 * @tparam TAcc Alpaka accelerator type.
 * 
 * @param acc Alpaka accelerator instance.
 * @param mask Input mask.
 * @param logical_lane_idx input logical lane index
 * 
 * @return Physical index of the lane in the mask 
 */

  template <alpaka::concepts::Acc TAcc, typename TMask = decltype(alpaka::warp::activemask(std::declval<TAcc&>()))>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE auto get_physical_lane_idx(TAcc const& acc,
                                                            TMask const mask,
                                                            std::int32_t logical_lane_idx) -> std::uint32_t {
    using signed_TMask = std::make_signed_t<TMask>;

    if constexpr (std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevCpu>)
      return 0;

    signed_TMask m = mask;

    while (logical_lane_idx--)
      m &= (m - 1);

    const auto pos = alpaka::ffs(acc, m);

    return static_cast<std::uint32_t>(pos - 1);
  }

  /**
 * @brief generic warp reduction
 *
 * @tparam TAcc Alpaka accelerator type.
 * 
 * @param acc Alpaka accelerator instance.
 * @param in input value to reduce
 * @param f reducer 
 * 
 * @return return reduced value (propagated to all lanes in the mask by default)
 */

  template <alpaka::concepts::Acc TAcc, typename reduce_t, typename reducer_t, bool all = true>
    requires std::is_arithmetic_v<reduce_t>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE auto warp_reduce(TAcc const& acc, reduce_t const in, const reducer_t f) -> reduce_t {
    constexpr unsigned int w_extent = get_warp_size<TAcc>();

    reduce_t result = in;

    if constexpr (std::is_same_v<Device, alpaka::DevCpu>)
      return result;

    CMS_UNROLL_LOOP
    for (unsigned int offset = w_extent / 2; offset > 0; offset /= 2) {
      result = f(result, alpaka::warp::shfl_down(acc, result, offset, w_extent));
    }

    if constexpr (all)
      result = alpaka::warp::shfl(acc, result, 0, w_extent);

    return result;
  }

  template <alpaka::concepts::Acc TAcc, typename reduce_t, int N, typename reducer_t, bool all = true>
    requires(VecArrayReducer<reducer_t, reduce_t, N> && std::is_arithmetic_v<reduce_t>)
  ALPAKA_FN_ACC ALPAKA_FN_INLINE auto warp_reduce(TAcc const& acc,
                                                  cms::alpakatools::VecArray<reduce_t, N> const& in,
                                                  const reducer_t f) -> cms::alpakatools::VecArray<reduce_t, N> {
    constexpr unsigned int w_extent = get_warp_size<TAcc>();

    cms::alpakatools::VecArray<reduce_t, N> result = in;

    if constexpr (std::is_same_v<Device, alpaka::DevCpu>)
      return result;

    CMS_UNROLL_LOOP
    for (unsigned int offset = w_extent / 2; offset > 0; offset /= 2) {
      cms::alpakatools::VecArray<reduce_t, N> tmp{};
      CMS_UNROLL_LOOP
      for (int i = 0; i < N; i++) {
        tmp[i] = alpaka::warp::shfl_down(acc, result[i], offset, w_extent);
      }
      result = f(result, tmp);
    }

    if constexpr (all)
      CMS_UNROLL_LOOP
    for (int i = 0; i < N; i++) {
      result[i] = alpaka::warp::shfl(acc, result[i], 0, w_extent);
    }

    return result;
  }

  /**
 * @brief Sparse warp reduction
 *
 * @tparam TAcc Alpaka accelerator type.
 * 
 * @param acc Alpaka accelerator instance
 * @param mask input mask 
 * @param in input value to reduce
 * @param f reducer 
 * 
 * @return return reduced value (propagated to all lanes in the mask by default)
 */

  template <alpaka::concepts::Acc TAcc,
            typename reduce_t,
            typename reducer_t,
            typename TMask = decltype(alpaka::warp::activemask(std::declval<TAcc&>())),
            bool all = true>
    requires std::is_arithmetic_v<reduce_t>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE auto warp_sparse_reduce(
      TAcc const& acc, TMask const mask, std::uint32_t const lane_idx, reduce_t const in, const reducer_t f)
      -> reduce_t {
    constexpr std::uint32_t w_extent = get_warp_size<TAcc>();

    if constexpr (std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevCpu>)
      return mask == 0 ? 0 : in;

    // Non-active lanes must skip the reduction:
    if (is_work_lane<TAcc>(mask, lane_idx) == false) {
      return in;
    }

    std::uint32_t nActiveLanes = alpaka::popcount(acc, mask);  // count number of active lanes

    // First check if this is just a single active lane in the warp:
    if (nActiveLanes == 1)
      return in;

    //Compute the next power of two:
    const std::uint32_t pow2 = w_extent - cms::alpakablasintrinsics::clz(acc, nActiveLanes - 1);
    const std::uint32_t pow2_boundary = 1 << pow2;

    const std::uint32_t logical_lane_idx = get_logical_lane_idx<TAcc>(acc, mask, lane_idx);

    reduce_t res = in;

    CMS_UNROLL_LOOP
    for (std::uint32_t offset = pow2_boundary / 2; offset > 0; offset /= 2) {
      const std::uint32_t logical_src_lane_idx = logical_lane_idx + offset;
      const std::uint32_t src_lane_idx = (logical_src_lane_idx < nActiveLanes)
                                             ? get_physical_lane_idx<TAcc>(acc, mask, logical_src_lane_idx)
                                             : lane_idx;

      const reduce_t neigh_res = warp::shfl_mask(acc, mask, res, src_lane_idx, w_extent);

      if (logical_src_lane_idx < nActiveLanes)
        res = f(res, neigh_res);
    }

    if constexpr (all) {
      const auto low_lane_idx = get_physical_lane_idx(acc, mask, 0);
      res = warp::shfl_mask(acc, mask, res, low_lane_idx, w_extent);
    }

    return res;
  }

  template <alpaka::concepts::Acc TAcc,
            typename reduce_t,
            int N,
            typename reducer_t,
            typename TMask = decltype(alpaka::warp::activemask(std::declval<TAcc&>())),
            bool all = true>
    requires(VecArrayReducer<reducer_t, reduce_t, N> && std::is_arithmetic_v<reduce_t>)
  ALPAKA_FN_ACC ALPAKA_FN_INLINE auto warp_sparse_reduce(TAcc const& acc,
                                                         TMask const mask,
                                                         std::uint32_t const lane_idx,
                                                         cms::alpakatools::VecArray<reduce_t, N> const& in,
                                                         const reducer_t f) -> cms::alpakatools::VecArray<reduce_t, N> {
    constexpr std::uint32_t w_extent = get_warp_size<TAcc>();

    if constexpr (std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevCpu>)
      return mask == 0 ? 0 : in;

    // Non-active lanes must skip the reduction:
    if (is_work_lane<TAcc>(mask, lane_idx) == false) {
      return in;
    }

    std::uint32_t nActiveLanes = alpaka::popcount(acc, mask);  // count number of active lanes

    // First check if this is just a single active lane in the warp:
    if (nActiveLanes == 1)
      return in;

    //Compute the next power of two:
    const std::uint32_t pow2 = w_extent - cms::alpakablasintrinsics::clz(acc, nActiveLanes - 1);
    const std::uint32_t pow2_boundary = 1 << pow2;

    const std::uint32_t logical_lane_idx = get_logical_lane_idx<TAcc>(acc, mask, lane_idx);

    cms::alpakatools::VecArray<reduce_t, N> res = in;

    CMS_UNROLL_LOOP
    for (std::uint32_t offset = pow2_boundary / 2; offset > 0; offset /= 2) {
      const std::uint32_t logical_src_lane_idx = logical_lane_idx + offset;
      const std::uint32_t src_lane_idx = (logical_src_lane_idx < nActiveLanes)
                                             ? get_physical_lane_idx<TAcc>(acc, mask, logical_src_lane_idx)
                                             : lane_idx;

      cms::alpakatools::VecArray<reduce_t, N> neigh_res{};
      CMS_UNROLL_LOOP
      for (int i = 0; i < N; i++) {
        neigh_res[i] = warp::shfl_mask(acc, mask, res, src_lane_idx, w_extent);
      }

      if (logical_src_lane_idx < nActiveLanes)
        res = f(res, neigh_res);
    }

    if constexpr (all) {
      const auto low_lane_idx = get_physical_lane_idx(acc, mask, 0);
      CMS_UNROLL_LOOP
      for (int i = 0; i < N; i++) {
        res[i] = alpaka::warp::shfl(acc, mask, res, low_lane_idx, w_extent);
      }
    }

    return res;
  }

  /**
 * @brief Performs warp-level sparse exclusive prefix sum (masked version of warp_exclusive_sum, see above )
 *
 * @tparam TAcc Alpaka accelerator type.
 * @tparam accum If true, broadcast total accumulated value to lowest active lane.
 * 
 * @param acc   Alpaka accelerator instance.
 * @param mask  input mask 
 * @param val   Value to include in the prefix sum.
 * @param lane_idx Current thread's lane index.
 * 
 * @return Exclusive prefix sum value for the current lane.
 */

  template <alpaka::concepts::Acc TAcc,
            typename TMask = decltype(alpaka::warp::activemask(std::declval<TAcc&>())),
            bool all = true>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE auto warp_sparse_exclusive_sum(TAcc const& acc,
                                                                TMask const mask,
                                                                std::uint32_t const val,
                                                                std::uint32_t const lane_idx) -> std::uint32_t {
    constexpr std::uint32_t w_extent = get_warp_size<TAcc>();

    if constexpr (std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevCpu>)
      return all == false ? 0 : (mask == 0 ? 0 : val);

    // Non-active lanes should skip the reduction:
    if (is_work_lane<TAcc>(mask, lane_idx) == false)
      return 0;

    // count number of active lanes
    const std::uint32_t nActiveLanes = alpaka::popcount(acc, mask);
    // First check if this is just a single active lane in the warp:
    if (nActiveLanes == 1)
      return val;  //nothing to do, note that this is the inclusive "sum": low lane always keeps the whole sum

    //Compute the next power of two:
    const std::uint32_t pow2 = w_extent - cms::alpakablasintrinsics::clz(acc, nActiveLanes - 1);
    const std::uint32_t pow2_boundary = 1 << pow2;

    const std::uint32_t logical_lane_idx = get_logical_lane_idx<TAcc>(acc, mask, lane_idx);

    std::uint32_t local_offset = val;

    for (std::uint32_t step = 1; step < pow2_boundary; step *= 2) {
      const std::uint32_t src_lane_idx =
          (logical_lane_idx >= step) ? get_physical_lane_idx<TAcc>(acc, mask, logical_lane_idx - step) : lane_idx;
      const std::uint32_t tmp_val = warp::shfl_mask(acc, mask, local_offset, src_lane_idx, w_extent);

      if (logical_lane_idx >= step)
        local_offset += tmp_val;
    }

    if constexpr (all) {
      const std::uint32_t high_lane_idx = get_physical_lane_idx<TAcc>(acc, mask, nActiveLanes - 1);
      // send last lane value (total tile offset) to lane idx = low_lane_idx:
      const std::uint32_t tmp = warp::shfl_mask(acc, mask, local_offset, high_lane_idx, w_extent);

      if (logical_lane_idx == 0)
        local_offset = tmp;  //lane 0 keeps full (inclusive for the last lane) sum
    }
    return logical_lane_idx == 0
               ? local_offset
               : local_offset - val;  //we return exclusive sum, except zero logical lane (which returns total offset)
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
