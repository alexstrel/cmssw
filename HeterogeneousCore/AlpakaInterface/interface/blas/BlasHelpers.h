#ifndef HeterogeneousCore_AlpakaInterface_interface_blas_BlasHelpers_h
#define HeterogeneousCore_AlpakaInterface_interface_blas_BlasHelpers_h

#include "HeterogeneousCore/AlpakaInterface/interface/blas/WarpIntrinsics.h"

#include <concepts>
#include <type_traits>

using namespace cms::alpakatools;
using namespace cms::alpakatools::blas;

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
  requires(std::is_same_v<alpaka::AccToTag<TAcc>, alpaka::TagGpuCudaRt> ||
           std::is_same_v<alpaka::AccToTag<TAcc>, alpaka::TagGpuHipRt>)
ALPAKA_FN_ACC ALPAKA_FN_INLINE bool isFullWarp(TAcc const& acc) {
  using warp_mask_t = decltype(alpaka::warp::activemask(acc));

  constexpr std::uint32_t w_extent = alpaka::warp::getSizeCompileTime<TAcc>();

  warp_mask_t const mask = alpaka::warp::activemask(acc);

  std::uint32_t const nActiveLanes = alpaka::popcount(acc, mask);

  return nActiveLanes == w_extent;
}

template <alpaka::concepts::Acc TAcc>
  requires std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevCpu>
ALPAKA_FN_ACC ALPAKA_FN_INLINE bool isFullWarp(TAcc const& acc) {
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
  requires(std::is_same_v<alpaka::AccToTag<TAcc>, alpaka::TagGpuCudaRt> ||
           std::is_same_v<alpaka::AccToTag<TAcc>, alpaka::TagGpuHipRt>)
ALPAKA_FN_ACC ALPAKA_FN_INLINE constexpr std::int32_t getWarpSize() {
  return alpaka::warp::getSizeCompileTime<TAcc>();
}

template <alpaka::concepts::Acc TAcc>
  requires std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevCpu>
ALPAKA_FN_ACC ALPAKA_FN_INLINE constexpr std::int32_t getWarpSize() {
  return 1;
}

/**
 * @brief Check that given lane is active in the custom lane mask
 *
 * @param mask Input mask.
 * @param mask Input lane index in the warp
 * 
 * @return True if active, otherwise false.
 */
template <alpaka::concepts::Acc TAcc>
constexpr auto isWorkLane(decltype(alpaka::warp::activemask(std::declval<TAcc&>())) const work_mask,
                          std::uint32_t const lane_idx) -> bool {
  return ((work_mask >> lane_idx) & 1);
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
template <alpaka::concepts::Acc TAcc>
ALPAKA_FN_ACC ALPAKA_FN_INLINE auto getLogicalLaneIdx(TAcc const& acc,
                                                      decltype(alpaka::warp::activemask(std::declval<TAcc&>()))
                                                          const mask,
                                                      std::uint32_t const lane_idx) -> std::uint32_t {
  if (lane_idx == 0)
    return lane_idx;  // nothing to do, phys idx coincide with the logical one.
  using TMask = decltype(mask);

  const TMask lane_mask = mask & ((static_cast<TMask>(1U) << lane_idx) - 1);
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
template <alpaka::concepts::Acc TAcc>
ALPAKA_FN_ACC ALPAKA_FN_INLINE auto getPhysicalLaneIdx(TAcc const& acc,
                                                       decltype(alpaka::warp::activemask(std::declval<TAcc&>()))
                                                           const mask,
                                                       std::int32_t logical_lane_idx) -> std::uint32_t {
  using signed_TMask = std::make_signed_t<std::remove_cvref_t<decltype(mask)>>;

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
ALPAKA_FN_ACC ALPAKA_FN_INLINE auto warpReduce(TAcc const& acc, reduce_t const in, const reducer_t f) -> reduce_t {
  constexpr unsigned int w_extent = getWarpSize<TAcc>();

  reduce_t result = in;

  if constexpr (std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevCpu>)
    return result;

  CMS_UNROLL_LOOP
  for (unsigned int offset = w_extent / 2; offset > 0; offset /= 2) {
    result = f(result, alpaka::warp::shfl_down(acc, result, offset, w_extent));
  }

  if constexpr (all)
    result = alpaka::warp::shfl(acc, result, 0, w_extent);

  return result;
}

/*
   * Same as above but for VecArray reductions
   */
template <alpaka::concepts::Acc TAcc, typename reduce_t, int N, typename reducer_t, bool all = true>
  requires(VecArrayReducer<reducer_t, reduce_t, N> && std::is_arithmetic_v<reduce_t>)
ALPAKA_FN_ACC ALPAKA_FN_INLINE auto warpReduce(TAcc const& acc,
                                               cms::alpakatools::VecArray<reduce_t, N> const& in,
                                               const reducer_t f) -> cms::alpakatools::VecArray<reduce_t, N> {
  constexpr unsigned int w_extent = getWarpSize<TAcc>();

  cms::alpakatools::VecArray<reduce_t, N> result = in;

  if constexpr (std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevCpu>)
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

  if constexpr (all) {
    CMS_UNROLL_LOOP
    for (int i = 0; i < N; i++) {
      result[i] = alpaka::warp::shfl(acc, result[i], 0, w_extent);
    }
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
template <alpaka::concepts::Acc TAcc, typename reduce_t, typename reducer_t, bool all = true>
  requires std::is_arithmetic_v<reduce_t>
ALPAKA_FN_ACC ALPAKA_FN_INLINE auto warpSparseReduce(TAcc const& acc,
                                                     decltype(alpaka::warp::activemask(std::declval<TAcc&>()))
                                                         const mask,
                                                     std::uint32_t const lane_idx,
                                                     reduce_t const in,
                                                     const reducer_t f) -> reduce_t {
  constexpr std::uint32_t w_extent = getWarpSize<TAcc>();

  if constexpr (std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevCpu>)
    return mask == 0 ? 0 : in;

  // Non-active lanes must skip the reduction:
  if (isWorkLane<TAcc>(mask, lane_idx) == false) {
    return in;
  }

  std::uint32_t nActiveLanes = alpaka::popcount(acc, mask);  // count number of active lanes

  // First check if this is just a single active lane in the warp:
  if (nActiveLanes == 1)
    return in;

  //Compute the next power of two:
  const std::uint32_t pow2 = w_extent - cms::alpakatools::blas::clz(acc, nActiveLanes - 1);
  const std::uint32_t pow2_boundary = 1 << pow2;

  const std::uint32_t logical_lane_idx = getLogicalLaneIdx<TAcc>(acc, mask, lane_idx);

  reduce_t res = in;

  CMS_UNROLL_LOOP
  for (std::uint32_t offset = pow2_boundary / 2; offset > 0; offset /= 2) {
    const std::uint32_t logical_src_lane_idx = logical_lane_idx + offset;
    const std::uint32_t src_lane_idx =
        (logical_src_lane_idx < nActiveLanes) ? getPhysicalLaneIdx<TAcc>(acc, mask, logical_src_lane_idx) : lane_idx;

    const reduce_t neigh_res = warp::shfl_mask(acc, mask, res, src_lane_idx, w_extent);

    if (logical_src_lane_idx < nActiveLanes)
      res = f(res, neigh_res);
  }

  if constexpr (all) {
    const auto low_lane_idx = getPhysicalLaneIdx(acc, mask, 0);
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
ALPAKA_FN_ACC ALPAKA_FN_INLINE auto warpSparseReduce(TAcc const& acc,
                                                     decltype(alpaka::warp::activemask(std::declval<TAcc&>()))
                                                         const mask,
                                                     std::uint32_t const lane_idx,
                                                     cms::alpakatools::VecArray<reduce_t, N> const& in,
                                                     const reducer_t f) -> cms::alpakatools::VecArray<reduce_t, N> {
  constexpr std::uint32_t w_extent = getWarpSize<TAcc>();

  if constexpr (std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevCpu>)
    return mask == 0 ? 0 : in;

  // Non-active lanes must skip the reduction:
  if (isWorkLane<TAcc>(mask, lane_idx) == false) {
    return in;
  }

  std::uint32_t nActiveLanes = alpaka::popcount(acc, mask);  // count number of active lanes

  // First check if this is just a single active lane in the warp:
  if (nActiveLanes == 1)
    return in;

  //Compute the next power of two:
  const std::uint32_t pow2 = w_extent - cms::alpakatools::blas::clz(acc, nActiveLanes - 1);
  const std::uint32_t pow2_boundary = 1 << pow2;

  const std::uint32_t logical_lane_idx = getLogicalLaneIdx<TAcc>(acc, mask, lane_idx);

  cms::alpakatools::VecArray<reduce_t, N> res = in;

  CMS_UNROLL_LOOP
  for (std::uint32_t offset = pow2_boundary / 2; offset > 0; offset /= 2) {
    const std::uint32_t logical_src_lane_idx = logical_lane_idx + offset;
    const std::uint32_t src_lane_idx =
        (logical_src_lane_idx < nActiveLanes) ? getPhysicalLaneIdx<TAcc>(acc, mask, logical_src_lane_idx) : lane_idx;

    cms::alpakatools::VecArray<reduce_t, N> neigh_res{};
    CMS_UNROLL_LOOP
    for (int i = 0; i < N; i++) {
      neigh_res[i] = warp::shfl_mask(acc, mask, res, src_lane_idx, w_extent);
    }

    if (logical_src_lane_idx < nActiveLanes)
      res = f(res, neigh_res);
  }

  if constexpr (all) {
    const auto low_lane_idx = getPhysicalLaneIdx(acc, mask, 0);
    CMS_UNROLL_LOOP
    for (int i = 0; i < N; i++) {
      res[i] = alpaka::warp::shfl(acc, mask, res, low_lane_idx, w_extent);
    }
  }

  return res;
}

#endif
