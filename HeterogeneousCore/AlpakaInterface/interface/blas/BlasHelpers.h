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

namespace extfloat {

  template <std::floating_point T>
  class ExtendedFloat;

  template <std::floating_point T>
  constexpr auto fast_two_sum(T a, T b) {
    const T s = warp::add_rn(a, b);
    const T z = warp::sub_rn(s, a);
    const T t = warp::sub_rn(b, z);

    return ExtendedFloat<T>{s, t};
  }

  template <std::floating_point T>
  constexpr auto two_sum(T a, T b) {
    const T s = warp::add_rn(a, b);

    const T a_prime = warp::sub_rn(s, b);
    const T b_prime = warp::sub_rn(s, a_prime);

    const T delta_a = warp::sub_rn(a, a_prime);
    const T delta_b = warp::sub_rn(b, b_prime);

    const T t = warp::add_rn(delta_a, delta_b);

    return ExtendedFloat<T>{s, t};
  }

  template <std::floating_point T>
  constexpr auto two_prod(T a, T b) {
    const T s = warp::mul_rn(a, b);
    const T t = warp::fma_rn(a, b, -s);

    return ExtendedFloat<T>{s, t};
  }

  enum class UpdateKind { UpdateLowComponent, UpdateHighComponent, InvalidUpdate };

  template <std::floating_point T>
  class ExtendedFloat {
  public:
    using value_type = T;

    constexpr ExtendedFloat() = default;
    constexpr ExtendedFloat(ExtendedFloat const&) = default;
    constexpr ExtendedFloat(ExtendedFloat&&) = default;

    template <std::floating_point S>
    constexpr ExtendedFloat(const S hi, const S lo) : reg{static_cast<T>(hi), static_cast<T>(lo)} {}

    template <std::floating_point S>
    constexpr ExtendedFloat(const S hi) : reg{static_cast<T>(hi), static_cast<T>(0.)} {}

    constexpr ExtendedFloat& operator=(ExtendedFloat const&) = default;
    constexpr ExtendedFloat& operator=(ExtendedFloat&&) = default;

    template <std::floating_point U>
    constexpr ExtendedFloat& operator=(ExtendedFloat<U> const& other) {
      reg[0] = static_cast<T>(other.high());
      reg[1] = static_cast<T>(other.low());
      return *this;
    }

    template <std::floating_point U>
    constexpr ExtendedFloat& operator=(U val) {
      reg[0] = static_cast<T>(val);
      reg[1] = T{0.};
      return *this;
    }

    constexpr T const HighComponent() const { return reg[0]; }
    constexpr T const LowComponent() const { return reg[1]; }

    template <UpdateKind which_update = UpdateKind::InvalidUpdate>
    constexpr void Accum(T const v) {
      if constexpr (which_update == UpdateKind::UpdateLowComponent) {
        reg[1] = warp::add_rn(reg[1], v);
      } else if constexpr (which_update == UpdateKind::UpdateHighComponent) {
        reg[0] = warp::add_rn(reg[0], v);
      } else {
        //nop
      }
    }

    template <bool extended_sum = false>
    constexpr auto Aggregate() const {
      if constexpr (extended_sum) {
        return fast_two_sum(reg[0], reg[1]);
      } else {
        return warp::add_rn(reg[0], reg[1]);
      }
    }

  private:
    T reg[2]{};
  };

  template <typename T>
  struct is_extended_float : std::false_type {};

  template <std::floating_point T>
  struct is_extended_float<ExtendedFloat<T>> : std::true_type {};

  template <typename T>
  concept ExtendedFloatType = is_extended_float<std::remove_cvref_t<T>>::value;

  template <typename T>
  struct floating_value;

  template <ExtendedFloatType T>
  struct floating_value<T> {
    using type = typename std::remove_cvref_t<T>::value_type;
  };

  template <std::floating_point T>
  struct floating_value<T> {
    using type = std::remove_cvref_t<T>;
  };

  template <typename T>
  using floating_value_t = typename floating_value<std::remove_cvref_t<T>>::type;

  template <std::floating_point T>
  constexpr ExtendedFloat<T> operator+(ExtendedFloat<T> const& x, ExtendedFloat<T> const& y) {
    constexpr bool use_fast_two_sum = true;

    auto hi_sum = two_sum(x.HighComponent(), y.HighComponent());

    auto const lo_sum = two_sum(x.LowComponent(), y.LowComponent());

    hi_sum.template Accum<UpdateKind::UpdateLowComponent>(lo_sum.HighComponent());

    auto v = hi_sum.template Aggregate<use_fast_two_sum>();

    v.template Accum<UpdateKind::UpdateLowComponent>(lo_sum.LowComponent());

    return v.template Aggregate<use_fast_two_sum>();
  }

  template <std::floating_point T>
  constexpr ExtendedFloat<T> operator*(ExtendedFloat<T> const& x, ExtendedFloat<T> const& y) {
    constexpr bool use_fast_two_sum = true;
    auto hi_prod = two_prod(x.HighComponent(), y.HighComponent());

    T const p = warp::mul_rn(x.HighComponent(), y.LowComponent());
    T const c = warp::fma_rn(x.LowComponent(), y.HighComponent(), p);

    hi_prod.template Accum<UpdateKind::UpdateLowComponent>(c);

    return hi_prod.template Aggregate<use_fast_two_sum>();
  }

  template <std::floating_point T>
  constexpr T const Hi(ExtendedFloat<T> const& x) {
    return x.HighComponent();
  }

  template <std::floating_point T>
  constexpr T const Lo(ExtendedFloat<T> const& x) {
    return x.LowComponent();
  }

}  // namespace extfloat

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
   * Same as above but for extended float reductions
   */

template <alpaka::concepts::Acc TAcc, typename reduce_t, typename reducer_t, bool all = true>
  requires(extfloat::is_extended_float<std::remove_cvref_t<reduce_t>>::value)
ALPAKA_FN_ACC ALPAKA_FN_INLINE auto warpReduce(TAcc const& acc, reduce_t const& in, const reducer_t f) -> reduce_t {
  using base_float_tp = extfloat::floating_value_t<reduce_t>;

  constexpr unsigned int w_extent = getWarpSize<TAcc>();

  reduce_t result = in;

  if constexpr (std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevCpu>)
    return result;

  CMS_UNROLL_LOOP
  for (unsigned int offset = w_extent / 2; offset > 0; offset /= 2) {
    base_float_tp hi = alpaka::warp::shfl_down(acc, result.HighComponent(), offset, w_extent);
    base_float_tp lo = alpaka::warp::shfl_down(acc, result.LowComponent(), offset, w_extent);

    result = f(result, reduce_t{hi, lo});
  }

  if constexpr (all) {
    base_float_tp hi = alpaka::warp::shfl_down(acc, result.HighComponent(), 0, w_extent);
    base_float_tp lo = alpaka::warp::shfl_down(acc, result.LowComponent(), 0, w_extent);

    return reduce_t{hi, lo};
  }

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

template <alpaka::concepts::Acc TAcc, typename reduce_t, typename reducer_t, bool all = true>
  requires(extfloat::is_extended_float<std::remove_cvref_t<reduce_t>>::value)
ALPAKA_FN_ACC ALPAKA_FN_INLINE auto warpSparseReduce(TAcc const& acc,
                                                     decltype(alpaka::warp::activemask(std::declval<TAcc&>()))
                                                         const mask,
                                                     std::uint32_t const lane_idx,
                                                     reduce_t const in,
                                                     const reducer_t f) -> reduce_t {
  using base_float_tp = extfloat::floating_value_t<reduce_t>;

  constexpr std::uint32_t w_extent = getWarpSize<TAcc>();

  if constexpr (std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevCpu>)
    return mask == 0 ? reduce_t{0.} : in;

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

    base_float_tp const neigh_res_hi = alpaka::warp::shfl_down(acc, res.HighComponent(), src_lane_idx, w_extent);
    base_float_tp const neigh_res_lo = alpaka::warp::shfl_down(acc, res.LowComponent(), src_lane_idx, w_extent);

    reduce_t neigh_res{neigh_res_hi, neigh_res_lo};

    if (logical_src_lane_idx < nActiveLanes)
      res = f(res, neigh_res);
  }

  if constexpr (all) {
    const auto low_lane_idx = getPhysicalLaneIdx(acc, mask, 0);

    base_float_tp const res_hi = alpaka::warp::shfl_down(acc, res.HighComponent(), low_lane_idx, w_extent);
    base_float_tp const res_lo = alpaka::warp::shfl_down(acc, res.LowComponent(), low_lane_idx, w_extent);

    return reduce_t{res_hi, res_lo};
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
