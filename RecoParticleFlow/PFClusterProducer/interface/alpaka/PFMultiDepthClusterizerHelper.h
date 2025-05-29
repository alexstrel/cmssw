#ifndef PFClusterProducer_plugins_alpaka_PFMultiDepthClusterizerHelper_h
#define PFClusterProducer_plugins_alpaka_PFMultiDepthClusterizerHelper_h

/**
 * @file PFMultiDepthClusterizerHelper.h
 * @brief Warp-level utility functions for particle flow multi-depth clustering.
 * 
 * This header provides basic warp-synchronous operations used in clustering algorithms,
 * including bitwise manipulations (least/most significant set bits) and masked
 * warp-exclusive sum computations.
 */

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusterWarpIntrinsics.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  /**
 * @brief Returns the position of the least significant set bit in a mask.
 *
 * @tparam TAcc Alpaka accelerator type.
 * 
 * @param acc   Alpaka accelerator instance.
 * @param x     Input bitmask.
 * 
 * @return Index of least significant 1 bit (0-based). (or -1 if x == 0).
 */
  template< typename TAcc >
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE unsigned int get_ls1b_idx(TAcc const& acc, const int x) {
    const int pos = alpaka::ffs(acc, x);
    return static_cast<unsigned int>(pos - 1);
  }

/**
 * @brief Clears the least significant set bit in a mask.
 *
 * @tparam TAcc  Alpaka accelerator type.
 * 
 * @param acc    Alpaka accelerator instance.
 * @param x      Input bitmask.
 * 
 * @return Bitmask with least significant 1 bit cleared.
 */
  
  template< typename TAcc >
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE unsigned int erase_ls1b(TAcc const& acc, const unsigned int x) {
    return (x & (x-1));
  }

/**
 * @brief Returns the position of the most significant set bit in a mask.
 *
 * @tparam TAcc Alpaka accelerator type.
 * 
 * @param acc Alpaka accelerator instance.
 * @param x Input bitmask.
 * 
 * @return Index of most significant 1 bit (0-based). (or -1 if x == 0)
 */

  template< typename TAcc >
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE unsigned int get_ms1b_idx(TAcc const& acc, const unsigned int x) {
    constexpr unsigned int size = sizeof(unsigned int)-1;
    const int pos = size - cms::alpakatools::clz(acc, x);
    return pos - 1;
  }

  /**
 * @brief Performs warp-level exclusive prefix sum under a custom lane mask.
 *
 * @tparam TAcc Alpaka accelerator type.
 * @tparam accum If true, broadcast total accumulated value to lowest active lane.
 * 
 * @param acc   Alpaka accelerator instance.
 * @param mask  Active lane mask.
 * @param val   Value to include in the prefix sum.
 * @param lane_idx Current thread's lane index.
 * 
 * @return Exclusive prefix sum value for the current lane.
 */

  template <typename TAcc, bool accum = true, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE unsigned int warp_exclusive_sum(TAcc const& acc, const unsigned int mask, unsigned int val, const unsigned int lane_idx) {
    if ( mask == 0x0 ) return 0;

    const unsigned int w_extent = alpaka::warp::getSize(acc);
    //
    unsigned int local_offset = 0;
    //
    CMS_UNROLL_LOOP
    for (unsigned int j = 1; j < w_extent; j *= 2) {
      const auto n = warp::shfl_up_mask(acc, mask, val, j, w_extent);
      if (lane_idx >= j) local_offset += n;
    }
    //
    warp::syncWarpThreads_mask(acc, mask);

    if constexpr (!accum) {
 	return local_offset;
    } else {
    	// Compute the lowest and the highest valid lane index in the mask:
    	const unsigned low_lane_idx  = get_ls1b_idx(acc, mask);
    	const unsigned high_lane_idx = get_ms1b_idx(acc, mask);

    	// send last lane value (total tile offset) to lane idx = low_lane_idx:
    	const unsigned active_mask = 1 | (1 << high_lane_idx);
    	const unsigned x = warp::shfl_mask(acc, active_mask, local_offset + val,  high_lane_idx,  w_extent);
    	//
    	if (lane_idx == low_lane_idx) local_offset = x;

    	warp::syncWarpThreads_mask(acc, mask);
    }
    return local_offset;
  }

}

#endif

