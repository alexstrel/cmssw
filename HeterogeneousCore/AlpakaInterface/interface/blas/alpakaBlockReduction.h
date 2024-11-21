#ifndef HeterogeneousCore_AlpakaInterface_interface_blas alpakaBlockReduction_h
#define HeterogeneousCore_AlpakaInterface_interface_blas_alpakaBlockReduction_h

#include "HeterogeneousCore/AlpakaInterface/interface/VecArray.h"

#include "HeterogeneousCore/AlpakaInterface/interface/blas/alpakaAtomicHelper.h"

namespace cms::alpakatools{
  namespace reduce{
    ALPAKA_FN_HOST_ACC inline std::uint32_t roundUpToPowerOf2(std::uint32_t N) {//

      if (N <= 1) {  return 2; }

      N--;

      N |= (N >> 1 );
      N |= (N >> 2 );
      N |= (N >> 4 );
      N |= (N >> 8 );
      N |= (N >> 16);

      return (N+1);
    }

    ALPAKA_FN_HOST_ACC inline std::uint32_t getSubgoupMask(std::uint32_t subgroup_size) {//

      std::uint32_t pw = (subgroup_size >> 1) & 0xF;

      if      ( 1 & pw ) return 0x3 ;
      else if ( 2 & pw ) return 0xF ;

      return 0xFF;
    }

    class WarpReduceKernel
    {
      public:
#ifdef (__CUDA_ARCH__)
        static constexpr std::int32_t default_warp_size = 32;//has to be alpaka::warp::getSize(acc)
#elif defined(__HIP_DEVICE_COMPILE__)
	static constexpr std::int32_t default_warp_size = 64;
#else	
	static constexpr std::int32_t default_warp_size = 1;
#endif

        WarpReduceKernel() = default;

        template< typename TAcc, typename transform_reducer_t >
        ALPAKA_FN_ACC inline auto apply(TAcc const& acc, typename transform_reducer_t::reduce_t const &in,  const transform_reducer_t f, bool all = true)
        -> std::enable_if_t<sizeof(typename transform_reducer_t::reduce_t) <= default_warp_size, typename transform_reducer_t::reduce_t>
        {
          using reduce_t = typename transform_reducer_t::reduce_t;

          std::int32_t const warpExtent = alpaka::warp::getSize(acc);

          auto r = f.get_reducer();

          reduce_t result = r.init(in);
          //
          for (std::uint32_t offset = warpExtent / 2; offset > 0; offset /= 2) {
            std::uint32_t const width = offset << 1;//only part of warp is used    
            result = r(result, alpaka::warp::shfl_down(acc, result, offset, width));
          }

          if (all) result = alpaka::warp::shfl(acc, result, 0, warpExtent);

          return result;
        }
        //
        template< typename TAcc, typename transform_reducer_t >
        ALPAKA_FN_ACC inline auto apply(TAcc const& acc, typename transform_reducer_t::reduce_t const &in,  const transform_reducer_t f, bool all = true)
        -> std::enable_if_t< (sizeof(typename transform_reducer_t::reduce_t) > default_warp_size), typename transform_reducer_t::reduce_t>
        {
          using reduce_t = typename transform_reducer_t::reduce_t;

          using atomic_t = typename cms::alpakatools::atomic_type<reduce_t>::type;

          constexpr std::size_t n = sizeof(reduce_t) / sizeof(atomic_t);

          atomic_t sum_tmp[n];

          memcpy(sum_tmp, &in, sizeof(reduce_t));

          CMS_UNROLL_LOOP
          for (std::size_t i = 0; i < n; i++) sum_tmp[i] = apply<TAcc,transform_reducer_t>(acc, sum_tmp[i], f, all);

          reduce_t result;

          memcpy(&result, sum_tmp, sizeof(reduce_t));

          return result;
        }

    };

    class BlockReduceKernel
    {
      public:

        BlockReduceKernel() = default;

        WarpReduceKernel  warp_reducer;

        template< typename TAcc, typename transform_reducer_t >
        ALPAKA_FN_ACC inline auto apply(TAcc const& acc, std::int32_t const batch, typename transform_reducer_t::reduce_t const &in,  const transform_reducer_t f, bool all = true)
        -> typename transform_reducer_t::reduce_t
        {
          using reduce_t       = typename transform_reducer_t::reduce_t;
          std::int32_t const warpExtent = alpaka::warp::getSize(acc);

          auto const blockExtent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
          
	  constexpr std::uint32_t max_w_items = 32;//64

          std::int32_t const w_items   = std::min( static_cast<std::int32_t> (blockExtent.prod() / warpExtent), static_cast<std::int32_t> (max_w_items) );
          //
          // Perform warp reduction using shuffle operations 
          auto result = warp_reducer.template apply<TAcc, transform_reducer_t >(acc, in, f, all);

          if (all && w_items == 1) return result; // short circuit for single warp CTA

          // we need to know block dimensionality to deduce the leading dimension
          constexpr auto nDim = alpaka::Dim<TAcc>::value;

          constexpr std::size_t leading_dim = nDim - 1;

          auto const threadIdx_x = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[leading_dim];
          auto const blockDim_x  = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[leading_dim];

          std::int32_t const warpIdx_x  = threadIdx_x / warpExtent;
          std::int32_t const laneIdx    = threadIdx_x % warpExtent;

          std::int32_t const w_items_x = std::min( static_cast<std::int32_t> (blockDim_x  / warpExtent), static_cast<std::int32_t> (max_w_items) );

          auto& sdata(alpaka::declareSharedVar<cms::alpakatools::VecArray<reduce_t, max_w_items>, __COUNTER__>(acc));

          // Write the reduced sum of each warp to shared memory
          if (laneIdx == 0) sdata[warpIdx_x+batch*w_items_x] = result;

          alpaka::syncBlockThreads(acc);

          // Reduce the results from all warps (assuming blockDim.x / warpSize warps per block)
          auto const w_items_t = roundUpToPowerOf2(w_items_x);
          //             
          if (threadIdx_x < std::max(static_cast<uint32_t>(warpExtent), w_items_t)) {

            result = threadIdx_x < w_items_x ? sdata[threadIdx_x+batch*w_items_x] : f.init();

            result = warp_reducer.template apply<TAcc, transform_reducer_t >(acc, result, f, all);
          }

          alpaka::syncBlockThreads(acc);

          return result;
       }
    };      	

  } // reduce
} //cms::alpakatools

#endif

