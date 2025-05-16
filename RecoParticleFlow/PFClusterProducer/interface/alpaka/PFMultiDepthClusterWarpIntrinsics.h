#ifndef PFClusterProducer_plugins_alpaka_PFMultiDepthClusterWarpIntrinsics_h
#define PFClusterProducer_plugins_alpaka_PFMultiDepthClusterWarpIntrinsics_h


#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace cms::alpakatools{
    namespace warp {

      template <typename TAcc>
      ALPAKA_FN_HOST_ACC inline void syncWarpThreads_mask(TAcc const& acc, unsigned mask) {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        // Alpaka CUDA backend
        if constexpr (alpaka::isAccelerator<TAcc>::value) {
          __syncwarp(mask); // Synchronize all threads within a warp
        }
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
        // Alpaka HIP backend
        if constexpr (alpaka::isAccelerator<TAcc>::value) {
          __builtin_amdgcn_wave_barrier();
        }
#endif
        // No-op for CPU accelerators 
      } 

      template <typename TAcc>
      ALPAKA_FN_HOST_ACC inline unsigned ballot_mask(TAcc const& acc, unsigned mask, int pred ) {
        unsigned res{0};
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        // Alpaka CUDA backend
        if constexpr (alpaka::isAccelerator<TAcc>::value) {
          res = __ballot_sync(mask, pred); // Synchronize all threads within a warp
        }
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
        // Alpaka HIP backend
        if constexpr (alpaka::isAccelerator<TAcc>::value) {
          // HIP equivalent for warp ballot
        }
#endif
        return res;
      }

      template <typename TAcc, typename T>
      ALPAKA_FN_HOST_ACC inline T shfl_mask(TAcc const& acc, unsigned mask, T var, int srcLane, int width ) {
        T res{};
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        // Alpaka CUDA backend
        if constexpr (alpaka::isAccelerator<TAcc>::value) {
          res = __shfl_sync(mask, var, srcLane, width); // Synchronize all threads within a warp
        }
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
        // Alpaka HIP backend
        if constexpr (alpaka::isAccelerator<TAcc>::value) {
          // HIP equivalent for warp __shfl_down_sync
        }
#endif
        return res;
      } 

      template <typename TAcc, typename T>
      ALPAKA_FN_HOST_ACC inline T shfl_down_mask(TAcc const& acc, unsigned mask, T var, int srcLane, int width ) {
        T res{};
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        // Alpaka CUDA backend
        if constexpr (alpaka::isAccelerator<TAcc>::value) {
          res = __shfl_down_sync(mask, var, srcLane, width); // Synchronize all threads within a warp
        }
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
        // Alpaka HIP backend
        if constexpr (alpaka::isAccelerator<TAcc>::value) {
          // HIP equivalent for warp __shfl_down_sync
        }
#endif
        return res;
      } 

      template <typename TAcc, typename T>
      ALPAKA_FN_HOST_ACC inline T shfl_up_mask(TAcc const& acc, unsigned mask, T var, int srcLane, int width ) {
        T res{};
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        // Alpaka CUDA backend
        if constexpr (alpaka::isAccelerator<TAcc>::value) {
          res = __shfl_up_sync(mask, var, srcLane, width); // Synchronize all threads within a warp
        }
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
        // Alpaka HIP backend
        if constexpr (alpaka::isAccelerator<TAcc>::value) {
          // HIP equivalent for warp __shfl_up_sync
        }
#endif
        return res;
      } 

      template <typename TAcc, typename T>
      ALPAKA_FN_HOST_ACC inline T match_any_mask(TAcc const& acc, unsigned mask, T var) {
        T res{};
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        // Alpaka CUDA backend
        if constexpr (alpaka::isAccelerator<TAcc>::value) {
          res = __match_any_sync(mask, var); // Synchronize all threads within a warp
        }
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
        // Alpaka HIP backend
        if constexpr (alpaka::isAccelerator<TAcc>::value) {
          // HIP equivalent for warp __match_any_sync
        }
#endif
        return res;
      } 

    } // end of warp exp

    // reverse the bit order of a (32-bit) unsigned integer.
    template <typename TAcc>
    ALPAKA_FN_HOST_ACC inline unsigned brev(TAcc const& acc, unsigned mask) {
      unsigned res{0};
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      // Alpaka CUDA backend
      if constexpr (alpaka::isAccelerator<TAcc>::value) {
        res = __brev(mask); 
      }
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
      // Alpaka HIP backend
      if constexpr (alpaka::isAccelerator<TAcc>::value) {
      
      }
#endif
      return res;
    }

    // count the number of leading zeros in a 32-bit unsigned integer
    template <typename TAcc>
    ALPAKA_FN_HOST_ACC inline unsigned clz(TAcc const& acc, unsigned mask) {
      unsigned res{0};
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      // Alpaka CUDA backend
      if constexpr (alpaka::isAccelerator<TAcc>::value) {
        res = __clz(mask); 
      }
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
      // Alpaka HIP backend
      if constexpr (alpaka::isAccelerator<TAcc>::value) {
 
      }
#endif
      return res;
    }     
    
  }// end of alpaka
} // end of alpaka namespace
#endif
