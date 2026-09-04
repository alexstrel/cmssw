#ifndef HeterogeneousCore_AlpakaInterface_interface_blas_alpakaReduceResource_h
#define HeterogeneousCore_AlpakaInterface_interface_blas_alpakaReduceResource_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/VecArray.h"

namespace cms::alpakatools {
  namespace reduce {

    template <alpaka::concepts::Acc TAcc, typename TQueue, typename T>
    class ReducerResource;

    template <alpaka::concepts::Acc TAcc, typename TQueue, typename T>
    decltype(auto) create_reduction_resources(TQueue queue, Idx nSrc) {
      auto const devAcc = alpaka::getDev(queue);

      auto max_reduce_blocks = 2 * alpaka::getAccDevProps<TAcc>(devAcc).m_multiProcessorCount;

      return ReducerResource<TAcc, TQueue, T>(queue, nSrc, max_reduce_blocks);
    }

    template <alpaka::concepts::Acc TAcc, typename TQueue, typename T>
    class ReducerResource {
    public:
      using ReduceT = std::remove_cvref_t<T>;

      using AtomicT = cms::alpakatools::atomic_type_t<ReduceT>;
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED_DUMMY
      using DeviceAtomicT = cuda::atomic<AtomicT, cuda::thread_scope_device>;
      using SystemAtomicT = cuda::atomic<AtomicT, cuda::thread_scope_system>;
      using CountT = cuda::atomic<int, cuda::thread_scope_device>;
#else
      using DeviceAtomicT = AtomicT;
      using SystemAtomicT = AtomicT;
      using CountT = int;
#endif

      template <alpaka::concepts::Acc TAcc_, typename TQueue_, typename T_>
      friend decltype(auto) create_reduction_resources(TQueue_ queue, Idx nSrc);

      auto get_host_reduce_ptr() { return result_h.data(); }
      auto get_device_reduce_ptr() { return result_d.data(); }
      auto get_partial_ptr() { return partial.data(); }

      auto& get_host_reduce() { return result_h; }
      auto& get_device_reduce() { return result_d; }
      auto& get_partial() { return partial; }

      auto get_count_ptr() { return count.data(); }

      auto get_partial_upper_limit() { return alpaka::getExtents(partial).prod(); }

      void fetch_reduce_data(TQueue queue) { alpaka::memcpy(queue, result_h, result_d); }

    private:
      alpaka::Buf<alpaka::DevCpu, SystemAtomicT, Dim1D, Idx> result_h;
      alpaka::Buf<TAcc, SystemAtomicT, Dim1D, Idx> result_d;
      alpaka::Buf<TAcc, DeviceAtomicT, Dim1D, Idx> partial;

      /** count array that is used to track the number of completed thread blocks at a given batch index */
      alpaka::Buf<TAcc, CountT, Dim1D, Idx> count;

      ReducerResource(TQueue queue, Idx nSrc, Idx n_blocks, const int numa_node_id = 0)
          : result_h(
                alpaka::allocBuf<SystemAtomicT, Idx>(alpaka::getDevByIdx(alpaka::PlatformCpu{}, numa_node_id), nSrc)),
            result_d(alpaka::allocBuf<SystemAtomicT, Idx>(alpaka::getDev(queue), nSrc)),
            partial(alpaka::allocBuf<DeviceAtomicT, Idx>(alpaka::getDev(queue), nSrc * n_blocks)),
            count(alpaka::allocBuf<CountT, Idx>(alpaka::getDev(queue), nSrc)) {
        alpaka::memset(queue, result_h, 0);
        alpaka::memset(queue, result_d, 0);
        alpaka::memset(queue, partial, 0);
        alpaka::memset(queue, count, 0);
      }
    };

  }  //namespace reduce
}  //namespace cms::alpakatools

#endif
