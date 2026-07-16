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
      using reduce_t = std::remove_cvref_t<T>;

      using atomic_t = cms::alpakatools::atomic_type_t<reduce_t>;
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED_DUMMY
      using device_atomic_t = cuda::atomic<atomic_t, cuda::thread_scope_device>;
      using system_atomic_t = cuda::atomic<atomic_t, cuda::thread_scope_system>;
      using count_t = cuda::atomic<int, cuda::thread_scope_device>;
#else
      using device_atomic_t = atomic_t;
      using system_atomic_t = atomic_t;
      using count_t = int;
#endif

      template <alpaka::concepts::Acc TAcc_, typename TQueue_, typename T_>
      friend decltype(auto) create_reduction_resources(TQueue_ queue, Idx nSrc);

      static ReducerResource<TAcc, TQueue, T>& get_reduction_resources(TQueue queue, Idx nSrc, Idx n_blocks) {
        static ReducerResource<TAcc, TQueue, T> instance(queue, nSrc, n_blocks);
        return instance;
      }

      auto get_host_reduce_ptr() { return result_h.data(); }
      auto get_device_reduce_ptr() { return result_d.data(); }
      auto get_partial_ptr() { return partial.data(); }

      auto& get_host_reduce() { return result_h; }
      auto& get_device_reduce() { return result_d; }
      auto& get_partial() { return partial; }

      auto get_count_ptr() { return count.data(); }

      void fetch_reduce_data(TQueue queue) { alpaka::memcpy(queue, result_h, result_d); }

    private:
      alpaka::Buf<alpaka::DevCpu, system_atomic_t, Dim1D, Idx> result_h;
      alpaka::Buf<TAcc, system_atomic_t, Dim1D, Idx> result_d;
      alpaka::Buf<TAcc, device_atomic_t, Dim1D, Idx> partial;

      /** count array that is used to track the number of completed thread blocks at a given batch index */
      alpaka::Buf<TAcc, count_t, Dim1D, Idx> count;

      ReducerResource(TQueue queue, Idx nSrc, Idx n_blocks, const int numa_node_id = 0, bool sync = false)
          : result_h(
                alpaka::allocBuf<system_atomic_t, Idx>(alpaka::getDevByIdx(alpaka::PlatformCpu{}, numa_node_id), nSrc)),
            result_d(alpaka::allocBuf<system_atomic_t, Idx>(alpaka::getDev(queue), nSrc)),
            partial(alpaka::allocBuf<device_atomic_t, Idx>(alpaka::getDev(queue), n_blocks)),
            count(alpaka::allocBuf<count_t, Idx>(alpaka::getDev(queue), nSrc)) {
        alpaka::memset(queue, result_h, 0);
        alpaka::memset(queue, result_d, 0);
        alpaka::memset(queue, partial, 0);
        alpaka::memset(queue, count, 0);
        //
        if (sync == false)
          alpaka::wait(queue);
      }
    };

  }  //namespace reduce
}  //namespace cms::alpakatools

#endif
