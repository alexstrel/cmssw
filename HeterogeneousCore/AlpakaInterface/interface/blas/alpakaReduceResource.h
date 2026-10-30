#ifndef HeterogeneousCore_AlpakaInterface_interface_blas_alpakaReduceResource_h
#define HeterogeneousCore_AlpakaInterface_interface_blas_alpakaReduceResource_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/VecArray.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

namespace cms::alpakatools {
  namespace reduce {

    template <alpaka::concepts::Acc TAcc, typename TQueue, typename T, bool use_augmented_reduction>
    class ReducerResource;

    template <alpaka::concepts::Acc TAcc, typename TQueue, typename T, bool use_augmented_reduction>
    decltype(auto) create_reduction_resources(TQueue queue, Idx const nSrc) {
      auto const devAcc = alpaka::getDev(queue);

      auto const max_reduce_blocks = 2 * alpaka::getAccDevProps<TAcc>(devAcc).m_multiProcessorCount;

      return ReducerResource<TAcc, TQueue, T, use_augmented_reduction>(queue, nSrc, max_reduce_blocks);
    }

    template <alpaka::concepts::Acc TAcc, typename TQueue, typename T, bool use_augmented_reduction>
    class ReducerResource {
    public:
      using DeviceT = decltype(alpaka::getDev(std::declval<TQueue const&>()));

      using DataT = std::remove_cvref_t<T>;
      using AtomicT = cms::alpakatools::atomic_type_t<DataT>;
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED_DUMMY
      using DeviceAtomicT = cuda::atomic<AtomicT, cuda::thread_scope_device>;
      using SystemAtomicT = cuda::atomic<AtomicT, cuda::thread_scope_system>;
      using CountT = cuda::atomic<int, cuda::thread_scope_device>;
#else
      using DeviceAtomicT = AtomicT;
      using SystemAtomicT = AtomicT;
      using CountT = std::int32_t;
#endif

      template <alpaka::concepts::Acc TAcc_, typename TQueue_, typename T_, bool use_augmented_reduction_>
      friend decltype(auto) create_reduction_resources(TQueue_ queue, Idx const nSrc);

      auto get_host_reduce_ptr() { return result_h.data(); }
      auto get_device_reduce_ptr() { return result_d.data(); }
      auto get_partial_ptr() { return partial.data(); }

      auto& get_host_reduce() { return result_h; }
      auto& get_device_reduce() { return result_d; }
      auto& get_partial() { return partial; }

      auto get_partial_stride() const { return partial_stride; }

      auto get_count_ptr() { return count.data(); }

      auto get_partial_upper_limit() {
        return use_augmented_reduction ? alpaka::getExtents(partial).prod() / 2 : alpaka::getExtents(partial).prod();
      }

      void fetch_reduce_data(TQueue queue) { alpaka::memcpy(queue, result_h, result_d); }

    private:
      const Idx partial_stride;

      cms::alpakatools::host_buffer<SystemAtomicT[]> result_h;
      cms::alpakatools::device_buffer<DeviceT, SystemAtomicT[]> result_d;
      cms::alpakatools::device_buffer<DeviceT, DeviceAtomicT[]> partial;
      /** count array that is used to track the number of completed thread blocks at a given batch index */
      cms::alpakatools::device_buffer<DeviceT, CountT[]> count;

      ReducerResource(TQueue queue, Idx nSrc, Idx n_blocks, const int numa_node_id = 0)
          : partial_stride(nSrc * n_blocks),
            result_h(cms::alpakatools::make_host_buffer<SystemAtomicT[]>(queue, nSrc * n_blocks)),
            result_d(cms::alpakatools::make_device_buffer<DeviceAtomicT[]>(queue, nSrc * n_blocks)),
            partial(cms::alpakatools::make_device_buffer<DeviceAtomicT[]>(
                queue, (use_augmented_reduction ? 2 : 1) * partial_stride)),
            count(cms::alpakatools::make_device_buffer<CountT[]>(queue, nSrc)) {
        alpaka::memset(queue, result_h, 0);
        alpaka::memset(queue, result_d, 0);
        alpaka::memset(queue, partial, 0);
        alpaka::memset(queue, count, 0);
      }
    };

  }  //namespace reduce
}  //namespace cms::alpakatools

#endif
