#ifndef HeterogeneousCore_AlpakaInterface_interface_blas_alpakaReduceResource_h
#define HeterogeneousCore_AlpakaInterface_interface_blas_alpakaReduceResource_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/VecArray.h"

namespace cms::alpakatools {
  namespace reduce {

    template<typename T, typename TAcc, typename TDevAcc, typename TQueue> class ReducerResource;

    template<typename T, typename TAcc, typename TDevAcc, typename TQueue>
    decltype(auto) create_reduction_resources(const TDevAcc &devAcc, TQueue queue, Idx nSrc) {

      auto max_reduce_blocks = 2 * alpaka::getAccDevProps<TAcc>(devAcc).m_multiProcessorCount;

      return ReducerResource<T, TAcc, TDevAcc, TQueue>(devAcc, queue, nSrc, max_reduce_blocks);
    }

    template <typename T, typename TAcc, typename TDevAcc, typename TQueue>
    class ReducerResource {
    public:
      using reduce_t = T;
      //
      using device_atomic_t = reduce_t;
      using system_atomic_t = reduce_t;
      using count_t = int;
      //
      using TBufHost = alpaka::Buf<alpaka::DevCpu, system_atomic_t, Dim1D, Idx>;
      using TBufAcc = alpaka::Buf<TAcc, system_atomic_t, Dim1D, Idx>;

      using TBuf1DAcc = alpaka::Buf<TAcc, device_atomic_t, Dim1D, Idx>;

      using TCountBufAcc = alpaka::Buf<TAcc, count_t, Dim1D, Idx>;

      template<typename T_, typename TAcc_, typename TDevAcc_, typename TQueue_>
      friend decltype(auto) create_reduction_resources(const TDevAcc_ &devAcc, TQueue_ queue, Idx nSrc);

      //static ReducerResource reduction_buffers;
      static ReducerResource<T, TAcc, TDevAcc, TQueue>& get_reduction_resources(const TDevAcc& devAcc,
                                                                                TQueue queue,
                                                                                Idx nSrc,
                                                                                Idx n_blocks) {
        static ReducerResource<T, TAcc, TDevAcc, TQueue> instance(devAcc, queue, nSrc, n_blocks);
        return instance;
      }
      //
      auto get_host_reduce_ptr() { return result_h.data(); }
      auto get_device_reduce_ptr() { return result_d.data(); }
      auto get_partial_ptr() { return partial.data(); }

      TBufHost& get_host_reduce() { return result_h; }
      TBufAcc& get_device_reduce() { return result_d; }
      TBuf1DAcc& get_partial() { return partial; }

      auto get_count_ptr() { return count.data(); }
      //
      auto fetch_reduce_data(TQueue queue) { alpaka::memcpy(queue, result_h, result_d); }

    private:
      TBufHost result_h;
      TBufAcc result_d;
      TBuf1DAcc partial;

      TCountBufAcc
          count; /** count array that is used to track the number of completed thread blocks at a given batch index */

      ReducerResource(const TDevAcc& devAcc, TQueue queue, Idx nSrc, Idx n_blocks, bool sync = false)
          : result_h(alpaka::allocBuf<system_atomic_t, Idx>(alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0), nSrc)),
            result_d(alpaka::allocBuf<system_atomic_t, Idx>(devAcc, nSrc)),
            partial(alpaka::allocBuf<device_atomic_t, Idx>(devAcc, n_blocks)),
            count(alpaka::allocBuf<count_t, Idx>(devAcc, nSrc)) {
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
