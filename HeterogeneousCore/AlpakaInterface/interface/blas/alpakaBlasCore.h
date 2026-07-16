#ifndef HeterogeneousCore_AlpakaInterface_interface_blas_alpakaBlasCore_h
#define HeterogeneousCore_AlpakaInterface_interface_blas_alpakaBlasCore_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/VecArray.h"
#include "HeterogeneousCore/AlpakaInterface/interface/blas/alpakaReducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/blas/alpakaBlockReduction.h"
#include "HeterogeneousCore/AlpakaInterface/interface/blas/alpakaReduceResource.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::multiblas {

  using namespace cms::alpakatools;

  template <typename Transformer, typename reduce_t_, typename Reducer = reduce::plus<reduce_t_>, bool site_unroll_ = false>
  class TransformReduceFunctor {
  public:
    using transformer_t = Transformer;
    using reducer_t = Reducer;

    using reduce_t = reduce_t_;

    transformer_t transformer;
    reducer_t reducer;

    static constexpr bool site_unroll = site_unroll_;

    TransformReduceFunctor(transformer_t t, reducer_t r) : transformer(t), reducer(r) {}

    static constexpr reduce_t init() { return reduce::zero<reduce_t>(); }

    template <typename U>
    static constexpr reduce_t init(U &in) {
      return reduce::set<reduce_t>(in);
    }

    ALPAKA_FN_ACC auto get_transformer() const { return transformer; }

    ALPAKA_FN_ACC auto get_reducer() const { return reducer; }
  };

  template <typename x_buf_t,
            typename y_buf_t,
            typename reducer_params_t,
            unsigned long long nSrc_ = 1,
            bool host_reduction = true>
  class TransformReduceArgs {
  public:
    static constexpr unsigned long long nSrc = nSrc_;

    using reduce_t = std::remove_cvref_t<reducer_params_t>::reduce_t;
    using count_t = std::remove_cvref_t<reducer_params_t>::count_t;

    using system_atomic_t = std::remove_cvref_t<reducer_params_t>::system_atomic_t;
    using device_atomic_t = std::remove_cvref_t<reducer_params_t>::device_atomic_t;
    //
    using Tx = typename alpaka::trait::ElemType<x_buf_t>::type;
    using Ty = typename alpaka::trait::ElemType<y_buf_t>::type;

    // Helper function to initialize the arrays
    template <typename buf_t, typename VecType>
    static auto init_vec_array(VecType &vec) {
      using T = typename alpaka::trait::ElemType<buf_t>::type;
      cms::alpakatools::VecArray<T *, nSrc> result;

      for (Idx i = 0; i < nSrc; ++i) {
        result[i] = const_cast<T *>(vec[i].data());
      }
      return result;
    }

    reduce_t *result_h;
    system_atomic_t *result_d;
    device_atomic_t *partial;

    const cms::alpakatools::VecArray<Tx *, nSrc> x;
    mutable cms::alpakatools::VecArray<Ty *, nSrc> y;

    count_t *count;

    reducer_params_t &reducer_params;

    TransformReduceArgs(reducer_params_t &params,
                        [[maybe_unused]] const std::vector<x_buf_t> &x_,
                        [[maybe_unused]] std::vector<y_buf_t> &y_)
        : result_h(host_reduction ? params.get_host_reduce_ptr() : nullptr),
          result_d(params.get_device_reduce_ptr()),
          partial(params.get_partial_ptr()),
          x(init_vec_array<x_buf_t>(x_)),
          y(init_vec_array<y_buf_t>(y_)),
          count(params.get_count_ptr()),
          reducer_params(params) {}

    TransformReduceArgs(reducer_params_t &params)
        : result_h(host_reduction ? params.get_host_reduce_ptr() : nullptr),
          result_d(params.get_device_reduce_ptr()),
          partial(params.get_partial_ptr()),
          count(params.get_count_ptr()),
          reducer_params(params) {}

    template <typename TQueue>
    auto fetch_data(TQueue queue) const {
      static_assert(host_reduction);
      reducer_params.fetch_reduce_data(queue);
    }
  };

  template <typename TransformReducer_t, typename Args>
  class MultiSrcTransformReducer {
  public:
    const Args args;

    TransformReducer_t f;

    multireduce::BlockReducer block_reducer;

    MultiSrcTransformReducer(TransformReducer_t f, const Args &args) : args(args), f(f) {}

    template <typename TQueue>
    auto fetch(TQueue queue) const {
      args.template fetch_data<TQueue>(queue);
    }

    auto host_reduced_values() const {
      using host_reduce_t = typename TransformReducer_t::reduce_t;

      std::vector<host_reduce_t> values(Args::nSrc);

      for (Idx i = 0; i < Args::nSrc; i++) {
        values[i] = args.result_h[i];
      }

      return values;
    }

    //these are helper methods to return correct thread/block indices and dimensions:
    template <alpaka::concepts::Acc TAcc>
    ALPAKA_FN_ACC inline Vec2D threads_2d(TAcc const &acc) {
      constexpr std::size_t nDim = alpaka::Dim<TAcc>::value;
      static_assert(alpaka::Dim<TAcc>::value <= 3u,
                    "The accelerator used for the Alpaka Kernel has to be at most 3 dimensional!");

      auto const exe_threads = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
      return Vec2D((nDim > 1 ? exe_threads[1] : 0), exe_threads[nDim - 1]);
    }

    template <alpaka::concepts::Acc TAcc>
    ALPAKA_FN_ACC inline Vec2D block_2d(TAcc const &acc) {
      constexpr std::size_t nDim = alpaka::Dim<TAcc>::value;
      static_assert(alpaka::Dim<TAcc>::value <= 3u,
                    "The accelerator used for the Alpaka Kernel has to be at most 3 dimensional!");

      auto const exe_block_div = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
      return Vec2D((nDim > 1 ? exe_block_div[1] : 1), exe_block_div[nDim - 1]);
    }

    template <alpaka::concepts::Acc TAcc, typename... T, bool use_cg_reduce = false, bool use_cg_reducer = false>
    ALPAKA_FN_ACC std::enable_if_t<alpaka::Dim<TAcc>::value <= 3, void> apply(TAcc const &acc,
                                                                              unsigned int const batch_idx,
                                                                              T... external_args) {
      using reducer_t = typename TransformReducer_t::reducer_t;
      using reduce_t = typename TransformReducer_t::reduce_t;
      using transformer_t = typename TransformReducer_t::transformer_t;

      // Set leading dim:
      constexpr std::size_t lDim = alpaka::Dim<TAcc>::value - 1;  // leading dimension

      auto const blockIdx_x = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[lDim];
      auto const [threadIdx_y, threadIdx_x] = threads_2d(acc);

      auto const gridDim_x = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[lDim];
      auto const [blockDim_y, blockDim_x] = block_2d(acc);

      // Block/Grid dims
      auto const i(static_cast<unsigned int>(threadIdx_x + blockIdx_x * blockDim_x));

      transformer_t transformer = f.get_transformer();

      reduce_t result = TransformReducer_t::init();

      if constexpr (sizeof...(external_args) == 0) {
        using x_type = std::remove_cvref_t<decltype(args.x)>;
        using y_type = std::remove_cvref_t<decltype(args.y)>;

        static_assert((cms::alpakatools::is_VecArray_v<x_type> and cms::alpakatools::is_VecArray_v<y_type>),
                      "All arguments must be of type cms::alpakatools::VecArray<T, N>.");
        result = transformer(acc, args.x, args.y, i, 0, batch_idx);
      } else {
        static_assert((cms::alpakatools::is_VecArray_v<T> && ...),
                      "All arguments must be of type cms::alpakatools::VecArray<T, N>.");
        result = transformer(acc, external_args..., i, 0, batch_idx);
      }

      result = block_reducer.template apply<TAcc, TransformReducer_t>(acc, batch_idx, result, f, true);

      auto &isLastBlockDone = alpaka::declareSharedVar<cms::alpakatools::VecArray<bool, Args::nSrc>, __COUNTER__>(acc);

      using count_t = typename Args::count_t;

      count_t *count = static_cast<count_t *>(args.count);

      auto d_result = args.result_d;
      auto d_partial = args.partial;

      if (threadIdx_x == 0 && threadIdx_y == 0) {
        d_partial[blockIdx_x + gridDim_x * batch_idx] = result;

        alpaka::mem_fence(acc, alpaka::memory_scope::Device{});
        unsigned int value =
            alpaka::atomicAdd(acc, static_cast<count_t *>(&count[batch_idx]), 1, alpaka::hierarchy::Blocks{});

        isLastBlockDone[batch_idx] = (value == (gridDim_x - 1));
      }

      alpaka::syncBlockThreads(acc);

      if (isLastBlockDone[batch_idx]) {
        reducer_t reducer = f.get_reducer();

        auto s = threadIdx_y * blockDim_x + threadIdx_x;

        auto accum = TransformReducer_t::init();

        while (s < gridDim_x) {
          accum = reducer(accum, d_partial[batch_idx * gridDim_x + s]);

          s += blockDim_x * blockDim_y;
        }

        result = block_reducer.template apply<TAcc, TransformReducer_t>(acc, batch_idx, accum, f, true);

        if (threadIdx_x == 0 && threadIdx_y == 0) {
          d_result[batch_idx] = result;
        }
      }
    }
  };

  /**
     * Generic transform-reduce device kernel, accepts two scalars and upto four containers with buffers
     */
  template <alpaka::concepts::Acc TAcc,
            typename TQueue,
            typename Transformer,
            typename reduce_t,
            typename Reducer,
            unsigned long long nSrc,
            typename TXBufAcc,
            typename TYBufAcc,
            typename... coeff_t>
  auto instantiateTransformReducer(const TQueue &queue,
                                   const std::vector<TXBufAcc> &x,
                                   std::vector<TYBufAcc> &y,
                                   coeff_t const &...a) {
    auto const nsrc = x.size();

    std::cout << "Created full msrc functor" << std::endl;

    if (nsrc != nSrc)
      std::cout << "Incorrect number of sources\n" << std::endl;

    auto const platformAcc = alpaka::Platform<TAcc>{};
    auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);

    auto max_reduce_blocks =
        2 * alpaka::getAccDevProps<TAcc>(devAcc).m_multiProcessorCount;  //only 2 blocks per MP are active

    auto &reduce_bufs = reduce::ReducerResource<TAcc, TQueue, typename Transformer::reduce_t>::get_reduction_resources(
        queue, nSrc, max_reduce_blocks);

    auto args = TransformReduceArgs<TXBufAcc, TYBufAcc, decltype(reduce_bufs), nSrc>(reduce_bufs, x, y);

    Transformer transform_func(a...);

    auto transform_reduce_func =
        TransformReduceFunctor<Transformer, reduce_t, Reducer, false>{transform_func, Reducer{}};

    return MultiSrcTransformReducer<decltype(transform_reduce_func), decltype(args)>(transform_reduce_func, args);
  }

  template <alpaka::concepts::Acc TAcc,
            typename TQueue,
            typename Transformer,
            typename reduce_t,
            typename Reducer,
            unsigned long long nSrc,
            typename TXBufAcc,
            typename TYBufAcc,
            typename... coeff_t>
  auto instantiateTransformReducer(const TQueue &queue,
                                   auto &reduce_bufs,
                                   const std::vector<TXBufAcc> &x,
                                   std::vector<TYBufAcc> &y,
                                   coeff_t const &...a) {
    auto const nsrc = x.size();

    if (nsrc != nSrc)
      std::cout << "Incorrect number of sources\n" << std::endl;

    std::cout << "Created reduced msrc functor" << std::endl;

    auto args = TransformReduceArgs<TXBufAcc, TYBufAcc, decltype(reduce_bufs), nSrc>(reduce_bufs, x, y);

    Transformer transform_func(a...);

    auto transform_reduce_func =
        TransformReduceFunctor<Transformer, reduce_t, Reducer, false>{transform_func, Reducer{}};

    return MultiSrcTransformReducer<decltype(transform_reduce_func), decltype(args)>(transform_reduce_func, args);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::multiblas

#endif
