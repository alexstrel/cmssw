#ifndef HeterogeneousCore_AlpakaInterface_interface_blas alpakaBlasCore_h
#define HeterogeneousCore_AlpakaInterface_interface_blas_alpakaBlasCore_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/VecArray.h"
#include "HeterogeneousCore/AlpakaInterface/interface/blas/alpakaReducer.h"

namespace cms::alpakatools {
  namespace blas {

    template <typename reduce_t_,
              typename Transformer,
              typename Reducer = reduce::plus<reduce_t_>,
              bool site_unroll_ = false>
    class TransformReduceFunctor {
    public:
      //
      using reduce_t = reduce_t_;
      using transformer_t = Transformer;
      using reducer_t = Reducer;
      //
      transformer_t transformer;
      reducer_t reducer;
      //
      static constexpr bool site_unroll = site_unroll_;

      TransformReduceFunctor(transformer_t t, reducer_t r) : transformer(t), reducer(r) {}

      ALPAKA_FN_ACC reduce_t operator()(const reduce_t &x, const reduce_t &y) const {
        return reducer(x, y);
      }     

      ALPAKA_FN_ACC auto init() const { return reducer.init(); }

      ALPAKA_FN_ACC auto get_reducer() const { return reducer; }
    };

    template <typename xz_buf_t, typename yw_buf_t, typename reducer_params_t, unsigned long long nSrc_ = 1, bool host_reduction = true>
    class TransformReduceArgs {
    public:
      static constexpr unsigned long long nSrc = nSrc_;

      using reduce_t = std::remove_cvref_t<reducer_params_t>::reduce_t;
      using count_t = std::remove_cvref_t<reducer_params_t>::count_t;

      using system_atomic_t = std::remove_cvref_t<reducer_params_t>::system_atomic_t;
      using device_atomic_t = std::remove_cvref_t<reducer_params_t>::device_atomic_t;
      //
      using Txz = typename alpaka::trait::ElemType<xz_buf_t>::type;
      using Tyw = typename alpaka::trait::ElemType<yw_buf_t>::type;

      // Helper function to initialize the arrays
      template <typename buf_t, typename VecType>
      static auto init_vec_array(VecType &vec) {
        using T = typename alpaka::trait::ElemType<buf_t>::type;
        cms::alpakatools::VecArray<T *, nSrc> result;

        for (int i = 0; i < nSrc; ++i) {
          result[i] = const_cast<T *>(vec[i].data());
        }
        return result;
      }

      reduce_t *result_h;
      system_atomic_t *result_d;
      device_atomic_t *partial;

      const cms::alpakatools::VecArray<Txz *, nSrc> x;
      mutable cms::alpakatools::VecArray<Tyw *, nSrc> y;
      mutable cms::alpakatools::VecArray<Tyw *, nSrc> w;
      const cms::alpakatools::VecArray<Txz *, nSrc> z;
      //
      count_t *count;
      //
      reducer_params_t &reducer_params;
      //
      TransformReduceArgs(reducer_params_t &params,
                          const std::vector<xz_buf_t> &x_,
                          std::vector<yw_buf_t> &y_,
                          [[maybe_unused]] std::vector<yw_buf_t> &w_,
                          [[maybe_unused]] const std::vector<xz_buf_t> &z_)
          : result_h(host_reduction ? params.get_host_reduce_ptr() : nullptr),
            result_d(params.get_device_reduce_ptr()),
            partial(params.get_partial_ptr()),
            x(init_vec_array<xz_buf_t>(x_)),
            y(init_vec_array<yw_buf_t>(y_)),
            w(init_vec_array<yw_buf_t>(w_)),
            z(init_vec_array<xz_buf_t>(z_)),
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

      BlockReducer block_reducer;

      MultiSrcTransformReducer(TransformReducer_t f, const Args &args) : f(f), args(args) {}

      template <typename TQueue>
      auto fetch(TQueue queue) const {
        args.template fetch_data<TQueue>(queue);
      }

      auto host_reduced_values() const {
        using host_reduce_t = typename TransformReducer_t::reduce_t;

        std::vector<host_reduce_t> values(Args::nSrc);
        //
        for (int i = 0; i < Args::nSrc; i++) {
          values[i] = args.result_h[i];
        }
        //
        return values;
      }

      //these are helper methods to return correct thread/block indices and dimensions:
      template <typename TAcc>
      ALPAKA_FN_ACC inline Vec2D threads_2d(TAcc const &acc) {
        constexpr std::size_t nDim = alpaka::Dim<TAcc>::value;
        static_assert(alpaka::Dim<TAcc>::value <= 3u,
                      "The accelerator used for the Alpaka Kernel has to be at most 3 dimensional!");

        auto const exe_threads = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
        return Vec2D((nDim > 1 ? exe_threads[1] : 0), exe_threads[nDim - 1]);
      }

      template <typename TAcc>
      ALPAKA_FN_ACC inline Vec2D block_2d(TAcc const &acc) {
        constexpr std::size_t nDim = alpaka::Dim<TAcc>::value;
        static_assert(alpaka::Dim<TAcc>::value <= 3u,
                      "The accelerator used for the Alpaka Kernel has to be at most 3 dimensional!");

        auto const exe_block_div = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        return Vec2D((nDim > 1 ? exe_block_div[1] : 1), exe_block_div[nDim - 1]);
      }

      template <typename TAcc, typename... T, bool use_cg_reduce = false, bool use_cg_reducer = false>
      ALPAKA_FN_ACC std::enable_if_t<alpaka::Dim<TAcc>::value <= 3, void> apply(TAcc const &acc,
                                                                                unsigned int const batch_idx,
                                                                                T... external_args) {
        // Set leading dim:
        constexpr std::size_t lDim = alpaka::Dim<TAcc>::value - 1;  // leading dimension
        //
        auto const blockIdx_x = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[lDim];
        auto const [threadIdx_y, threadIdx_x] = threads_2d(acc);

        auto const gridDim_x = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[lDim];
        auto const [blockDim_y, blockDim_x] = block_2d(acc);

        // Block/Grid dims
        auto const i(static_cast<unsigned int>(threadIdx_x + blockIdx_x * blockDim_x));

        auto result = 0.f;

        if constexpr (sizeof...(external_args) == 0) {
          using x_type = std::remove_cvref_t<decltype(args.x)>;
          using y_type = std::remove_cvref_t<decltype(args.y)>;
          using w_type = std::remove_cvref_t<decltype(args.w)>;
          using z_type = std::remove_cvref_t<decltype(args.z)>;

          static_assert((cms::alpakatools::is_vector_array_v<x_type> and cms::alpakatools::is_vector_array_v<y_type> and
                         cms::alpakatools::is_vector_array_v<w_type> and cms::alpakatools::is_vector_array_v<z_type>),
                        "All arguments must be of type cms::alpakatools::VecArray<T, N>.");
          result = f.template transform<TAcc, typename Args::Txz, typename Args::Tyw>(
              acc, args.x, args.y, args.w, args.z, i, 0, batch_idx);
        } else {
          static_assert((cms::alpakatools::is_vector_array_v<T> && ...),
                        "All arguments must be of type cms::alpakatools::VecArray<T, N>.");
          result = f.template transform<TAcc, typename Args::Txz, typename Args::Tyw>(
              acc, external_args..., i, 0, batch_idx);
        }

        result = block_reducer.template apply<TAcc, TransformReducer_t>(acc, batch_idx, result, f, true);

        auto &isLastBlockDone =
            alpaka::declareSharedVar<cms::alpakatools::VecArray<bool, Args::nSrc>, __COUNTER__>(acc);

        using count_t = typename Args::count_t;

        count_t *count = static_cast<count_t *>(args.count);

        auto d_result = args.result_d;
        auto d_partial = args.partial;

        if (threadIdx_x == 0 and threadIdx_y == 0) {
          d_partial[blockIdx_x + gridDim_x * batch_idx] = result;

          alpaka::mem_fence(acc, alpaka::memory_scope::Device{});
          auto value =
              alpaka::atomicAdd(acc, static_cast<count_t *>(&count[batch_idx]), 1, alpaka::hierarchy::Blocks{});

          isLastBlockDone[batch_idx] = (value == (gridDim_x - 1));
        }

        alpaka::syncBlockThreads(acc);

        if (isLastBlockDone[batch_idx]) {
          auto ii = threadIdx_y * blockDim_x + threadIdx_x;

          result = f.init();

          while (ii < gridDim_x) {
            result = f.template reduce<TAcc>(acc, result, d_partial[batch_idx * gridDim_x + ii]);

            ii += blockDim_x * blockDim_y;
          }
          //
          result = block_reducer.template apply<TAcc, TransformReducer_t>(acc, batch_idx, result, f, true);

          if (threadIdx_x == 0 and threadIdx_y == 0) {
            d_result[batch_idx] = result;
          }
        }
      }
    };

    /**
     * Generic transform-reduce device kernel, accepts two scalars and upto four containers with buffers
     */
    template <typename TAcc,
              typename TDevAcc,
              typename TQueue,
              typename TXZBufAcc,
              typename TYWBufAcc,
              typename reduce_t,
              typename coeff_t,
              typename Functor,
              unsigned long long nSrc = 1,
              bool... control_flags>
    auto instantiateTransformReducer(const TDevAcc &devAcc,
                                           TQueue &queue,
                                     const coeff_t &a,
                                     const coeff_t &b,
                                     const std::vector<TXZBufAcc> &x,
                                           std::vector<TYWBufAcc> &y,
                                           std::vector<TYWBufAcc> &w,
                                     const std::vector<TXZBufAcc> &z) {
      auto const nsrc = x.size();

      if (nsrc != nSrc)
        std::cout << "Incorrect number of sources\n" << std::endl;

      auto max_reduce_blocks =
          2 * alpaka::getAccDevProps<TAcc>(devAcc).m_multiProcessorCount;  //only 2 blocks per MP are active

      auto &reduce_bufs = reduce::ReducerResource<reduce_t, TAcc, TDevAcc, TQueue>::get_reduction_resources(
          devAcc, queue, nSrc, max_reduce_blocks);

      auto args = TransformReduceArgs<TXZBufAcc, TYWBufAcc, decltype(reduce_bufs), nSrc>(reduce_bufs, x, y, w, z);
      //
      Functor tr_func(a, b);

      return MultiSrcTransformReducer<Functor, decltype(args)>(tr_func, args);
    }

   template <typename TAcc,
              typename TDevAcc,
              typename TQueue,
              typename TXZBufAcc,
              typename TYWBufAcc,
              typename reduce_t,
              typename coeff_t,
              typename Functor,
              unsigned long long nSrc = 1,
              bool... control_flags>
    auto instantiateTransformReducer(const TDevAcc &devAcc,
                                           TQueue &queue,
                                     const coeff_t &a,
                                     const coeff_t &b,
                                     const std::vector<TXZBufAcc> &x,
                                           std::vector<TYWBufAcc> &y,
                                           std::vector<TYWBufAcc> &w,
                                     const std::vector<TXZBufAcc> &z,
				           auto                   &reduce_bufs) {
      auto const nsrc = x.size();

      if (nsrc != nSrc)
        std::cout << "Incorrect number of sources\n" << std::endl;

      auto args = TransformReduceArgs<TXZBufAcc, TYWBufAcc, decltype(reduce_bufs), nSrc>(reduce_bufs, x, y, w, z);
      //
      Functor tr_func(a, b);

      return MultiSrcTransformReducer<Functor, decltype(args)>(tr_func, args);
    }
    

  }  // namespace blas
}  // namespace cms::alpakatools

#endif
