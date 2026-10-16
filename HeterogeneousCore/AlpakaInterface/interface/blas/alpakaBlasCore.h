#ifndef HeterogeneousCore_AlpakaInterface_interface_blas_alpakaBlasCore_h
#define HeterogeneousCore_AlpakaInterface_interface_blas_alpakaBlasCore_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/VecArray.h"
#include "HeterogeneousCore/AlpakaInterface/interface/blas/alpakaReducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/blas/alpakaBlockReduction.h"
#include "HeterogeneousCore/AlpakaInterface/interface/blas/alpakaReduceResource.h"

//namespace ALPAKA_ACCELERATOR_NAMESPACE::multiblas {
namespace cms::alpakatools::multiblas {

  using namespace ALPAKA_ACCELERATOR_NAMESPACE;

  template <typename ReduceT_, typename Init_, typename Reducer_, typename Transformer_, bool site_unroll_ = false>
  class TransformReduceFunctor {
  public:
    using ReduceT = ReduceT_;
    using InitT = Init_;
    using ReducerT = Reducer_;
    using TransformerT = Transformer_;

    InitT initializer;
    ReducerT reducer;
    TransformerT transformer;

    static constexpr bool site_unroll = site_unroll_;

    TransformReduceFunctor(InitT i, ReducerT r, TransformerT t) : initializer(i), reducer(r), transformer(t) {}

    static constexpr ReduceT Init() { return InitT{}(0.); }

    template <typename U>
    static constexpr ReduceT Init(U const &in) {
      return InitT{}(in);
    }

    ALPAKA_FN_ACC auto GetTransformer() const { return transformer; }

    ALPAKA_FN_ACC auto GetReducer() const { return reducer; }
  };

  template <typename XBufT,
            typename YBufT,
            typename reducer_params_t,
            unsigned long long nSrc_ = 1,
            bool host_reduction = true>
  class TransformReduceArgs {
  public:
    static constexpr unsigned long long nSrc = nSrc_;

    using DataT = std::remove_cvref_t<reducer_params_t>::DataT;
    using CountT = std::remove_cvref_t<reducer_params_t>::CountT;

    using SystemAtomicT = std::remove_cvref_t<reducer_params_t>::SystemAtomicT;
    using DeviceAtomicT = std::remove_cvref_t<reducer_params_t>::DeviceAtomicT;

    using Tx = typename alpaka::trait::ElemType<XBufT>::type;
    using Ty = typename alpaka::trait::ElemType<YBufT>::type;

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

    // Helper function to get compute range
    template <typename VecType>
    static auto get_max_range(VecType &vec) {
      auto const max_it =
          std::ranges::max_element(vec, {}, [](auto const &buffer) { return alpaka::getExtents(buffer).prod(); });

      return alpaka::getExtents(*max_it).prod();
    }

    DataT *result_h;
    SystemAtomicT *result_d;
    DeviceAtomicT *partial;

    const cms::alpakatools::VecArray<Tx *, nSrc> x;
    mutable cms::alpakatools::VecArray<Ty *, nSrc> y;

    CountT *count;

    reducer_params_t &reducer_params;

    const Idx max_range;

    TransformReduceArgs(reducer_params_t &params, const std::vector<XBufT> &x_, std::vector<YBufT> &y_)
        : result_h(host_reduction ? params.get_host_reduce_ptr() : nullptr),
          result_d(params.get_device_reduce_ptr()),
          partial(params.get_partial_ptr()),
          x(init_vec_array<XBufT>(x_)),
          y(init_vec_array<YBufT>(y_)),
          count(params.get_count_ptr()),
          reducer_params(params),
          max_range(get_max_range(y_)) {}

    TransformReduceArgs(reducer_params_t &params, Idx const compute_range)
        : result_h(host_reduction ? params.get_host_reduce_ptr() : nullptr),
          result_d(params.get_device_reduce_ptr()),
          partial(params.get_partial_ptr()),
          count(params.get_count_ptr()),
          reducer_params(params),
          max_range(compute_range) {}

    template <typename TQueue>
    auto fetch_data(TQueue queue) const {
      static_assert(host_reduction);
      reducer_params.fetch_reduce_data(queue);
    }

    auto get_reducer_limit() const { return reducer_params.get_partial_upper_limit(); }

    auto get_max_range() const { return max_range; }

    template <extfloat::ExtendedFloatType T>
    ALPAKA_FN_ACC void StorePartial(T const val, const std::uint32_t idx, const std::uint32_t stride = 0) const {
      static_assert(std::is_convertible_v<extfloat::floating_value_t<T>, DeviceAtomicT>);
      partial[idx] = static_cast<DeviceAtomicT>(reduce::HighComponent(val));
      partial[idx + stride] = static_cast<DeviceAtomicT>(reduce::LowComponent(val));
    }

    template <typename T>
      requires std::is_convertible_v<T, DeviceAtomicT>
    ALPAKA_FN_ACC void StorePartial(T const val, const std::uint32_t idx, const std::uint32_t stride = 0) const {
      partial[idx] = static_cast<DeviceAtomicT>(val);
    }

    template <extfloat::ExtendedFloatType T>
    ALPAKA_FN_ACC T LoadPartial(const std::uint32_t idx, const std::uint32_t stride = 0) const {
      using BaseFloat = typename extfloat::floating_value_t<T>;

      BaseFloat const val_hi = reduce::HighComponent(partial[idx]);
      BaseFloat const val_lo = reduce::LowComponent(partial[idx + stride]);

      return T(val_hi, val_lo);
    }

    template <typename T>
      requires std::is_convertible_v<T, DeviceAtomicT>
    ALPAKA_FN_ACC T LoadPartial(const std::uint32_t idx, const std::uint32_t stride = 0) const {
      T const val = static_cast<T>(partial[idx]);
      return val;
    }
  };

  template <typename TransformReducerT, typename Args>
  class MultiSrcTransformReducer {
  public:
    const Args args;

    TransformReducerT f;

    multireduce::BlockReducer block_reducer;

    MultiSrcTransformReducer(TransformReducerT f, const Args &args) : args(args), f(f) {}

    auto GetReducerLimit() const { return args.get_reducer_limit(); }

    auto GetMaxRange() const { return args.get_max_range(); }

    template <typename TQueue>
    auto Fetch(TQueue const &queue) const {
      args.template fetch_data<TQueue>(queue);
    }

    auto HostReducedValues() const {
      using DataT = typename Args::DataT;

      std::vector<DataT> values(Args::nSrc);

      for (Idx i = 0; i < Args::nSrc; i++) {
        values[i] = args.result_h[i];
      }

      return values;
    }

    template <typename TQueue>
    auto HostReducedValues(TQueue const &queue) const {
      using DataT = typename Args::DataT;

      Fetch(queue);

      std::vector<DataT> values(Args::nSrc);

      alpaka::wait(queue);

      for (Idx i = 0; i < Args::nSrc; i++) {
        values[i] = args.result_h[i];
      }

      return values;
    }

    template <typename TQueue>
    auto DeviceReducedValues(TQueue const &queue) const {
      using ReduceT = typename TransformReducerT::ReduceT;

      auto values = alpaka::allocBuf<ReduceT, Idx>(alpaka::getDev(queue), Args::nSrc);

      alpaka::memcpy(queue, values, args.result_d);

      return values;
    }

    //these are helper methods to return correct thread/block indices and dimensions:
    template <alpaka::concepts::Acc TAcc>
    ALPAKA_FN_ACC inline Vec2D threads_2d(TAcc const &acc) const {
      constexpr std::size_t nDim = alpaka::Dim<TAcc>::value;
      static_assert(alpaka::Dim<TAcc>::value <= 3u,
                    "The accelerator used for the Alpaka Kernel has to be at most 3 dimensional!");

      auto const exe_threads = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
      return Vec2D((nDim > 1 ? exe_threads[1] : 0), exe_threads[nDim - 1]);
    }

    template <alpaka::concepts::Acc TAcc>
    ALPAKA_FN_ACC inline Vec2D block_2d(TAcc const &acc) const {
      constexpr std::size_t nDim = alpaka::Dim<TAcc>::value;
      static_assert(alpaka::Dim<TAcc>::value <= 3u,
                    "The accelerator used for the Alpaka Kernel has to be at most 3 dimensional!");

      auto const exe_block_div = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
      return Vec2D((nDim > 1 ? exe_block_div[1] : 1), exe_block_div[nDim - 1]);
    }

    template <alpaka::concepts::Acc TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const &acc, std::uint32_t const begin, std::uint32_t const end) const {
      using ReducerT = typename TransformReducerT::ReducerT;
      using ReduceT = typename TransformReducerT::ReduceT;
      using TransformerT = typename TransformReducerT::TransformerT;

      // batch idx is always the slowest index:
      std::uint32_t const batch_idx = static_cast<Idx>(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0]);

      // Set leading dim:
      constexpr std::size_t lDim = alpaka::Dim<TAcc>::value - 1;  // leading dimension

      auto const blockIdx_x = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[lDim];
      auto const [threadIdx_y, threadIdx_x] = threads_2d(acc);

      auto const gridDim_x = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[lDim];
      auto const [blockDim_y, blockDim_x] = block_2d(acc);

      TransformerT transformer = f.GetTransformer();

      auto i(static_cast<std::uint32_t>(threadIdx_x + blockIdx_x * blockDim_x) + begin);

      using Xtype = std::remove_cvref_t<decltype(args.x)>;
      using Ytype = std::remove_cvref_t<decltype(args.y)>;

      static_assert((cms::alpakatools::is_VecArray_v<Xtype> and cms::alpakatools::is_VecArray_v<Ytype>),
                    "All arguments must be of type cms::alpakatools::VecArray<T, N>.");

      ReducerT reducer = f.GetReducer();

      auto out = TransformReducerT::Init();

      while (i < end) {
        auto reduce_val = transformer(acc, args.x, args.y, i, 0, batch_idx);

        out = reducer(out, block_reducer.template apply<TAcc, TransformReducerT>(acc, batch_idx, reduce_val, f, true));
        i += gridDim_x * blockDim_x;
      }

      auto &isLastBlockDone = alpaka::declareSharedVar<bool[Args::nSrc], __COUNTER__>(acc);

      using CountT = typename Args::CountT;

      CountT *count = static_cast<CountT *>(args.count);

      if (threadIdx_x == 0 && threadIdx_y == 0) {
        auto const blockDim_z = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0];

        args.StorePartial(out, blockIdx_x + gridDim_x * batch_idx, gridDim_x * blockDim_z);

        alpaka::mem_fence(acc, alpaka::memory_scope::Device{});
        unsigned int value =
            alpaka::atomicAdd(acc, static_cast<CountT *>(&count[batch_idx]), 1, alpaka::hierarchy::Blocks{});

        isLastBlockDone[batch_idx] = (value == (gridDim_x - 1));
      }

      alpaka::syncBlockThreads(acc);

      if (isLastBlockDone[batch_idx]) {
        auto s = threadIdx_y * blockDim_x + threadIdx_x;

        auto accum = TransformReducerT::Init();

        auto const blockDim_z = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0];

        while (s < gridDim_x) {
          accum = reducer(
              accum, args.template LoadPartial<decltype(accum)>(batch_idx * gridDim_x + s, gridDim_x * blockDim_z));

          s += blockDim_x * blockDim_y;
        }

        out = block_reducer.template apply<TAcc, TransformReducerT>(acc, batch_idx, accum, f, true);

        if (threadIdx_x == 0 && threadIdx_y == 0) {
          auto d_result = args.result_d;
          d_result[batch_idx] = reduce::ReducerResult(out);
        }
      }
    }
  };

  template <std::size_t nSrc,
            typename TXBufAcc,
            typename TYBufAcc,
            typename ReduceT,
            typename Init,
            typename Reducer,
            typename Transformer>
  auto instantiateTransformReducer(auto &reduce_bufs,
                                   std::vector<TXBufAcc> const &x,
                                   std::vector<TYBufAcc> &y,
                                   Init init,
                                   Reducer reducer,
                                   Transformer transformer) {
    using TransformReduceT = TransformReduceFunctor<ReduceT, Init, Reducer, Transformer, false>;

    using ArgsT = TransformReduceArgs<TXBufAcc, TYBufAcc, decltype(reduce_bufs), nSrc>;

    TransformReduceT transform_reduce_func{init, reducer, transformer};

    ArgsT args{reduce_bufs, x, y};

    return MultiSrcTransformReducer<TransformReduceT, ArgsT>{transform_reduce_func, args};
  }
}  // namespace cms::alpakatools::multiblas

#endif
