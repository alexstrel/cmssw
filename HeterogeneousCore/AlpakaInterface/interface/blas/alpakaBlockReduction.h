#ifndef HeterogeneousCore_AlpakaInterface_interface_blas_alpakaBlockReduction_h
#define HeterogeneousCore_AlpakaInterface_interface_blas_alpakaBlockReduction_h

#include "HeterogeneousCore/AlpakaInterface/interface/VecArray.h"
#include "HeterogeneousCore/AlpakaInterface/interface/blas/AtomicType.h"
#include "HeterogeneousCore/AlpakaInterface/interface/blas/BlasHelpers.h"

#include <utility>

namespace ALPAKA_ACCELERATOR_NAMESPACE::multireduce {

  using namespace cms::alpakatools;

  template <typename Reducer>
  concept AugmentedReducer = requires {
    { std::remove_cvref_t<Reducer>::use_augmented_reduction } -> std::convertible_to<bool>;
  };

  template <typename Reducer>
  inline constexpr bool use_augmented_reduction_v =
      AugmentedReducer<Reducer> ? static_cast<bool>(std::remove_cvref_t<Reducer>::use_augmented_reduction) : false;

  template <typename T, typename TransformReducer>
  concept AtomicReduceData =
      std::same_as<std::remove_cvref_t<T>, cms::alpakatools::atomic_type_t<typename TransformReducer::ReduceT>> ||
      std::same_as<std::remove_cvref_t<T>,
                   cms::alpakatools::VecArray<cms::alpakatools::atomic_type_t<typename TransformReducer::ReduceT>, 2>>;

  class WarpReducer {
  public:
    WarpReducer() = default;

    template <alpaka::concepts::Acc TAcc, typename DataT, typename TransformReducerT>
      requires(AtomicReduceData<typename TransformReducerT::ReduceT, TransformReducerT> ||
               extfloat::ExtendedFloatType<typename TransformReducerT::ReduceT>)
    ALPAKA_FN_ACC auto apply(TAcc const& acc,
                             DataT const& in,
                             const TransformReducerT f,
                             unsigned int const laneIdx,
                             bool all = false) const {
      using ReducerT = typename TransformReducerT::ReducerT;
      using ReduceT = typename TransformReducerT::ReduceT;

      auto accum = TransformReducerT::Init(in);

      ReducerT r = f.GetReducer();

      if (isFullWarp(acc)) {
        accum = all ? warpReduce<TAcc, ReduceT, ReducerT, true>(acc, accum, r)
                    : warpReduce<TAcc, ReduceT, ReducerT, false>(acc, accum, r);
      } else {
        accum =
            all ? warpSparseReduce<TAcc, ReduceT, ReducerT, true>(acc, alpaka::warp::activemask(acc), laneIdx, accum, r)
                : warpSparseReduce<TAcc, ReduceT, ReducerT, false>(
                      acc, alpaka::warp::activemask(acc), laneIdx, accum, r);
      }

      return accum;
    }

    template <alpaka::concepts::Acc TAcc, typename DataT, typename TransformReducerT, bool use_augmented_reduction = false>
    ALPAKA_FN_ACC auto apply(
        TAcc const& acc, DataT const& in, const TransformReducerT f, unsigned int const laneIdx, bool all = true) const
      requires VecArrayType<DataT> && (std::remove_cvref_t<DataT>::N <= 16)
    {
      using accum_t = cms::alpakatools::atomic_type_t<DataT>;

      constexpr std::size_t n_elems = DataT::N;

      cms::alpakatools::VecArray<accum_t, n_elems> accum{};

      if constexpr (std::same_as<std::remove_cvref_t<typename DataT::value_t>, std::remove_cvref_t<accum_t>>) {
        accum = in;
      } else {
        CMS_UNROLL_LOOP
        for (std::size_t i = 0; i < n_elems; i++) {
          accum[i] = TransformReducerT::Init(in[i]);
        }
      }

      CMS_UNROLL_LOOP
      for (std::size_t i = 0; i < n_elems; i++)
        accum[i] = apply<TAcc, accum_t, TransformReducerT>(acc, accum[i], f, laneIdx, all);

      return accum;
    }
  };

  template <std::uint32_t nSrc_>
  class BlockReducer {
  public:
    static constexpr std::uint32_t nSrc = nSrc_;

    BlockReducer() = default;

    WarpReducer warp_reducer;

    template <alpaka::concepts::Acc TAcc, typename TransformReducerT>
    ALPAKA_FN_ACC inline auto apply(TAcc const& acc,
                                    int const batch,
                                    typename TransformReducerT::ReduceT const& in,
                                    const TransformReducerT f,
                                    bool all = true) const -> typename TransformReducerT::ReduceT {
      using ReduceT = typename TransformReducerT::ReduceT;

      constexpr int w_extent = getWarpSize<TAcc>();

      auto const blockExtent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);

      constexpr unsigned int max_w_items = getWarpSize<TAcc>();

      int const w_items =
          alpaka::math::min(acc, static_cast<int>(blockExtent.prod() / w_extent), static_cast<int>(max_w_items));

      // we need to know block dimensionality to deduce the leading dimension
      constexpr auto nDim = alpaka::Dim<TAcc>::value;

      constexpr std::size_t leading_dim = nDim - 1;

      auto const threadIdx_x = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[leading_dim];
      auto const blockDim_x = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[leading_dim];

      unsigned int const warpIdx_x = threadIdx_x / w_extent;
      unsigned int const laneIdx = threadIdx_x % w_extent;

      // Perform warp reduction using shuffle operations
      auto res = warp_reducer.template apply<TAcc, ReduceT, TransformReducerT>(acc, in, f, laneIdx, false);

      if (all && w_items == 1) {  // short circuit for single warp CTA
        return res;
      }

      using accum_t = typename decltype([] {
        if constexpr (extfloat::ExtendedFloatType<ReduceT>) {
          return std::type_identity<extfloat::floating_value_t<ReduceT>>{};
        } else {
          return std::type_identity<atomic_type_t<ReduceT>>{};
        }
      }())::type;

      unsigned int const w_items_x =
          alpaka::math::min(acc, static_cast<int>(blockDim_x / w_extent), static_cast<int>(max_w_items));

      constexpr std::size_t n_elements = [] {
        if constexpr (extfloat::ExtendedFloatType<ReduceT>) {
          return std::size_t{2};
        } else {
          return n_atomic_elements<ReduceT>();
        }
      }();

      constexpr std::size_t n = max_w_items * n_elements;
      constexpr std::size_t buffer_alignment = 64;
      constexpr std::size_t buffer_size = nSrc * n * sizeof(accum_t);

      using smem_t = reduce::SmemBuffer<buffer_size, buffer_alignment>;
      auto& sdata(alpaka::declareSharedVar<smem_t, __COUNTER__>(acc));

      auto const blockDim_z = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0];
      // Write the reduced sum of each warp to shared memory
      if (laneIdx == 0) {
        auto smem_cache = sdata.template as<accum_t>();
        if constexpr (std::is_floating_point_v<decltype(res)> || extfloat::is_extended_float<decltype(res)>::value) {
          smem_cache[warpIdx_x + batch * w_items_x] = reduce::HighComponent(res);
          if constexpr (extfloat::is_extended_float<decltype(res)>::value) {
            constexpr std::size_t stride = n / 2;
            smem_cache[warpIdx_x + batch * w_items_x + stride] = reduce::LowComponent(res);
          }
        } else {
          const std::size_t stride = w_items_x * blockDim_z;
          CMS_UNROLL_LOOP
          for (std::size_t i = 0; i < n_elements; i++) {
            smem_cache[warpIdx_x + batch * w_items_x + i * stride] = res[i];
          }
        }
      }

      alpaka::syncBlockThreads(acc);

      // Reduce the results from all warps (assuming blockDim.x / warpSize warps per block)
      if (threadIdx_x < w_items_x) {
        auto smem_cache = sdata.template as<accum_t>();

        if constexpr (std::is_floating_point_v<decltype(res)>) {
          res = smem_cache[threadIdx_x + batch * w_items_x];
        } else if constexpr (extfloat::is_extended_float<decltype(res)>::value) {
          constexpr std::size_t stride = n / 2;

          accum_t const h = smem_cache[threadIdx_x + batch * w_items_x];
          accum_t const l = smem_cache[threadIdx_x + batch * w_items_x + stride];

          res = ReduceT{h, l};
        } else {
          const std::size_t stride = w_items_x * blockDim_z;

          CMS_UNROLL_LOOP
          for (std::size_t i = 0; i < n_elements; i++) {
            res[i] = smem_cache[threadIdx_x + batch * w_items_x + i * stride];
          }
        }

        res = warp_reducer.template apply<TAcc, decltype(res), TransformReducerT>(acc, res, f, laneIdx, false);
      }

      alpaka::syncBlockThreads(acc);

      if (all) {
        auto smem_cache = sdata.template as<accum_t>();

        if (threadIdx_x == 0) {
          if constexpr (std::is_floating_point_v<decltype(res)> || extfloat::is_extended_float<decltype(res)>::value) {
            smem_cache[0 + batch * w_items_x] = reduce::HighComponent(res);
            if constexpr (extfloat::is_extended_float<decltype(res)>::value) {
              constexpr std::size_t stride = n / 2;

              smem_cache[0 + batch * w_items_x + stride] = reduce::LowComponent(res);
            }
          } else {
            const std::size_t stride = w_items_x * blockDim_z;

            CMS_UNROLL_LOOP
            for (std::size_t i = 0; i < n_elements; i++) {
              smem_cache[0 + batch * w_items_x + i * stride] = res[i];
            }
          }
        }
        alpaka::syncBlockThreads(acc);

        if constexpr (std::is_floating_point_v<decltype(res)>) {
          res = smem_cache[0 + batch * w_items_x];
        } else if constexpr (extfloat::is_extended_float<decltype(res)>::value) {
          constexpr std::size_t stride = n / 2;
          accum_t const h = smem_cache[0 + batch * w_items_x];
          accum_t const l = smem_cache[0 + batch * w_items_x + stride];
          res = ReduceT{h, l};
        } else {
          CMS_UNROLL_LOOP
          for (std::size_t i = 0; i < n_elements; i++) {
            res[i] = smem_cache[0 + batch * w_items_x + i * w_items_x * blockDim_z];
          }
        }
      }
      return res;
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::multireduce
#endif
