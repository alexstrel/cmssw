#ifndef HeterogeneousCore_AlpakaInterface_interface_blas_alpakaBlockReduction_h
#define HeterogeneousCore_AlpakaInterface_interface_blas_alpakaBlockReduction_h

#include "HeterogeneousCore/AlpakaInterface/interface/VecArray.h"
#include "HeterogeneousCore/AlpakaInterface/interface/blas/AtomicType.h"
#include "HeterogeneousCore/AlpakaInterface/interface/blas/BlasHelpers.h"

#include <utility>

namespace ALPAKA_ACCELERATOR_NAMESPACE::multireduce {

  using namespace cms::alpakatools;

  template <typename Reducer>
  concept CompensatedSum = requires {
    { std::remove_cvref_t<Reducer>::use_compensated_sum } -> std::convertible_to<bool>;
  };

  template <typename T, typename TransformReducer>
  concept AtomicReduceData =
      std::same_as<std::remove_cvref_t<T>, cms::alpakatools::atomic_type_t<typename TransformReducer::ReduceT>> ||
      std::same_as<std::remove_cvref_t<T>,
                   cms::alpakatools::VecArray<cms::alpakatools::atomic_type_t<typename TransformReducer::ReduceT>, 2>>;

  template <typename Reducer>
  inline constexpr bool use_compensated_sum_v =
      CompensatedSum<Reducer> ? static_cast<bool>(std::remove_cvref_t<Reducer>::use_compensated_sum) : false;

  template <typename TransformReducer, typename T, bool use_sloppy_reduce_ = false>
    requires(AtomicReduceData<T, TransformReducer> || std::floating_point<std::remove_cvref_t<T>>)
  constexpr auto init_accum(T const& in) {
    using ReducerT = typename TransformReducer::ReducerT;
    using AtomicT = cms::alpakatools::atomic_type_t<T>;

    constexpr bool use_sloppy_reduce = use_compensated_sum_v<ReducerT> & use_sloppy_reduce_;

    if constexpr (use_sloppy_reduce) {
      using kahan_accumulator_t = reduce::kahan_accumulator<AtomicT>;

      kahan_accumulator_t r{};

      if constexpr (std::is_same_v<T, kahan_accumulator_t>) {
        r = in;
      } else {
        reduce::accum(r) = reduce::set<AtomicT>(in);
        reduce::compensation(r) = reduce::zero<AtomicT>();
      }

      return r;
    } else {
      return reduce::set<AtomicT, T>(in);
    }
  }

  class WarpReducer {
  public:
    WarpReducer() = default;

    template <alpaka::concepts::Acc TAcc, typename DataT, typename TransformReducerT, bool use_sloppy_reduce_ = false>
      requires AtomicReduceData<DataT, TransformReducerT>
    ALPAKA_FN_ACC auto apply(TAcc const& acc,
                             DataT const& in,
                             const TransformReducerT f,
                             unsigned int const laneIdx,
                             bool all = true) const {
      using ReducerT = typename TransformReducerT::ReducerT;
      using AtomicT = cms::alpakatools::atomic_type_t<typename TransformReducerT::ReduceT>;

      constexpr bool use_sloppy_reduce = use_compensated_sum_v<ReducerT> & use_sloppy_reduce_;

      using accum_t = std::conditional_t<use_sloppy_reduce, reduce::kahan_accumulator<AtomicT>, AtomicT>;

      // Deduce number of the atomic elements
      auto accum = init_accum<TransformReducerT, AtomicT, use_sloppy_reduce>(in);

      ReducerT r = f.get_reducer();

      if (isFullWarp(acc)) {
        accum = all ? warpReduce<TAcc, AtomicT, ReducerT, true>(acc, accum, r)
                    : warpReduce<TAcc, AtomicT, ReducerT, false>(acc, accum, r);
      } else {
        accum =
            all ? warpSparseReduce<TAcc, AtomicT, ReducerT, true>(acc, alpaka::warp::activemask(acc), laneIdx, accum, r)
                : warpSparseReduce<TAcc, AtomicT, ReducerT, false>(
                      acc, alpaka::warp::activemask(acc), laneIdx, accum, r);
      }

      return accum;
    }

    template <alpaka::concepts::Acc TAcc, typename DataT, typename TransformReducerT, bool use_sloppy_reduce = false>
    ALPAKA_FN_ACC auto apply(
        TAcc const& acc, DataT const& in, const TransformReducerT f, unsigned int const laneIdx, bool all = true) const
      requires VecArrayType<DataT> && (std::remove_cvref_t<DataT>::N >= 3)
    {
      using AtomicT = cms::alpakatools::atomic_type_t<DataT>;

      using accum_t = decltype(init_accum<TransformReducerT, AtomicT, use_sloppy_reduce>(in[0]));

      constexpr std::size_t n_elems = DataT::N;

      cms::alpakatools::VecArray<accum_t, n_elems> accum{};

      if constexpr (std::same_as<std::remove_cvref_t<typename DataT::value_t>, std::remove_cvref_t<accum_t>>) {
        accum = in;
      } else {
        CMS_UNROLL_LOOP
        for (std::size_t i = 0; i < n_elems; i++) {
          accum[i] = init_accum<TransformReducerT, AtomicT, use_sloppy_reduce>(in[i]);
        }
      }

      CMS_UNROLL_LOOP
      for (std::size_t i = 0; i < n_elems; i++)
        accum[i] = apply<TAcc, accum_t, TransformReducerT>(acc, accum[i], f, laneIdx, all);

      return accum;
    }
  };

  class BlockReducer {
  public:
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

      constexpr bool use_sloppy_reduction = use_compensated_sum_v<typename TransformReducerT::ReducerT>;

      // Perform warp reduction using shuffle operations
      auto res = warp_reducer.template apply<TAcc, ReduceT, TransformReducerT>(acc, in, f, laneIdx, all);

      using AtomicT = cms::alpakatools::atomic_type_t<ReduceT>;

      constexpr std::size_t n_elements = cms::alpakatools::n_atomic_elements<ReduceT>();

      ReduceT out{};  //either float type or VecArray<float, N> type

      if (all && w_items == 1) {  // short circuit for single warp CTA

        if constexpr (n_elements == 1) {  //
          static_assert(std::same_as<ReduceT, AtomicT>);

          out = reduce::result<decltype(res), use_sloppy_reduction>(res);

        } else {
          static_assert(is_VecArray_v<ReduceT>);

          if constexpr (use_sloppy_reduction) {
            cms::alpakatools::VecArray<AtomicT, n_elements> res_tmp{};

            CMS_UNROLL_LOOP
            for (std::size_t i = 0; i < n_elements; i++) {
              res_tmp[i] = reduce::result<decltype(res[i]), true>(res[i]);
            }

            memcpy(out.data(), res_tmp.data(), sizeof(ReduceT::value_t) * n_elements);

          } else {
            memcpy(out.data(), res.data(), sizeof(ReduceT::value_t) * n_elements);
          }
        }
        return out;
      }

      using accum_t =
          typename std::conditional<use_sloppy_reduction, reduce::kahan_accumulator<AtomicT>, AtomicT>::type;

      unsigned int const w_items_x =
          alpaka::math::min(acc, static_cast<int>(blockDim_x / w_extent), static_cast<int>(max_w_items));

      constexpr std::size_t n = max_w_items * n_elements;
      constexpr std::size_t buffer_alignment = 64;
      constexpr std::size_t buffer_size = n * sizeof(accum_t);

      using smem_t = reduce::SmemBuffer<buffer_size, buffer_alignment>;
      auto& sdata(alpaka::declareSharedVar<smem_t, __COUNTER__>(acc));

      auto const blockDim_z = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0];
      // Write the reduced sum of each warp to shared memory
      if (laneIdx == 0) {
        auto smem_cache = sdata.template as<AtomicT>();
        if constexpr (n_elements == 1) {
          smem_cache[warpIdx_x + batch * w_items_x] = reduce::result<decltype(res), use_sloppy_reduction>(res);
          if constexpr (use_sloppy_reduction) {
            const std::size_t offset = w_items_x * blockDim_z;
            smem_cache[warpIdx_x + batch * w_items_x + offset] = reduce::compensation(res);
          }
        } else {
          const std::size_t stride = w_items_x * blockDim_z;
          CMS_UNROLL_LOOP
          for (std::size_t i = 0; i < n_elements; i++) {
            smem_cache[warpIdx_x + batch * w_items_x + i * stride] =
                reduce::result<decltype(res[i]), use_sloppy_reduction>(res[i]);
            if constexpr (use_sloppy_reduction) {
              const std::size_t offset = w_items_x * blockDim_z * n_elements;
              smem_cache[warpIdx_x + batch * w_items_x + i * stride + offset] = reduce::compensation(res[i]);
            }
          }
        }
      }

      alpaka::syncBlockThreads(acc);

      // Reduce the results from all warps (assuming blockDim.x / warpSize warps per block)
      if (threadIdx_x < w_items_x) {
        auto smem_cache = sdata.template as<AtomicT>();
        if constexpr (n_elements == 1) {
          if constexpr (use_sloppy_reduction) {
            reduce::accum(res) = smem_cache[threadIdx_x + batch * w_items_x];
            const std::size_t offset = w_items_x * blockDim_z;
            reduce::compensation(res) = smem_cache[threadIdx_x + batch * w_items_x + offset];
          } else {
            res = smem_cache[threadIdx_x + batch * w_items_x];
          }
        } else {
          const std::size_t stride = w_items_x * blockDim_z;
          CMS_UNROLL_LOOP
          for (std::size_t i = 0; i < n_elements; i++) {
            const std::size_t stride = w_items_x * blockDim_z;
            if constexpr (use_sloppy_reduction) {
              reduce::accum(res[i]) = smem_cache[threadIdx_x + batch * w_items_x + i * stride];
              const std::size_t offset = w_items_x * blockDim_z * n_elements;
              reduce::compensation(res[i]) = smem_cache[threadIdx_x + batch * w_items_x + i * stride + offset];
            } else {
              res[i] = smem_cache[threadIdx_x + batch * w_items_x + i * stride];
            }
          }
        }

        res = warp_reducer.template apply<TAcc, decltype(res), TransformReducerT>(acc, res, f, laneIdx, all);
      }

      alpaka::syncBlockThreads(acc);
      //return res;
      if (all) {
        auto smem_cache = sdata.template as<AtomicT>();

        if (threadIdx_x == 0) {
          if constexpr (n_elements == 1) {
            smem_cache[0 + batch * w_items_x] = reduce::result<decltype(res), use_sloppy_reduction>(res);
          } else {
            CMS_UNROLL_LOOP
            for (std::size_t i = 0; i < n_elements; i++) {
              smem_cache[0 + batch * w_items_x + i * w_items_x * blockDim_z] =
                  reduce::result<decltype(res[i]), use_sloppy_reduction>(res[i]);
            }
          }
        }
        alpaka::syncBlockThreads(acc);

        if constexpr (n_elements == 1) {
          out = smem_cache[0 + batch * w_items_x];
        } else {
          CMS_UNROLL_LOOP
          for (std::size_t i = 0; i < n_elements; i++) {
            out[i] = smem_cache[0 + batch * w_items_x + i * w_items_x * blockDim_z];
          }
        }
      }

      return out;
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::multireduce
#endif
