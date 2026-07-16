#ifndef HeterogeneousCore_AlpakaInterface_interface_blas_alpakaTransformReduceKernel_h
#define HeterogeneousCore_AlpakaInterface_interface_blas_alpakaTransformReduceKernel_h

#include <iostream>
#include <type_traits>

#include <ranges>
#include <variant>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/blas/alpakaBlasCore.h"
#include "HeterogeneousCore/AlpakaInterface/interface/blas/alpakaReduceResource.h"

using namespace cms::alpakatools;
using namespace ALPAKA_ACCELERATOR_NAMESPACE;

template <alpaka::concepts::Acc TAcc, typename TransformReducer>
class TransformReduceKernel {
public:
  ALPAKA_FN_ACC auto operator()(TAcc const &acc, TransformReducer const &transform_reducer) const -> void {
    // batch idx is always the slowest index
    auto const batch_idx = static_cast<Idx>(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0]);
    transform_reducer.apply(acc, batch_idx);
  }
};

template <typename TQueue,
          typename Transformer,
          typename reduce_t,
          typename Reducer,
          std::size_t nSrc,
          bool host_values,
          bool create_resources,
          typename TXBufAcc,
          typename TYBufAcc,
          typename... coeff_t>
auto transform_reduce(TQueue &queue, const std::vector<TXBufAcc> &x, std::vector<TYBufAcc> &y, coeff_t const &...a) {
  using transformer_t = Transformer;
  using reducer_t = Reducer;

  if (y.size() != nSrc || x.size() != nSrc) {
    throw std::invalid_argument("launchTransformReduce(): x and y must contain nSrc buffers");
  }

  static_assert(nSrc > 0, "nSrc must be positive");
  static_assert(nSrc <= 16, "nSrc exceeds the maximum allowed value 16");

  auto const max_it =
      std::ranges::max_element(y, {}, [](auto const &buffer) { return alpaka::getExtents(buffer).prod(); });

  Idx const N = alpaka::getExtents(*max_it).prod();

  const std::uint32_t wExtend = alpaka::getPreferredWarpSize(alpaka::getDev(queue));

  const Idx block_x_dim = (Idx{1024} / static_cast<Idx>(nSrc) / wExtend) * wExtend;

  Idx const grid_x_dim = (N + block_x_dim - 1) / block_x_dim;

  Vec3D const grid_size{1, 1, grid_x_dim};
  Vec3D const block_size{nSrc, 1, block_x_dim};

  alpaka::WorkDivMembers<Dim3D, Idx> workDiv{grid_size, block_size, Vec3D::ones()};

  [[maybe_unused]] auto reduce_bufs = [&]() {
    if constexpr (create_resources) {
      return cms::alpakatools::reduce::create_reduction_resources<Acc3D, TQueue, reduce_t>(queue, nSrc);
    } else {
      return std::monostate{};
    }
  }();

  auto msrc_functor = multiblas::instantiateTransformReducer<Acc3D,
                                                             TQueue,
                                                             transformer_t,
                                                             reduce_t,
                                                             reducer_t,
                                                             decltype(reduce_bufs),
                                                             nSrc,
                                                             TXBufAcc,
                                                             TYBufAcc,
                                                             coeff_t...>(queue, reduce_bufs, x, y, a...);

  TransformReduceKernel<Acc3D, std::remove_cvref_t<decltype(msrc_functor)>> tr_compute_kernel;

  alpaka::exec<Acc3D>(queue, workDiv, tr_compute_kernel, msrc_functor);

  if constexpr (host_values) {
    return msrc_functor.template host_reduced_values<TQueue>(queue);
  } else {
    return msrc_functor.template device_reduced_values<TQueue>(queue);
  }
}

#endif
