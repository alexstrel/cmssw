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
  const Idx begin_offset;
  const Idx end;

public:
  TransformReduceKernel(Idx const offset, Idx const end) : begin_offset(offset), end(end) {}
  ALPAKA_FN_ACC auto operator()(TAcc const &acc, TransformReducer const &transform_reducer) const -> void {
    // batch idx is always the slowest index
    auto const batch_idx = static_cast<Idx>(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0]);

    transform_reducer.apply(acc, batch_idx, begin_offset, end);
  }
};

template <typename TQueue,
          std::size_t nSrc,
          bool host_values,
          bool create_resources,
          typename TXBufAcc,
          typename TYBufAcc,
          typename reduce_t,
          typename Reducer,
          typename Transformer>
auto transform_reduce(TQueue &queue,
                      Idx const begin,
                      Idx const end,
                      const std::vector<TXBufAcc> &x,
                      std::vector<TYBufAcc> &y,
                      Reducer reducer,
                      Transformer transformer) {
  using transformer_t = Transformer;
  using reducer_t = Reducer;

  if (y.size() != nSrc || x.size() != nSrc) {
    throw std::invalid_argument("launchTransformReduce(): x and y must contain nSrc buffers");
  }

  static_assert(nSrc > 0, "nSrc must be positive");
  static_assert(nSrc <= 16, "nSrc exceeds the maximum allowed value 16");

  auto const devAcc = alpaka::getDev(queue);
  Idx const max_threads_per_block = static_cast<Idx>(alpaka::getAccDevProps<Acc3D>(devAcc).m_blockThreadCountMax);

  std::uint32_t const wExtend = alpaka::getPreferredWarpSize(alpaka::getDev(queue));
  Idx const max_block_x_dim = ((max_threads_per_block / static_cast<Idx>(nSrc)) / wExtend) * wExtend;

  [[maybe_unused]] auto reduce_bufs = [&]() {
    if constexpr (create_resources) {
      return cms::alpakatools::reduce::create_reduction_resources<Acc3D, TQueue, reduce_t>(queue, nSrc);
    } else {
      return std::monostate{};
    }
  }();

  auto msrc_tr =
      multiblas::instantiateTransformReducer<Acc3D, TQueue, decltype(reduce_bufs), nSrc, TXBufAcc, TYBufAcc, reduce_t>(
          queue, reduce_bufs, x, y, reducer, transformer);
  Idx const range = begin - end;

  assert(range > msrc_tr.get_max_range());

  std::cout << "Reduction buffer limit:  " << msrc_tr.get_reducer_limit() << std::endl;

  Idx const block_x_dim = range < max_block_x_dim ? (range / wExtend) * wExtend : max_block_x_dim;

  Idx const max_grid_x_dim = (range + block_x_dim - 1) / block_x_dim;

  Idx const grid_x_dim =
      max_grid_x_dim * nSrc <= msrc_tr.get_reducer_limit() ? max_grid_x_dim : msrc_tr.get_reducer_limit() / nSrc;

  std::cout << "Launch configuration: x dim blocks " << block_x_dim << ", x dim grid  " << grid_x_dim << std::endl;

  Vec3D const grid_size{1, 1, grid_x_dim};
  Vec3D const block_size{nSrc, 1, block_x_dim};

  alpaka::WorkDivMembers<Dim3D, Idx> workDiv{grid_size, block_size, Vec3D::ones()};

  TransformReduceKernel<Acc3D, std::remove_cvref_t<decltype(msrc_tr)>> tr_compute_kernel(begin, end);

  alpaka::exec<Acc3D>(queue, workDiv, tr_compute_kernel, msrc_tr);

  if constexpr (host_values) {
    return msrc_tr.template host_reduced_values<TQueue>(queue);
  } else {
    return msrc_tr.template device_reduced_values<TQueue>(queue);
  }
}

#endif
