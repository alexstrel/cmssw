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

template <alpaka::concepts::Acc TAcc,
          typename Queue,
          std::size_t NSrc,
          bool CopyToHost,
          bool CreateResources,
          bool DoSiteUnroll>
struct TransformReducePolicy {
  static_assert(NSrc > 0, "Number of src must be positive");
  static_assert(NSrc <= 16, "Number of src exceeds the maximum allowed value 16");
  static_assert(NSrc == 1 || (NSrc > 1 && std::same_as<std::remove_cvref_t<TAcc>, Acc3D>),
                "TAcc must be 3-dimensional for NSrc > 1");

  using acc_type = TAcc;
  using queue_type = Queue;

  Queue &queue;  // WARNING: dangling reference risk!

  static constexpr std::size_t nSrc = NSrc;
  static constexpr bool copy_to_host = CopyToHost;
  static constexpr bool create_resources = CreateResources;
};

template <alpaka::concepts::Acc TAcc,
          std::size_t NSrc,
          bool CopyToHost,
          bool CreateResources,
          bool DoSiteUnroll,
          typename Queue>
auto make_transform_reduce_policy(Queue &queue) {
  return TransformReducePolicy<TAcc, Queue, NSrc, CopyToHost, CreateResources, DoSiteUnroll>{queue};
}

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

template <typename TPolicy,
          typename TXBufAcc,
          typename TYBufAcc,
          typename reduce_t,
          typename Init,
          typename Reducer,
          typename Transformer>
auto transform_reduce(TPolicy &policy,
                      Idx const begin,
                      Idx const end,
                      const std::vector<TXBufAcc> &x,
                      std::vector<TYBufAcc> &y,
                      Init init,
                      Reducer reducer,
                      Transformer transformer) {
  using acc_t = typename TPolicy::acc_type;
  using queue_t = typename TPolicy::queue_type;

  using transformer_t = Transformer;
  using reducer_t = Reducer;

  constexpr std::size_t nSrc = TPolicy::nSrc;

  if (y.size() != nSrc || x.size() != nSrc) {
    throw std::invalid_argument("launchTransformReduce(): x and y must contain nSrc buffers");
  }

  auto const devAcc = alpaka::getDev(policy.queue);
  Idx const max_threads_per_block = static_cast<Idx>(alpaka::getAccDevProps<acc_t>(devAcc).m_blockThreadCountMax);

  std::uint32_t const wExtend = alpaka::getPreferredWarpSize(alpaka::getDev(policy.queue));
  Idx const max_block_x_dim = ((max_threads_per_block / static_cast<Idx>(nSrc)) / wExtend) * wExtend;

  [[maybe_unused]] auto reduce_bufs = [&]() {
    if constexpr (TPolicy::create_resources) {
      return cms::alpakatools::reduce::create_reduction_resources<acc_t, queue_t, reduce_t>(policy.queue, nSrc);
    } else {
      return std::monostate{};
    }
  }();

  auto msrc_tr =
      multiblas::instantiateTransformReducer<acc_t, queue_t, decltype(reduce_bufs), nSrc, TXBufAcc, TYBufAcc, reduce_t>(
          policy.queue, reduce_bufs, x, y, init, reducer, transformer);
  Idx const range = begin - end;

  assert(range > msrc_tr.get_max_range());

  std::cout << "Reduction buffer limit:  " << msrc_tr.get_reducer_limit() << std::endl;

  std::cout << "Test init:  " << init(101.01) << std::endl;

  Idx const block_x_dim = range < max_block_x_dim ? (range / wExtend) * wExtend : max_block_x_dim;

  Idx const max_grid_x_dim = (range + block_x_dim - 1) / block_x_dim;

  Idx const grid_x_dim =
      max_grid_x_dim * nSrc <= msrc_tr.get_reducer_limit() ? max_grid_x_dim : msrc_tr.get_reducer_limit() / nSrc;

  std::cout << "Launch configuration: x dim blocks " << block_x_dim << ", x dim grid  " << grid_x_dim << std::endl;

  Vec3D const grid_size{1, 1, grid_x_dim};
  Vec3D const block_size{nSrc, 1, block_x_dim};

  alpaka::WorkDivMembers<Dim3D, Idx> workDiv{grid_size, block_size, Vec3D::ones()};

  TransformReduceKernel<acc_t, std::remove_cvref_t<decltype(msrc_tr)>> tr_compute_kernel(begin, end);

  alpaka::exec<acc_t>(policy.queue, workDiv, tr_compute_kernel, msrc_tr);

  if constexpr (TPolicy::copy_to_host) {
    return msrc_tr.template host_reduced_values<queue_t>(policy.queue);
  } else {
    return msrc_tr.template device_reduced_values<queue_t>(policy.queue);
  }
}

#endif
