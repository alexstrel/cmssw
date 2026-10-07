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

  using AccT = TAcc;
  using QueueT = Queue;

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

template <typename TPolicy, typename TXBufAcc, typename TYBufAcc, typename Init, typename Reducer, typename Transformer>
auto transform_reduce(TPolicy &policy,
                      Idx const begin,
                      Idx const end,
                      const std::vector<TXBufAcc> &in,
                      std::vector<TYBufAcc> &out,
                      Init init,
                      Reducer reducer,
                      Transformer transformer) {
  using AccT = typename TPolicy::AccT;
  using QueueT = typename TPolicy::QueueT;

  using TransformerT = Transformer;
  using ReducerT = Reducer;
  using ReduceT = typename Reducer::reduce_t;

  constexpr std::size_t nSrc = TPolicy::nSrc;

  if (out.size() != nSrc || in.size() != nSrc) {
    throw std::invalid_argument("launchTransformReduce(): x and y must contain nSrc buffers");
  }

  auto const devAcc = alpaka::getDev(policy.queue);
  Idx const max_threads_per_block = static_cast<Idx>(alpaka::getAccDevProps<AccT>(devAcc).m_blockThreadCountMax);

  std::uint32_t const wExtend = alpaka::getPreferredWarpSize(alpaka::getDev(policy.queue));
  Idx const max_block_x_dim = ((max_threads_per_block / static_cast<Idx>(nSrc)) / wExtend) * wExtend;

  auto reduce_bufs = cms::alpakatools::reduce::create_reduction_resources<AccT, QueueT, ReduceT>(policy.queue, nSrc);

  auto msrc_tr = cms::alpakatools::multiblas::instantiateTransformReducer<nSrc, TXBufAcc, TYBufAcc, ReduceT>(
      reduce_bufs, in, out, init, reducer, transformer);

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

  alpaka::exec<AccT>(policy.queue, workDiv, msrc_tr, begin, end);

  if constexpr (TPolicy::copy_to_host) {
    return msrc_tr.template host_reduced_values<QueueT>(policy.queue);
  } else {
    return msrc_tr.template device_reduced_values<QueueT>(policy.queue);
  }
}

#endif
