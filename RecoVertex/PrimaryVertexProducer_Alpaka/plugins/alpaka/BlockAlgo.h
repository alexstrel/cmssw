#ifndef RecoVertex_PrimaryVertexProducer_Alpaka_plugins_alpaka_BlockAlgo_h
#define RecoVertex_PrimaryVertexProducer_Alpaka_plugins_alpaka_BlockAlgo_h

#include "DataFormats/PortableVertex/interface/alpaka/VertexDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class BlockAlgo {
  public:
    BlockAlgo(Queue& queue, const uint32_t nT, int32_t blockSize, double blockOverlap); // Just configuration and making job divisions
    void createBlocks(Queue& queue, const portablevertex::TrackDeviceCollection& inputTrack, portablevertex::TrackDeviceCollection& trackInBlocks); // The actual block creation

  private:
    int32_t blockSize;
    double blockOverlap;
    WorkDiv<Dim1D> workDiv;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoVertex_PrimaryVertexProducer_Alpaka_plugins_alpaka_BlockAlgo_h
