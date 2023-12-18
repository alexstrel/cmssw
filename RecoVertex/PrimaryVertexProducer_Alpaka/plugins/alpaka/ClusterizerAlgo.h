#ifndef RecoVertex_PrimaryVertexProducer_Alpaka_plugins_alpaka_ClusterizerAlgo_h
#define RecoVertex_PrimaryVertexProducer_Alpaka_plugins_alpaka_ClusterizerAlgo_h

#include "DataFormats/PortableVertex/interface/alpaka/VertexDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  struct clusterParameters {
      double Tmin;
      double Tpurge;
      double Tstop;
      double vertexSize;
      double coolingFactor;
      double d0CutOff;
      double dzCutOff;
      double uniquetrkweight;
      double uniquetrkminp;
      double zmerge;
      double sel_zrange;
      int32_t convergence_mode;
     ;double delta_lowT;
      double delta_highT;
  };

  class ClusterizerAlgo {
  public:
    ClusterizerAlgo(Queue& queue, const uint32_t nT, int32_t blockSize, clusterParameters cPar); // Just configuration and making job divisions
    void clusterize(Queue& queue, const portablevertex::TrackDeviceCollection& inputTracks, portablevertex::VertexDeviceCollection& deviceVertex); // Clusterization
    void arbitrate(Queue& queue, const portablevertex::TrackDeviceCollection& inputTracks, portablevertex::VertexDeviceCollection& deviceVertex); // Arbitration
  private:
    WorkDiv<Dim1D> workDivCluster;
    WorkDiv<Dim1D> workDivArbitrate;
    clusterParameters cParams;    
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoVertex_PrimaryVertexProducer_Alpaka_plugins_alpaka_ClusterizerAlgo_h
