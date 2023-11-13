#ifndef RecoVertex_PrimaryVertexProducer_Alpaka_plugins_alpaka_FitterAlgo_h
#define RecoVertex_PrimaryVertexProducer_Alpaka_plugins_alpaka_FitterAlgo_h

#include "DataFormats/PortableVertex/interface/alpaka/VertexDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class FitterAlgo {
  public:
    void fill(Queue& queue, portablevertex::TrackDeviceCollection& trackCollection, portablevertex::VertexDeviceCollection& vertexCollection, portablevertex::BeamSpotDeviceCollection& beamSpot) const;
    //void configure(edm::ParameterSet const& config);
  private:
    double Tmin;
    double Tpurge;
    double Tstop;
    double vertexSize_;
    double coolingFactor_;
    double d0CutOff_;
    double dzCutOff_;
    double uniquetrkweight_;
    double uniquetrkminp_;
    double zmerge_;
    double sel_zrange_;
    double convergence_mode_;
    double delta_lowT_;
    double delta_highT_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoVertex_PrimaryVertexProducer_Alpaka_plugins_alpaka_FitterAlgo_h
