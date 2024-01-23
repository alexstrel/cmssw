#ifndef RecoVertex_PrimaryVertexProducer_Alpaka_plugins_alpaka_FitterAlgo_h
#define RecoVertex_PrimaryVertexProducer_Alpaka_plugins_alpaka_FitterAlgo_h

#include "DataFormats/PortableVertex/interface/alpaka/VertexDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  struct fitterParameters {
    double chi2cutoff;
    double minNdof;
    bool useBeamSpotContraint;
    double maxDistanceToBeam;
  };

  class FitterAlgo {
  public:
    FitterAlgo(Queue& queue, const uint32_t nV, fitterParameters fPar); // Just configuration and making job divisions
    void fit(Queue& queue, const portablevertex::TrackDeviceCollection& deviceTrack, portablevertex::VertexDeviceCollection& deviceVertex, const portablevertex::BeamSpotDeviceCollection& deviceBeamSpot); // The actual fitting
  private:
    double chi2cutoff;
    double minNdof;
    bool useBeamSpotContraint;
    double maxDistanceToBeam;
    WorkDiv<Dim1D> workDiv;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoVertex_PrimaryVertexProducer_Alpaka_plugins_alpaka_FitterAlgo_h
