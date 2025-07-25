#ifndef RecoEgamma_EgammaElectronProducers_plugins_alpaka_PixelMatchingAlgo_h
#define RecoEgamma_EgammaElectronProducers_plugins_alpaka_PixelMatchingAlgo_h


#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "DataFormats/EgammaReco/interface/alpaka/EleSeedDeviceCollection.h"
#include "DataFormats/EgammaReco/interface/alpaka/SuperclusterDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include <HeterogeneousCore/AlpakaInterface/interface/VecArray.h>

using Vector3d = cms::alpakatools::VecArray<double, 3>;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PixelMatchingAlgo {
  public:
    void printEleSeeds(Queue& queue, reco::EleSeedDeviceCollection& collection) const;
    void printSCs(Queue& queue, reco::SuperclusterDeviceCollection& collection) const;
    void matchSeeds(Queue& queue, reco::EleSeedDeviceCollection& collection, reco::SuperclusterDeviceCollection& collectionSCs, double vtx_X, double vtx_Y, double vtx_Z) const;

  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
