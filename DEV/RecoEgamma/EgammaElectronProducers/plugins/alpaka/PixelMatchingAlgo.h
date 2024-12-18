#ifndef RecoEgamma_EgammaElectronProducers_plugins_alpaka_PixelMatchingAlgo_h
#define RecoEgamma_EgammaElectronProducers_plugins_alpaka_PixelMatchingAlgo_h

#include <Eigen/Core>

#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "DataFormats/EgammaReco/interface/alpaka/EleSeedDeviceCollection.h"
#include "DataFormats/EgammaReco/interface/alpaka/SuperclusterDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

using Vector3f = Eigen::Matrix<double, 3, 1>;//3f -> 3d??

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PixelMatchingAlgo {
  public:
    void printEleSeeds(Queue& queue, reco::EleSeedDeviceCollection& collection) const;
    void printSCs(Queue& queue, reco::SuperclusterDeviceCollection& collection) const;
    void matchSeeds(Queue& queue, reco::EleSeedDeviceCollection& collection, reco::SuperclusterDeviceCollection& collectionSCs, double vtx_X, double vtx_Y, double vtx_Z) const;

  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
