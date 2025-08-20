#ifndef DataFormats_EgammaReco_interface_alpaka_EleSeedDeviceCollection_h
#define DataFormats_EgammaReco_interface_alpaka_EleSeedDeviceCollection_h

#include <Eigen/Core>
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/PortableTestObjects/interface/TestSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "DataFormats/EgammaReco/interface/EleSeedSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace reco {
    using namespace ::reco;
    using EleSeedDeviceCollection = PortableCollection<EleSeedSoA>;
  }  // namespace portabletest
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DataFormats_PortableTestObjects_interface_alpaka_TestDeviceCollection_h
