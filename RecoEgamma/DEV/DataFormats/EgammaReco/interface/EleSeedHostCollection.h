#ifndef DataFormats_EgammaReco_interface_EleSeedHostCollection_h
#define DataFormats_EgammaReco_interface_EleSeedHostCollection_h

#include <Eigen/Core>
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/TestSoA.h"
#include "DataFormats/EgammaReco/interface/EleSeedSoA.h"

namespace reco {
  using EleSeedHostCollection = PortableHostCollection<EleSeedSoA>;
}  // namespace reco

#endif  // DataFormats_PortableTestObjects_interface_TestHostCollection_h
