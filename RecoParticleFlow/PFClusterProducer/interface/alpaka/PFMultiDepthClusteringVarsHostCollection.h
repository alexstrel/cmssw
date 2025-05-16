#ifndef RecoParticleFlow_PFRecHitProducer_interface_alpaka_PFMultiDepthClusteringVarsHostCollection_h 
#define RecoParticleFlow_PFRecHitProducer_interface_alpaka_PFMultiDepthClusteringVarsHostCollection_h

#include "RecoParticleFlow/PFClusterProducer/interface/PFMultiDepthClusteringVarsSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace reco {
  using PFMultiDepthClusteringVarsHostCollection = PortableHostCollection<::reco::PFMultiDepthClusteringVarsSoA>;
}  // namespace reco

#endif
