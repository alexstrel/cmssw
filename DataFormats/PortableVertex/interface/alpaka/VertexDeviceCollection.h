#ifndef DataFormats_PortableVertex_interface_alpaka_VertexDeviceCollection_h
#define DataFormats_PortableVertex_interface_alpaka_VertexDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/PortableVertex/interface/VertexSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace portablevertex {

    // make the names from the top-level portablevertex namespace visible for unqualified lookup
    // inside the ALPAKA_ACCELERATOR_NAMESPACE::portablevertex namespace
    using namespace::portablevertex;

    using VertexDeviceCollection = PortableCollection<VertexSoA>;
    using TrackDeviceCollection = PortableCollection<TrackSoA>;
    using BeamSpotDeviceCollection = PortableCollection<BeamSpotSoA>;
  }  // namespace portablevertex

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DataFormats_PortableVertex_interface_alpaka_VertexDeviceCollection_h
