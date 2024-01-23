#include "DataFormats/PortableVertex/interface/alpaka/VertexDeviceCollection.h"
#include "DataFormats/PortableVertex/interface/VertexHostCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"

#include "BlockAlgo.h"
#include "ClusterizerAlgo.h"
#include "FitterAlgo.h"



namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * This class does vertexing by
   * - consuming set of portablevertex::Track
   * - clusterizing them into track clusters
   * - fitting cluster properties to vertex coordinates
   * - produces a device vertex product (portablevertex::Vertex)
   */
  class PrimaryVertexProducer_Alpaka : public global::EDProducer<> {
  public:
    PrimaryVertexProducer_Alpaka(edm::ParameterSet const& config){
      trackToken_     = consumes(config.getParameter<edm::InputTag>("TrackLabel"));
      beamSpotToken_  = consumes(config.getParameter<edm::InputTag>("BeamSpotLabel"));
      devicePutToken_ = produces();
      blockSize       = config.getParameter<int32_t>("blockSize"); 
      blockOverlap    = config.getParameter<double>("blockOverlap");
      fitterParams = {
        .chi2cutoff            = config.getParameter<edm::ParameterSet>("TkFitterParameters").getParameter<double>("chi2cutoff"),
        .minNdof               = config.getParameter<edm::ParameterSet>("TkFitterParameters").getParameter<double>("minNdof"), 
        .useBeamSpotContraint  = config.getParameter<edm::ParameterSet>("TkFitterParameters").getParameter<bool>("useBeamSpotContraint"),
        .maxDistanceToBeam     = config.getParameter<edm::ParameterSet>("TkFitterParameters").getParameter<double>("maxDistanceToBeam")
      };
      clusterParams = {
        .Tmin   = config.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<double>("Tmin"),
        .Tpurge = config.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<double>("Tpurge"),
        .Tstop  = config.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<double>("Tstop"),
        .vertexSize = config.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<double>("vertexSize"),
        .coolingFactor = config.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<double>("coolingFactor"),
        .d0CutOff = config.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<double>("d0CutOff"),
        .dzCutOff = config.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<double>("dzCutOff"),
        .uniquetrkweight = config.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<double>("uniquetrkweight"),
        .uniquetrkminp = config.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<double>("uniquetrkminp"),
        .zmerge = config.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<double>("zmerge"),
        .sel_zrange = config.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<double>("zrange"),
        .convergence_mode = config.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<int>("convergence_mode"),
        .delta_lowT = config.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<double>("delta_lowT"),
        .delta_highT = config.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<double>("delta_highT")
      };
    }

    void produce(device::Event& iEvent, device::EventSetup const& iSetup) {
      const portablevertex::TrackDeviceCollection& inputtracks   = iEvent.get(trackToken_);
      const portablevertex::BeamSpotDeviceCollection& beamSpot     = iEvent.get(beamSpotToken_);
      int32_t nT = inputtracks.view().nT();
      int32_t nBlocks = int32_t ((nT+blockSize)/(blockOverlap*blockSize)); // In a couple of edge cases we will end up with an empty block, but this cleans after itself in the clusterizer anyhow
     
      // Now the device collections we still need
      portablevertex::TrackDeviceCollection tracksInBlocks{nBlocks*blockSize, iEvent.queue()}; // As high as needed
      portablevertex::VertexDeviceCollection deviceVertex{512, iEvent.queue()}; // Hard capped to 512

      // run the algorithm, potentially asynchronously
      //// First create the individual blocks
      BlockAlgo blockKernel_{iEvent.queue(), inputtracks.view().nT(), blockSize, blockOverlap}; 
      blockKernel_.createBlocks(iEvent.queue(), inputtracks, tracksInBlocks);
      // Need to have the blocks created before launching the next step
      alpaka::wait(iEvent.queue());
      //// Then run the clusterizer per blocks
      ClusterizerAlgo clusterizerKernel_{iEvent.queue(), tracksInBlocks.view().nT(), blockSize, clusterParams}; 
      clusterizerKernel_.clusterize(iEvent.queue(), tracksInBlocks, deviceVertex);
      // Need to have all vertex before arbitrating and deciding what we keep
      alpaka::wait(iEvent.queue()); 
      clusterizerKernel_.arbitrate(iEvent.queue(), tracksInBlocks, deviceVertex);
      alpaka::wait(iEvent.queue());
      //// And then fit
      FitterAlgo fitterKernel_{iEvent.queue(), deviceVertex.view().nV(), fitterParams};
      fitterKernel_.fit(iEvent.queue(), tracksInBlocks, deviceVertex, beamSpot);
      // This puts it in the event as a portable object!  
      iEvent.emplace(devicePutToken_, std::move(deviceVertex));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("TrackLabel");
      desc.add<edm::InputTag>("BeamSpotLabel");
      desc.add<double>("blockOverlap");
      desc.add<int32_t>("blockSize");
      edm::ParameterSetDescription parf0;
      parf0.add<double>("chi2cutoff", 2.5);
      parf0.add<double>("minNdof", 0.0);
      parf0.add<bool>("useBeamSpotContraint", true);
      parf0.add<double>("maxDistanceToBeam", 1.0);
      desc.add<edm::ParameterSetDescription>("TkFitterParameters",parf0);
      edm::ParameterSetDescription parc0;
      parc0.add<double>("d0CutOff", 3.0);
      parc0.add<double>("Tmin", 2.0);
      parc0.add<double>("delta_lowT", 0.001);
      parc0.add<double>("zmerge", 0.01);
      parc0.add<double>("dzCutOff", 3.0);
      parc0.add<double>("Tpurge", 2.0);
      parc0.add<int32_t>("convergence_mode", 0);
      parc0.add<double>("delta_highT", 0.01);
      parc0.add<double>("Tstop", 0.5);
      parc0.add<double>("coolingFactor", 0.6);
      parc0.add<double>("vertexSize", 0.006);
      parc0.add<double>("uniquetrkweight", 0.8);
      parc0.add<double>("uniquetrkminp", 0.0);
      parc0.add<double>("zrange", 4.0);
      desc.add<edm::ParameterSetDescription>("TkClusParameters",parc0);
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    edm::EDGetTokenT<portablevertex::TrackDeviceCollection> trackToken_;
    edm::EDGetTokenT<portablevertex::BeamSpotDeviceCollection> beamSpotToken_;
    device::EDPutToken<portablevertex::VertexDeviceCollection> devicePutToken_;
    int32_t blockSize;
    double blockOverlap;
    fitterParameters fitterParams;
    clusterParameters clusterParams;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(PrimaryVertexProducer_Alpaka);
