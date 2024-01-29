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
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"



namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * This class does
   * - consume set of reco::Tracks
   * - converting them to a Alpaka-friendly dataformat
   * - put the Alpaka dataformat in the device for later consumption
   */
  struct filterParameters {
      double maxSignificance;
      double maxdxyError;
      double maxdzError;
      double minpAtIP;
      double maxetaAtIP;
      double maxchi2;
      int minpixelHits;
      int mintrackerHits;
      double vertexSize;
      double d0CutOff;
  };

  class PortableTrackSoAProducer : public global::EDProducer<> {
  public:
    PortableTrackSoAProducer(edm::ParameterSet const& config) {
      theConfig       = config;
      trackToken_     = consumes<reco::TrackCollection>(config.getParameter<edm::InputTag>("TrackLabel"));
      beamSpotToken_  = consumes<reco::BeamSpot>(config.getParameter<edm::InputTag>("BeamSpotLabel"));
      devicePutToken_ = produces();
      fParams = {
       .maxSignificance=config.getParameter<edm::ParameterSet>("TkFilterParameters").getParameter<double>("maxD0Significance"),
       .maxdxyError    =config.getParameter<edm::ParameterSet>("TkFilterParameters").getParameter<double>("maxD0Error"),
       .maxdzError     =config.getParameter<edm::ParameterSet>("TkFilterParameters").getParameter<double>("maxDzError"),
       .minpAtIP       =config.getParameter<edm::ParameterSet>("TkFilterParameters").getParameter<double>("minPt"),
       .maxetaAtIP     =config.getParameter<edm::ParameterSet>("TkFilterParameters").getParameter<double>("maxEta"),
       .maxchi2        =config.getParameter<edm::ParameterSet>("TkFilterParameters").getParameter<double>("maxNormalizedChi2"),
       .minpixelHits   =config.getParameter<edm::ParameterSet>("TkFilterParameters").getParameter<int>("minPixelLayersWithHits"),
       .mintrackerHits =config.getParameter<edm::ParameterSet>("TkFilterParameters").getParameter<int>("minSiliconLayersWithHits"),
       .vertexSize     =config.getParameter<edm::ParameterSet>("TkFilterParameters").getParameter<double>("vertexSize"),
       .d0CutOff       =config.getParameter<edm::ParameterSet>("TkFilterParameters").getParameter<double>("d0CutOff")
      };      
    }

    void produce(edm::StreamID sid, device::Event& iEvent, device::EventSetup const& iSetup) const override {
      // Get input collections from event
      auto tracks = iEvent.getHandle(trackToken_).product();
      auto beamSpot    = iEvent.getHandle(beamSpotToken_).product();
      int32_t tsize_   = tracks->size();

      // Host collections
      portablevertex::TrackHostCollection hostTracks{tsize_, iEvent.queue()};
      auto& tview = hostTracks.view();

      // Fill Host collections with input
      tview.totweight() = 0;
      tview.nT() = 0;
      int32_t nTrueTracks = 0; // This will keep track of how many we actually copy to device, only those that pass filter
      for (int32_t idx = 0; idx < tsize_ ; idx ++){
        double weight = convertTrack(tview[nTrueTracks], tracks->at(idx), *beamSpot, fParams, idx, nTrueTracks);
        if (weight > 0){
          nTrueTracks       += 1;
	  tview.nT()        += 1;
	  tview.totweight() += weight;
	}
      }

      // Create device collections and copy into device
      portablevertex::TrackDeviceCollection deviceTracks{tsize_, iEvent.queue()};
      
      alpaka::memcpy(iEvent.queue(), deviceTracks.buffer(), hostTracks.buffer());

      // And put into the event
      iEvent.emplace(devicePutToken_, std::move(deviceTracks));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("TrackLabel");
      desc.add<edm::InputTag>("BeamSpotLabel");
      edm::ParameterSetDescription psd0;
      psd0.add<double>("maxNormalizedChi2", 10.0);
      psd0.add<double>("minPt", 0.0);
      psd0.add<std::string>("algorithm", "filter");
      psd0.add<double>("maxEta", 2.4);
      psd0.add<double>("maxD0Significance", 4.0);
      psd0.add<double>("maxD0Error", 1.0);
      psd0.add<double>("maxDzError", 1.0);
      psd0.add<std::string>("trackQuality", "any");
      psd0.add<int>("minPixelLayersWithHits", 2);
      psd0.add<int>("minSiliconLayersWithHits", 5);      
      psd0.add<double>("vertexSize", 0.006);
      psd0.add<double>("d0CutOff", 0.10);
      desc.add<edm::ParameterSetDescription>("TkFilterParameters",psd0);
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    edm::EDGetTokenT<reco::TrackCollection> trackToken_;
    edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
    device::EDPutToken<portablevertex::TrackDeviceCollection> devicePutToken_;
    edm::ParameterSet theConfig;
    static double convertTrack(portablevertex::TrackHostCollection::View::element out, const reco::Track in, const reco::BeamSpot bs, filterParameters fParams, int32_t idx, int32_t order);
    static void convertBeamSpot(portablevertex::BeamSpotHostCollection::View::element out, const reco::BeamSpot in);
    filterParameters fParams;
  }; //PortableTrackSoAProducer declaration

  double PortableTrackSoAProducer::convertTrack(portablevertex::TrackHostCollection::View::element out, const reco::Track in, const reco::BeamSpot bs, filterParameters fParams, int32_t idx, int32_t order){
    bool isGood = false;
    double weight = -1;
    if( in.dxyError(bs) > 0){
      if ((in.dxy(bs)/in.dxyError(bs) < fParams.maxSignificance) && (in.dxyError(bs) < fParams.maxdxyError) && (in.dzError() < fParams.maxdzError) && (in.pt() > fParams.minpAtIP) && (std::fabs(in.eta()) < fParams.maxetaAtIP) && (in.chi2()<fParams.maxchi2) && (in.hitPattern().pixelLayersWithMeasurement()>fParams.minpixelHits) && (in.hitPattern().trackerLayersWithMeasurement() > fParams.mintrackerHits) && (in.vz() < 1000.)) isGood = true;
    }
    if (isGood){ // Then define vertex-related stuff like weights
      if (fParams.d0CutOff > 0){
        // significance in transverse plane
	double significance = in.dxy(bs)/in.dxyError(bs);
        // weight is based on transversal displacement of the track	
        weight = 1 + exp(significance*significance + fParams.d0CutOff * fParams.d0CutOff);
      }
      out.x() = in.vx();
      out.y() = in.vy();
      out.z() = in.vz();
      out.px() = in.px();
      out.py() = in.py();
      out.pz() = in.pz();
      out.weight() = weight;
      // The original index in the reco::Track collection so we can go back to it eventually
      out.tt_index() = idx;
      out.dz2() = in.dzError()*in.dzError();
      // Modified dz2 to account correlations and vertex size for clusterizer 
      // dz^2 + (bs*pt)^2*pz^2/pt^2 + vertexSize^2
      double oneoverdz2 = (in.dzError()*in.dzError()) + ((bs.BeamWidthX()*bs.BeamWidthX()*in.px()*in.px()) + (bs.BeamWidthY()*bs.BeamWidthY()*in.py()*in.py()))*in.pz()*in.pz()/(in.pt()*in.pt()) + fParams.vertexSize*fParams.vertexSize;
      oneoverdz2 = 1./oneoverdz2;
      out.oneoverdz2() = oneoverdz2;
      out.dxy2AtIP() = in.dxyError(bs)*in.dxyError(bs);
      out.dxy2()     = in.dxyError()*in.dxyError();

      out.order() = order;
      // All of these are initializers for the vertexing 
      out.sum_Z() = 0; // partition function sum
      out.kmin() = 0; // minimum vertex identifier, will loop from kmin to kmax-1. At the start only one vertex
      out.kmax() = 1; // maximum vertex identifier, will loop from kmin to kmax-1. At the start only one vertex
      out.aux1() = 0; // for storing various things in between kernels
      out.aux2() = 0; // for storing various things in between kernels
      out.isGood() = true; // if we are here, we are to keep this track
    }
    return weight;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(PortableTrackSoAProducer);
