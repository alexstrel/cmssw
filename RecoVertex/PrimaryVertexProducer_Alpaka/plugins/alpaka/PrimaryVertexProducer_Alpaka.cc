#include "DataFormats/PortableVertex/interface/alpaka/VertexDeviceCollection.h"
#include "DataFormats/PortableVertex/interface/VertexHostCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"

#include "ClusterizerAlgo.h"
#include "FitterAlgo.h"



namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * This class does vertexing by
   * - consuming set of reco::Tracks
   * - converting them to a Alpaka-friendly dataformat
   * - clusterizing them into track clusters
   * - fitting cluster properties to vertex coordinates
   * - produces a device vertex EDProduct (that gets transferred to host automatically if needed)
   */
  class PrimaryVertexProducer_Alpaka : public stream::EDProducer<> {
  public:
    PrimaryVertexProducer_Alpaka(edm::ParameterSet const& config)
        : tsize_{config.getParameter<edm::ParameterSet>("trackSize").getParameter<int32_t>(
              EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE))}, 
	  vsize_{config.getParameter<edm::ParameterSet>("vertexSize").getParameter<int32_t>(
              EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE))} {
      // clusterizer_.configure(config);
      trackToken_ = consumes<reco::TrackCollection>(config.getParameter<edm::InputTag>("TrackLabel"));
      beamSpotToken_ = consumes<reco::BeamSpot>(config.getParameter<edm::InputTag>("BeamSpotLabel"));
      devicePutToken_ = produces(config.getParameter<std::string>("productInstanceName"));
    }

    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override {
      auto tracks   = iEvent.getHandle(trackToken_).product();
      auto beamSpot = iEvent.getHandle(beamSpotToken_).product(); 

      // Host collections
      auto hostTracks   = std::make_unique<portablevertex::TrackHostCollection>(tsize_, iEvent.queue());
      auto hostBeamSpot = std::make_unique<portablevertex::BeamSpotHostCollection>(iEvent.queue());
      auto hostVertex   = std::make_unique<portablevertex::VertexHostCollection>(vsize_, iEvent.queue());

      // Fill Host tracks
      unsigned int nTrueTracks = 0; 
      for (unsigned int idx = 0; idx < tracks->size() ; idx ++){
        double significance     = tracks->at(idx).stateAtBeamLine().transverseImpactParameter().significance();
        double dxy2		= tracks->at(idx).stateAtBeamLine().transverseImpactParameter().error()*tracks->at(idx).stateAtBeamLine().transverseImpactParameter().error();
        double dz2		= std::pow(tracks->at(idx).track().dzError(),2);
        double pAtIP		= tracks->at(idx).impactPointState().globalMomentum().transverse();
        double pxAtPCA		= tracks->at(idx).stateAtBeamLine().trackStateAtPCA().momentum().x();
        double pyAtPCA		= tracks->at(idx).stateAtBeamLine().trackStateAtPCA().momentum().y();
        double pzAtPCA		= tracks->at(idx).stateAtBeamLine().trackStateAtPCA().momentum().z();
        double pt2AtPCA         = tracks->at(idx).stateAtBeamLine().trackStateAtPCA().momentum().perp2();
        double bx		= tracks->at(idx).stateAtBeamLine().beamSpot().BeamWidthX();
        double by		= tracks->at(idx).stateAtBeamLine().beamSpot().BeamWidthY();
        double x		= tracks->at(idx).stateAtBeamLine().trackStateAtPCA().position().x();
        double y                = tracks->at(idx).stateAtBeamLine().trackStateAtPCA().position().y();
        double z                = tracks->at(idx).stateAtBeamLine().trackStateAtPCA().position().z();
        double etaAtIP		= std::fabs(tracks->at(idx).impactPointState().globalMomentum().eta());
        double chi2		= tracks->at(idx).normalizedChi2();
        int8_t nPixelHits	= tracks->at(idx).hitPattern().pixelLayersWithMeasurement();
        int8_t nTrackerHits     = tracks->at(idx).hitPattern().trackerLayersWithMeasurement();

        bool isGood = false;
        double weight = 0;
        //if ((significance < fParams.maxSignificance) && (dxy2 < fParams.maxdxyError*fParams.maxdxyError) && (dz2 < fParams.maxdzError*fParams.maxdzError) && (pAtIP > fParams.minpAtIP) && (std::fabs(etaAtIP) < fParams.maxetaAtIP) && (chi2 < fParams.maxchi2) && (nPixelHits >= fParams.minpixelHits) && (nTrackerHits >= fParams.mintrackerHits)){
        isGood = true;
        //}
        // And now the stuff for the clusterizer
        if (isGood){
            weight = 1.;  
            if (std::fabs(z) > 1000.){ 
                isGood = false;
                weight = 0;
                continue;
            }
            else{ // Get dz2 for the track
               // dz2 is zerror^2 + (bx*px + by*py)^2*pz^2/(pt^4) + vertex_size^2
               double oneoverdz2 = dz2 
	       + (std::pow(bx*pxAtPCA,2)+ std::pow(by*pyAtPCA,2))* std::pow(pzAtPCA,2)/(std::pow(pt2AtPCA,2)) 
	       //+ std::pow(fParams.vertexSize,2); // TODO:: For sure ways to optimize this
               oneoverdz2 = 1./oneoverdz2;
               if (not(std::isfinite(oneoverdz2)) || oneoverdz2< std::numeric_limits<double>::min()){ // Bad track dz2 is taken out
                   isGood = false;
                   weight = 0;
                   continue;
               }
               else{
                   if (fParams.d0CutOff > 0){ // Track weights are activated only if there is a non-zero cutoff
                       // weight is 1/(1 + e^{sig^2 - d0cutoff^2})
                       weight = 1. ; /// (1+exp(significance*significance - fParams.d0CutOff*fParams.d0CutOff ));
                       if (not(std::isfinite(weight)) || weight< std::numeric_limits<double>::epsilon()){ // Bad track weight is taken out
                           isGood = false;
                           weight = 0;
                           continue;
                       }
                   }
                   // If we are here, the track is to be passed to the clusterizer. So initialize the clusterizer stuff
                   // really save track now!
                   hostTracks->view().totweight() += weight;
                   hostTracks->view()[nTrueTracks].x() = x;
                   hostTracks->view()[nTrueTracks].y() = y;
                   hostTracks->view()[nTrueTracks].z() = z;
                   hostTracks->view()[nTrueTracks].xAtIP() = tracks->at(idx).impactPointState().globalPosition().x();
                   hostTracks->view()[nTrueTracks].yAtIP() = tracks->at(idx).impactPointState().globalPosition().y();
                   hostTracks->view()[nTrueTracks].px() = pxAtPCA;
                   hostTracks->view()[nTrueTracks].py() = pyAtPCA;
                   hostTracks->view()[nTrueTracks].pz() = pzAtPCA;
                   hostTracks->view()[nTrueTracks].weight() = weight;
                   hostTracks->view()[nTrueTracks].tt_index() = idx;
                   hostTracks->view()[nTrueTracks].dz2() = dz2;
                   hostTracks->view()[nTrueTracks].oneoverdz2() = oneoverdz2;
                   hostTracks->view()[nTrueTracks].dxy2() = dxy2;
                   hostTracks->view()[nTrueTracks].dxy2AtIP() = tracks->at(idx).track().dxyError()*tracks->at(idx).track().dxyError();
                   hostTracks->view()[nTrueTracks].order() = nTrueTracks;
                   hostTracks->view()[nTrueTracks].sum_Z() = 0;
                   hostTracks->view()[nTrueTracks].kmin() = 0; // will loop from kmin to kmax-1. At the start only one vertex
                   hostTracks->view()[nTrueTracks].kmax() = 1;
                   hostTracks->view()[nTrueTracks].aux1() = 0;
                   hostTracks->view()[nTrueTracks].aux2() = 0;
                   hostTracks->view()[nTrueTracks].isGood() = true;
                   hostTracks->view().nT() += weight;
               }
            }    
        }
      } 

      // Initialize BeamSpot
      hostBeamSpot->view().x() = beamSpot->position().x();
      hostBeamSpot->view().y() = beamSpot->position().y();
      hostBeamSpot->view().sx() = beamSpot->rotatedCovariance3D()(0,0);
      hostBeamSpot->view().sy() = beamSpot->rotatedCovariance3D().(1,1);

      // Now the device SoAs
      auto deviceTracks   = std::make_unique<portablevertex::TrackDeviceCollection>(tsize_, iEvent.queue());
      auto deviceVertex   = std::make_unique<portablevertex::VertexDeviceCollection>(vsize_, iEvent.queue());
      auto deviceBeamSpot = std::make_unique<portablevertex::BeamSpotDeviceCollection>(1, iEvent.queue());

      // Copy input to device, no need to copy vertex
      alpaka::memcpy(iEvent.queue(), deviceTracks->buffer(), hostTracks->buffer());
      alpaka::memcpy(iEvent.queue(), deviceBeamSpot->buffer(), hostBeamSpot->buffer());

      // run the algorithm, potentially asynchronously
      //// First clusterize
      clusterizer_.fill(iEvent.queue(), *deviceTracks, *deviceVertex);
      //// And then fit
      fitter_.fill(iEvent.queue(), *deviceTracks, *deviceVertex, *deviceBeamSpot);

      // Copy output back vertex to host
      alpaka::memcpy(iEvent.queue(), hostVertex->buffer(), deviceVertex->buffer());

      // Last, do the conversion back to reco::Vertex
      auto result = std::make_unique<reco::VertexCollection>();
      reco::VertexCollection& vColl = (*result);
      for (unsigned int iV = 0; iV < hostVertex->view().nV() ; iV++){
          if (not(hostVertex->view()[iV].isGood())) continue;
          AlgebraicSymMatrix33 err;
          err[0][0] = hostVertex->view()[iV].errx();
          err[1][1] = hostVertex->view()[iV].erry();
          err[2][2] = hostVertex->view()[iV].errz();
          reco::Vertex newV(math::GlobalPoint(hostVertex->view()[iV].x(), hostVertex->view()[iV].y(), hostVertex->view()[iV].z()), err, hostVertex->view()[iV].chi2(), hostVertex->view()[iV].ndof(), hostVertex->view()[iV].ntracks());
          for (unsigned int iT=0; iT <  hostVertex->view()[iV].ntracks(); iT++) {
             unsigned int new_itrack = hostVertex->view()[iV].track_id()[iT];
             newV.add(tracks->at(new_itrack), hostVertex->view()[iV].track_weight()[iT]);
          }
          vColl.push_back(newV);
      }
      // This would put it as a portable object!  
      iEvent.put(devicePutToken_, std::move(deviceVertex));
      // And finally put it in the event! It doesn't work :(
      //iEvent.put(std::move(vColl));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("TrackLabel");
      desc.add<edm::InputTag>("BeamSpotLabel");

      desc.add<std::string>("productInstanceName", "");

      edm::ParameterSetDescription psetSize;
      psetSize.add<int32_t>("alpaka_serial_sync");
      psetSize.add<int32_t>("alpaka_cuda_async");
      desc.add("tsize", psetSize);
      desc.add("vsize", psetSize);

      descriptions.addWithDefaultLabel(desc);
    }

  private:
    edm::EDGetTokenT<reco::TrackCollection> trackToken_;
    edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
    device::EDPutToken<portablevertex::VertexDeviceCollection> devicePutToken_;

    // This is the maximum vertex size in the input/output
    const int32_t tsize_;
    const int32_t vsize_;
    // implementation of the algorithm
    ClusterizerAlgo clusterizer_;
    FitterAlgo fitter_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(PrimaryVertexProducer_Alpaka);
