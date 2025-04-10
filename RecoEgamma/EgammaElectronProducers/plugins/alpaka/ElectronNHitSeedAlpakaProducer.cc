#include <Eigen/Core>

#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/EleSeedSoA.h"
#include "DataFormats/EgammaReco/interface/SuperClusterSoA.h"
#include "DataFormats/EgammaReco/interface/alpaka/SuperclusterDeviceCollection.h"
#include "DataFormats/EgammaReco/interface/SuperclusterHostCollection.h"
#include "DataFormats/EgammaReco/interface/alpaka/EleSeedDeviceCollection.h"
#include "DataFormats/EgammaReco/interface/EleSeedHostCollection.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/ParametrizedEngine/interface/ParabolicParametrizedMagneticField.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaServices/interface/alpaka/AlpakaService.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"

// Additional includes for testing / comparing implementations
#include "PixelMatchingAlgo.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronUtilities.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/PerpendicularBoundPlaneBuilder.h"
#include "TrackingTools/TrajectoryState/interface/ftsFromVertexToPoint.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/alpaka/ftsFromVertexToPointPortable.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/alpaka/helixBarrelPlaneCrossingByCircle.h"
#include "DataFormats/EgammaReco/interface/EleRelPointPairPortable.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/alpaka/helixArbitraryPlaneCrossing.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/alpaka/helixArbitraryPlaneCrossing2Order.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/alpaka/helixForwardPlaneCrossing.h"

#include "DataFormats/EgammaReco/interface/Plane.h"

using Vector3d = Eigen::Matrix<double, 3, 1>;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

	class ElectronNHitSeedAlpakaProducer : public global::EDProducer<> {
	public:
		ElectronNHitSeedAlpakaProducer(const edm::ParameterSet& pset): 
			deviceToken_{produces()},
			size_{pset.getParameter<int32_t>("size")},
			initialSeedsToken_(consumes(pset.getParameter<edm::InputTag>("initialSeeds"))),
			beamSpotToken_(consumes(pset.getParameter<edm::InputTag>("beamSpot"))),
			magFieldToken_(esConsumes()),
			geomToken(esConsumes())
			{
				superClustersTokens_ = consumes(pset.getParameter<edm::InputTag>("superClusters"));
			}

		void produce(edm::StreamID sid, device::Event& event, device::EventSetup const& iSetup) const override {

			auto vprim_ = event.get(beamSpotToken_).position();
			GlobalPoint vprim(vprim_.x(), vprim_.y(), vprim_.z());
			Vector3d vertex{vprim.x(),vprim.y(),vprim.z()};

			// NEW EVENT 

			std::cout<<" -----> NEW EVENT with vprim : "<<vprim_<<std::endl;

			// Get MagField ESProduct for comparing & Geom ESProduct 
			auto const& magField = iSetup.getData(magFieldToken_);
			const TrackerGeometry* theG = &iSetup.getData(geomToken);

			PropagatorWithMaterial backwardPropagator_ = PropagatorWithMaterial(oppositeToMomentum, 0.000511, &magField);

			std::vector<reco::SuperClusterRef> superClusterRefVec;

			for (auto& superClusRef : event.get(superClustersTokens_)) 
				superClusterRefVec.push_back(superClusRef);  
	
			int32_t superClusterCollectionSize = superClusterRefVec.size();
			reco::SuperclusterHostCollection hostProductSCs{superClusterCollectionSize, event.queue()};

			std::vector<TrajectorySeed> seedRefVec;
			for (auto& initialSeedRef : event.get(initialSeedsToken_)) 
				seedRefVec.push_back(initialSeedRef);  

			int32_t seedCollectionSize = seedRefVec.size();
			reco::EleSeedHostCollection hostProductSeeds{seedCollectionSize, event.queue()};

			auto& viewSCs = hostProductSCs.view();
			auto& viewSeeds = hostProductSeeds.view();

			std::cout<<" -----> Collection sizes SCs: "<<superClusterCollectionSize << " Seeds " <<seedCollectionSize<<std::endl;
			////////////////////////////////////////////////////////////////////
			// Fill in SOAs
			// Technically should separate in different producers that create the SoAs ?
			// Info on SoAs : https://github.com/cms-sw/cmssw/blob/master/DataFormats/SoATemplate/README.md
			////////////////////////////////////////////////////////////////////

			int32_t i = 0;
	        for (auto& superClusRef : superClusterRefVec)
			{
				viewSCs[i].id() =  i;
				viewSCs[i].scSeedTheta() =  superClusRef->seed()->position().theta();
				viewSCs[i].scPhi() = superClusRef->position().phi();
				viewSCs[i].scR() = superClusRef->position().r();
				viewSCs[i].scEnergy() = superClusRef->energy();
				++i;
			}


			// To figure out is there is another way to bulid this SoA
			i = 0;
			for (auto& initialSeedRef : seedRefVec) 
			{	
				viewSeeds[i].nHits() = initialSeedRef.nHits();
				viewSeeds[i].id() = i;		
				viewSeeds[i].isMatched() = 0;		
				viewSeeds[i].matchedScID() = -1;
			
				auto hitIt = initialSeedRef.recHits().begin();
			
				// Hit 0
				const auto& recHit0 = *hitIt;
				const auto& pos0 = recHit0.globalPosition();
				const auto& surf0 = recHit0.det()->surface().position();
				const auto& rot0 = recHit0.det()->surface().rotation().z();
				viewSeeds[i].hit0detectorID()  = (recHit0.geographicalId().subdetId() == PixelSubdetector::PixelBarrel) ? 1 : 0;
				viewSeeds[i].hit0isValid() = recHit0.isValid();
				viewSeeds[i].hit0Pos() = Eigen::Vector3d(pos0.x(), pos0.y(), pos0.z());
				viewSeeds[i].surf0Pos() = Eigen::Vector3d(surf0.x(), surf0.y(), surf0.z());
				viewSeeds[i].surf0Rot() = Eigen::Vector3d(rot0.x(), rot0.y(), rot0.z());
			
				// Hit 1
				++hitIt;
				const auto& recHit1 = *hitIt;
				const auto& pos1 = recHit1.globalPosition();
				const auto& surf1 = recHit1.det()->surface().position();
				const auto& rot1 = recHit1.det()->surface().rotation().z();
				viewSeeds[i].hit1detectorID()  = (recHit1.geographicalId().subdetId() == PixelSubdetector::PixelBarrel) ? 1 : 0;
				viewSeeds[i].hit1isValid() = recHit1.isValid();
				viewSeeds[i].hit1Pos() = Eigen::Vector3d(pos1.x(), pos1.y(), pos1.z());
				viewSeeds[i].surf1Pos() = Eigen::Vector3d(surf1.x(), surf1.y(), surf1.z());
				viewSeeds[i].surf1Rot() = Eigen::Vector3d(rot1.x(), rot1.y(), rot1.z());
			
				// Hit 2
				if (initialSeedRef.nHits() > 2) {
					++hitIt;
					const auto& recHit2 = *hitIt;
					const auto& pos2 = recHit2.globalPosition();
					const auto& surf2 = recHit2.det()->surface().position();
					const auto& rot2 = recHit2.det()->surface().rotation().z();
					viewSeeds[i].hit2detectorID()  = (recHit2.geographicalId().subdetId() == PixelSubdetector::PixelBarrel) ? 1 : 0;
					viewSeeds[i].hit2isValid() = recHit2.isValid();
					viewSeeds[i].hit2Pos() = Eigen::Vector3d(pos2.x(), pos2.y(), pos2.z());
					viewSeeds[i].surf2Pos() = Eigen::Vector3d(surf2.x(), surf2.y(), surf2.z());
					viewSeeds[i].surf2Rot() = Eigen::Vector3d(rot2.x(), rot2.y(), rot2.z());
				} else {
					// Zero initialization
					viewSeeds[i].hit2Pos().setZero();
					viewSeeds[i].surf2Pos().setZero();
					viewSeeds[i].surf2Rot().setZero();
					viewSeeds[i].hit2detectorID()  = 0;
					viewSeeds[i].hit2isValid() = 0;
				}
			
				++i;
			}

			// Create device products & copy to device
			reco::SuperclusterDeviceCollection deviceProductSCs{superClusterCollectionSize, event.queue()};
			alpaka::memcpy(event.queue(), deviceProductSCs.buffer(), hostProductSCs.buffer());
			reco::EleSeedDeviceCollection deviceProductSeeds{seedCollectionSize, event.queue()};
			alpaka::memcpy(event.queue(), deviceProductSeeds.buffer(), hostProductSeeds.buffer());

			// Print the SoA 
			// algo_.printSCs(event.queue(), deviceProductSCs);
			// algo_.printEleSeeds(event.queue(), deviceProductSeeds);
			// alpaka::wait(event.queue()); 
			// Matching algorithm
			algo_.matchSeeds(event.queue(), deviceProductSeeds, deviceProductSCs,vertex(0),vertex(1), vertex(2));
			alpaka::wait(event.queue()); 
			alpaka::memcpy(event.queue(), hostProductSeeds.buffer(), deviceProductSeeds.buffer());
			alpaka::wait(event.queue()); 

			auto& view = hostProductSeeds.view();
			std::cout<<"view.metadata().size() "<<view.metadata().size()<<std::endl;
			
			for (int i = 0; i < view.metadata().size(); ++i) {
				if(view[i].isMatched()>0){
					std::cout << "  Seed " << i << ":" << std::endl;
					std::cout << "  nHits: " << view[i].nHits() << std::endl;
					std::cout << "  isMatched: " << view[i].isMatched() << std::endl;
					std::cout << "  matchedScID: " << view[i].matchedScID() << std::endl;
				}
			}
			event.emplace(deviceToken_, std::move(deviceProductSeeds));
		}

		static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
			edm::ParameterSetDescription desc;
			desc.add<int32_t>("size");
			desc.add<edm::InputTag>("initialSeeds", {"hltElePixelSeedsCombined"});
			desc.add<edm::InputTag>("beamSpot", {"hltOnlineBeamSpot"});
  			desc.add<edm::InputTag>("superClusters", {"hltEgammaSuperClustersToPixelMatch"});
			descriptions.addWithDefaultLabel(desc);
		}

	private:
		const device::EDPutToken<reco::EleSeedDeviceCollection> deviceToken_;
		const int32_t size_;
		const edm::EDGetTokenT<TrajectorySeedCollection> initialSeedsToken_;
		const edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
		edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;
		const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken;
		edm::EDGetTokenT<std::vector<reco::SuperClusterRef>> superClustersTokens_;
		PixelMatchingAlgo const algo_{};
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(ElectronNHitSeedAlpakaProducer);

