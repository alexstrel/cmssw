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
			// Technically should separate in different producers that create the SoAs
			// Info on SoAs : https://github.com/cms-sw/cmssw/blob/master/DataFormats/SoATemplate/README.md


			/////////////////////////////////////////////////////////////
			// Can I use these maps for proper conversion back to legacy?
			/////////////////////////////////////////////////////////////
			// In order to write out a reduced collection of matched seeds?
			// Might want to create some sort of assiciation SoA

			// std::map<int, reco::SuperClusterRef> superClusterRefMap_;
			// std::map<int, TrajectorySeed> seedRefMap_;

			int32_t i = 0;
	        for (auto& superClusRef : superClusterRefVec)
			{
				viewSCs[i].id() =  i;
				viewSCs[i].scSeedTheta() =  superClusRef->seed()->position().theta();
				viewSCs[i].scPhi() = superClusRef->position().phi();
				viewSCs[i].scR() = superClusRef->position().r();
				viewSCs[i].scEnergy() = superClusRef->energy();
				// Filling in a map with the whole object
				// superClusterRefMap_[i] = superClusRef;
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

			// reco::ElectronSeedCollection eleSeeds{};

			// for (int i = 0; i < view.metadata().size(); ++i) {
			// 	if (view[i].isMatched() > 0) {
			// 		int matchedScID = view[i].matchedScID();
			// 		auto scIter = superClusterRefMap_.find(matchedScID);
			//  		std::cout << "  matchedScID: " << view[i].matchedScID() << std::endl;

			// 		if (scIter != superClusterRefMap_.end()) {
			// 			const reco::SuperClusterRef& superClusRef = scIter->second;
			// 			auto seedIter = seedRefMap_.find(view[i].id());
			// 			if (seedIter != seedRefMap_.end()) {
			// 				const TrajectorySeed& matchedSeed = seedIter->second;
			// 				reco::ElectronSeed eleSeed(matchedSeed);
			// 				reco::ElectronSeed::CaloClusterRef caloClusRef(superClusRef);
			// 				eleSeed.setCaloCluster(caloClusRef);
			// 				eleSeeds.emplace_back(eleSeed);
			// 			}
			// 		} else {
			// 			std::cerr << "No SuperCluster found for SC ID " << matchedScID << std::endl;
			// 		}
			// 	}
			// }
			// std::cout << "New eleSeeds size " << eleSeeds.size() << std::endl;
			// superClusterRefMap_.clear();
			// seedRefMap_.clear();

			// Shouldnt I get some sort of print out from the GPU?
      		//alpaka::wait(event.queue()); 


			// For testing developments wrt legacy implementations
	        for (auto& superClusRef : event.get(superClustersTokens_)) 
			{
				float x = superClusRef->position().r() * sin(superClusRef->seed()->position().theta()) * cos(superClusRef->position().phi());
				float y = superClusRef->position().r() * sin(superClusRef->seed()->position().theta()) * sin(superClusRef->position().phi());
				float z = superClusRef->position().r() * cos(superClusRef->seed()->position().theta());
				GlobalPoint center(x, y, z);
				float theMagField = magField.inTesla(center).mag();
				Vector3d position{x,y,z};
				GlobalPoint sc(GlobalPoint::Polar(superClusRef->seed()->position().theta(),  //seed theta
                                                superClusRef->position().phi(),    //supercluster phi
                                                superClusRef->position().r()));    //supercluster r
			
				for (int charge : {1,-1}) 
				{
					auto freeTS = trackingTools::ftsFromVertexToPoint(magField, sc, vprim, superClusRef->energy(), 1);
					auto initialTrajState = TrajectoryStateOnSurface(freeTS, *PerpendicularBoundPlaneBuilder{}(freeTS.position(), freeTS.momentum()));

					auto newfreeTS = ftsFromVertexToPointPortable::ftsFromVertexToPoint(position, vertex, superClusRef->energy(),charge,magneticFieldParabolicPortable::magneticFieldAtPoint(position));			
					Vector3d testposition = {newfreeTS.position(0),newfreeTS.position(1),newfreeTS.position(2)};
					Vector3d testmomentum = {newfreeTS.momentum(0),newfreeTS.momentum(1),newfreeTS.momentum(2)};

					auto transverseCurvature = [](const Vector3d& p, int charge, const float& magneticFieldZ) -> float {
							return -2.99792458e-3f * (charge / sqrt(p(0) * p(0) + p(1) * p(1))) * magneticFieldZ;  
					};

					int notValid_old = 0;
					int notValid_new = 0;
					int seeds = 0;
					for (auto& initialSeedRef : event.get(initialSeedsToken_)) 
					{			
						++seeds;
						auto const& recHit = *(initialSeedRef.recHits().begin() + 0);  

						if(!recHit.isValid())      
							continue;

						auto state = backwardPropagator_.propagate(initialTrajState, recHit.det()->surface());

						if(!state.isValid())      
							++notValid_old;

						Vector3d recHitpos{recHit.globalPosition().x(),recHit.globalPosition().y(),recHit.globalPosition().z()};
						Vector3d surfPosition{recHit.det()->surface().position().x(),recHit.det()->surface().position().y(),recHit.det()->surface().position().z()};
						Vector3d surfRotation{recHit.det()->surface().rotation().z().x(),recHit.det()->surface().rotation().z().y(),recHit.det()->surface().rotation().z().z()};
						Vector3d x2{0,0,0};
						Vector3d p2{0,0,0};
						double rho = 0.;
						double s = 0.;					
						bool theSolExists = false;

						PlanePortable::Plane<Vector3d> plane{surfPosition,surfRotation};
						rho = transverseCurvature(testmomentum,charge,magneticFieldParabolicPortable::magneticFieldAtPoint(position));

						constexpr float small = 1.e-6;  // for orientation of planes
						auto u = plane.normalVector();
						if (std::abs(u(2)) < small) {
							// HelixBarrelPlaneCrossing,
							Propagators::helixBarrelPlaneCrossing(testposition,testmomentum,rho,Propagators::oppositeToMomentum,surfPosition,surfRotation,theSolExists,x2,p2,s);
						} 
						else if ((std::abs(u(0)) < small) && (std::abs(u(1)) < small)) 
						{
							// forward plane HelixForwardPlaneCrossing
							Propagators::helixForwardPlaneCrossing(testposition,testmomentum,rho,Propagators::oppositeToMomentum,plane,s,x2,p2,theSolExists);
						} 
						else {
							// arbitrary plane HelixArbitraryPlaneCrossing
							Propagators::helixBarrelPlaneCrossing(testposition,testmomentum,rho,Propagators::oppositeToMomentum,surfPosition,surfRotation,theSolExists,x2,p2,s);
							//Propagators::helixArbitraryPlaneCrossing(testposition,testmomentum,rho,Propagators::oppositeToMomentum,plane,s,x2,p2,theSolExists); // Should check if there is a logic bug - giving similar results but also non valid solutions
						}

						if(!theSolExists)
							++notValid_new;

						if(!state.isValid())
							continue;

						if(!theSolExists)
							continue;

						p2.normalize(); 
						p2*= testmomentum.norm();

						if(false){
							std::cout<<" New "<< rho <<"  and old "<<freeTS.transverseCurvature() <<std::endl;
							std::cout<<" recHit.det()->surface().position() "<<recHit.det()->surface().position()<<" rotation? "<<recHit.det()->surface().rotation().z()<<std::endl;
							std::cout<<" Print out legacy fts position "<< freeTS.position() <<"  and new implementation one : "<<testposition(0) <<" "<<testposition(1)<<" "<<testposition(2)<<std::endl;
							std::cout<<" Print out legacy fts momentum "<< freeTS.momentum() <<"  and new implementation one : "<<testmomentum(0) <<" "<<testmomentum(1)<<" "<<testmomentum(2) <<std::endl;
							std::cout<<" initialTrajState pos "<< initialTrajState.globalPosition()<<"  and momentum  "<<initialTrajState.globalMomentum()<<std::endl;
							std::cout<<" surfPosition " <<surfPosition(0)<<" "<<surfPosition(1)<<" "<<surfPosition(2)<<std::endl;
							std::cout<<" surfRotation " <<surfRotation(0)<<" "<<surfRotation(1)<<" "<<surfRotation(2)<<std::endl;
							std::cout<<" pt = startingDir.head(2).norm() "<< testmomentum.head(2).norm() << " and the equivalent "<< initialTrajState.globalMomentum().perp() <<std::endl;
							std::cout<<" test plane stuff  norm vec"<< plane.normalVector() << "recHit.det()->surface().normalVector "<<recHit.det()->surface().normalVector()<<std::endl;
							std::cout<<" test plane stuff localZ "<< -plane.localZ(testposition) << "recHit.det()->surface().normalVector "<< -recHit.det()->surface().localZ(GlobalPoint(initialTrajState.globalPosition()))<<std::endl;
							std::cout<<" Initial: "<< state.globalParameters().position()<<"   and new "<< x2(0) <<" "<< x2(1) << " "<< x2(2)<<std::endl;
							std::cout<<" Initial: "<< state.globalParameters().momentum()<<"   and new "<< p2(0) <<" "<< p2(1) << " "<< p2(2)<<std::endl;
							std::cout<<" The path length is : "<< s << std::endl;
							EleRelPointPair pointPair(recHit.globalPosition(), state.globalParameters().position(), vprim);
							EleRelPointPairPortable::EleRelPointPair<Vector3d> pair(recHitpos,x2,vertex);
							printf("Old point pair dZ %lf, dPerp %lf, and dPhi %lf\n",pointPair.dZ(),pointPair.dPerp(),pointPair.dPhi());
							printf("New point pair dZ %lf, dPerp %lf, and dPhi %lf \n",pair.dZ(),pair.dPerp(),pair.dPhi());
						}

						if(false){
							std::cout<<" Initial: "<< state.globalParameters().position()<<"   and new "<< x2(0) <<" "<< x2(1) << " "<< x2(2)<<std::endl;
							std::cout<<" Initial: "<< state.globalParameters().momentum()<<"   and new "<< p2(0) <<" "<< p2(1) << " "<< p2(2)<<std::endl;
						}
					}

					if(false){
						std::cout<<"Number of seeds: "<<seeds<<" Propagation notValid_old "<< notValid_old << "  notValid_new " <<notValid_new<<std::endl;
						printf("Print out legacy fts position %f and new fts position  and %lf \n",freeTS.position().x(), testposition(0));
						printf("For SC i=%d Energy is :%f , theta is :%f,  r is : %f \n",i,superClusRef->energy(),superClusRef->seed()->position().theta(),superClusRef->position().r()) ;
						printf(" view %lf ", viewSCs[i].scR());
						printf("Magnetic field full  = %f and the parabolic approximation %f ", theMagField, magneticFieldParabolicPortable::magneticFieldAtPoint(position));
					}
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

