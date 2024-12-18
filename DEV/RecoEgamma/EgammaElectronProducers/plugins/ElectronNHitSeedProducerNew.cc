#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/GeometrySurface/interface/SimpleCylinderBounds.h"
#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"

#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronUtilities.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"

//#include "RecoEgamma/EgammaElectronProducers/interface/helixToBarrelPropagator.h"
//#include "RecoEgamma/EgammaElectronProducers/interface/helixToArbitraryPlanePropagator.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/PerpendicularBoundPlaneBuilder.h"
#include "TrackingTools/TrajectoryState/interface/ftsFromVertexToPoint.h"

class ElectronNHitSeedProducerNew : public edm::global::EDProducer<> {
public:
  ElectronNHitSeedProducerNew(const edm::ParameterSet&);
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const final;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  float getZVtxFromExtrapolation(const GlobalPoint&, const GlobalPoint&, const GlobalPoint&) const;
  float getCutValue(float , float , float , float ) const;
  reco::ElectronSeedCollection acceptThisSeed(TrajectorySeed,const edm::Ref<std::vector<reco::SuperCluster>>,reco::ElectronSeedCollection) const;

private:
  const edm::EDGetTokenT<TrajectorySeedCollection> initialSeedsToken_;
  const edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;
  const edm::EDPutTokenT<reco::ElectronSeedCollection> putToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken;
  edm::EDGetTokenT<std::vector<reco::SuperClusterRef>> superClustersTokens_;
};

ElectronNHitSeedProducerNew::ElectronNHitSeedProducerNew(const edm::ParameterSet& pset)
    : initialSeedsToken_(consumes(pset.getParameter<edm::InputTag>("initialSeeds"))),
      beamSpotToken_(consumes(pset.getParameter<edm::InputTag>("beamSpot"))),
      magFieldToken_(esConsumes()),  // Is it constant magnetic field?
      putToken_{produces<reco::ElectronSeedCollection>()},
      geomToken(esConsumes()) {
        superClustersTokens_ = consumes(pset.getParameter<edm::InputTag>("superClusters"));
}

void ElectronNHitSeedProducerNew::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("initialSeeds", {"hltElePixelSeedsCombined"});
  desc.add<edm::InputTag>("beamSpot", {"hltOnlineBeamSpot"});
  desc.add<edm::InputTag>("superClusters", {"hltEgammaSuperClustersToPixelMatch"});
  descriptions.add("electronNHitSeedProducerNew", desc);
}

void ElectronNHitSeedProducerNew::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {

	bool debug = false;
	auto vprim_ = iEvent.get(beamSpotToken_).position();
	GlobalPoint vprim(vprim_.x(), vprim_.y(), vprim_.z());

	reco::ElectronSeedCollection eleSeeds{};

	auto const& magField = iSetup.getData(magFieldToken_);
	const TrackerGeometry* theG = &iSetup.getData(geomToken);

	PropagatorWithMaterial forwardPropagator_ = PropagatorWithMaterial(alongMomentum, 0.000511, &magField);
	PropagatorWithMaterial backwardPropagator_ = PropagatorWithMaterial(oppositeToMomentum, 0.000511, &magField);

	int nseeds = 0;
	for (auto& initialSeedRef : iEvent.get(initialSeedsToken_))
	++nseeds;
	int nClusters = 0;
	for (auto& superClusRef : iEvent.get(superClustersTokens_)) 
	++nClusters;

	std::cout << " ========================== START HERE ==============================" << std::endl;
	std::cout << "\n\n NEW EVENT \n ";
	std::cout << "Number of seeds in seed collection per event: " << nseeds << std::endl;
	std::cout << "Number of SCs in collection per event: " << nClusters << std::endl;

	int accepted = 0;
	int total_doub = 0;
	int total_trip = 0;
	int total_seeds = 0;

	for (auto& initialSeedRef : iEvent.get(initialSeedsToken_)) 
	{
		++total_seeds;
		bool isMatched = 0;

		if(debug){
			std::cout<<"New seed"<<std::endl;
			std::cout<<"Number of hits "<< initialSeedRef.nHits() <<std::endl;
			std::cout<<"Loop over SCs"<<std::endl;
		} 

		for (auto& superClusRef : iEvent.get(superClustersTokens_)) 
		{
			if(debug)
				std::cout<<"Supercluster information "<<superClusRef->seed()->position()<< " and energy "<<superClusRef->energy() <<std::endl;

			if(isMatched)
			continue;

			float et = superClusRef->energy() *std::sin(superClusRef->seed()->position().theta());

			GlobalPoint sc(GlobalPoint::Polar(superClusRef->seed()->position().theta(),  //seed theta
													superClusRef->position().phi(),    //supercluster phi
													superClusRef->position().r()));    //supercluster r
			for (int charge : {1,-1}) 
			{
				if(isMatched)
					continue;
				auto freeTS = trackingTools::ftsFromVertexToPoint(magField, sc, vprim, superClusRef->energy(), charge);
				auto initialTrajState = TrajectoryStateOnSurface(freeTS, *PerpendicularBoundPlaneBuilder{}(freeTS.position(), freeTS.momentum()));

				//=====  First match ==============//
				bool isFirst = true;
				auto const& recHit = *(initialSeedRef.recHits().begin() + 0);  
				if(!recHit.isValid())      
					continue;
				auto state = backwardPropagator_.propagate(initialTrajState, recHit.det()->surface());
				if(!state.isValid())
					continue;
				
				EleRelPointPair pointPair(recHit.globalPosition(), state.globalParameters().position(), vprim);
				float dRZ = recHit.geographicalId().subdetId() == PixelSubdetector::PixelBarrel ? pointPair.dZ() : pointPair.dPerp();
				float dPhiMax = getCutValue(et, 0.05, 20., -0.002);
				float dRZMax = getCutValue(et, 9999., 0., 0);

				if(debug)
				{
					std::cout<<"isFirstMatch : "<<isFirst<<std::endl;
					std::cout<<"RecHit.globalPosition() " <<recHit.globalPosition()<<" &  state.globalParameters().position() "<< state.globalParameters().position() <<" & vprim "<<vprim<<std::endl;
					std::cout<<"dPhiMax "<<dPhiMax<<"  dRZMax "<< getCutValue(et, 9999., 0., 0)<<std::endl;
					std::cout<<"dRZ "<<dRZ<<"  pointPair.dPhi()  "<< pointPair.dPhi()<<std::endl;
				}

				if (dPhiMax >= 0 && std::abs(pointPair.dPhi()) > dPhiMax)
					continue;
				if(dRZMax >= 0 && std::abs(dRZ) > dRZMax)
					continue;

				double zVertex = getZVtxFromExtrapolation(vprim, recHit.globalPosition(), sc);
				GlobalPoint vertex(vprim_.x(), vprim_.y(), zVertex);

				if(debug)
				{
					std::cout<<"Calculation of the z vertex "<<std::endl;
					std::cout<<"vprim "<<vprim<<" recHit.globalposition() "<<recHit.globalPosition()<<" sc "<<sc<<std::endl;
					std::cout<<"  zVextex result "<<zVertex<<std::endl;
				}

				//=====  Second match ==============//
				isFirst = false;
				if(debug)
					std::cout<<"isFirstMatch : "<<isFirst<<std::endl;
				
				auto const& recHit1 = *(initialSeedRef.recHits().begin() + 1);

				if(!recHit1.isValid())      
					continue;
				auto firstMatchFreeTraj = trackingTools::ftsFromVertexToPoint(magField, recHit.globalPosition(), vertex, superClusRef->energy(), charge);
				auto surf = forwardPropagator_.propagate(firstMatchFreeTraj,recHit1.det()->surface());
				if(!surf.isValid())
					continue;
				
				EleRelPointPair pointPair2(recHit1.globalPosition(), surf.globalParameters().position(), vertex);
				dRZ = recHit1.geographicalId().subdetId() == PixelSubdetector::PixelBarrel ? pointPair2.dZ() : pointPair2.dPerp();
				dPhiMax = getCutValue(et, 0.003, 0., 0.);
				dRZMax = getCutValue(et, 0.05, 30., -0.002);

				if(debug)
				{
					std::cout<<"recHit.globalPosition() " <<recHit1.globalPosition()<<"  state.globalParameters().position() "<< surf.globalParameters().position() <<" vertex "<<vertex<<std::endl;
					std::cout<<"dPhiMax "<<dPhiMax<<"  pointPair.dPhi()  "<< pointPair2.dPhi()<<std::endl;
					std::cout<<"dRZMax "<<dRZMax<<"  dRZ "<< dRZ <<std::endl;
				}

				if(dPhiMax >= 0 && std::abs(pointPair2.dPhi()) > dPhiMax)
					continue;
				if(dRZMax >= 0 && std::abs(dRZ) > dRZMax)
					continue;

				eleSeeds = acceptThisSeed(initialSeedRef, superClusRef, eleSeeds);
				++accepted;
				isMatched = 1;
				if(initialSeedRef.nHits()==2)
					++total_doub;
				if(initialSeedRef.nHits()==3)
					++total_trip;
			}
		}
	}

	std::cout<<" Parsed : "<< total_seeds<<" seeds"<<std::endl;
	std::cout<< " Accepted : "<<accepted<<std::endl;
	std::cout<< " Accepted doublets : "<<total_doub<<std::endl;
	std::cout<< " Accepted triplets : "<<total_trip<<std::endl;
	std::cout << "New eleSeeds size " << eleSeeds.size() << std::endl;
	iEvent.emplace(putToken_, std::move(eleSeeds));
}

// compute the z vertex from the candidate position and the found pixel hit
float ElectronNHitSeedProducerNew::getZVtxFromExtrapolation(const GlobalPoint& primeVtxPos,
                                                            const GlobalPoint& hitPos,
                                                            const GlobalPoint& candPos) const {
	auto sq = [](float x) { return x * x; };
	auto calRDiff = [sq](const GlobalPoint& p1, const GlobalPoint& p2) {
		return std::sqrt(sq(p2.x() - p1.x()) + sq(p2.y() - p1.y()));
	};
	const double r1Diff = calRDiff(primeVtxPos, hitPos);
	const double r2Diff = calRDiff(hitPos, candPos);
	return hitPos.z() - r1Diff * (candPos.z() - hitPos.z()) / r2Diff;
}


float ElectronNHitSeedProducerNew::getCutValue(float et, float highEt, float highEtThres, float lowEtGrad) const {
	return highEt + std::min(0.f, et - highEtThres) * lowEtGrad;
}

reco::ElectronSeedCollection ElectronNHitSeedProducerNew::acceptThisSeed(
        TrajectorySeed initialSeedRef,
        const edm::Ref<std::vector<reco::SuperCluster>> superClusRef,
        reco::ElectronSeedCollection eleSeeds) const {
	reco::ElectronSeed eleSeed(initialSeedRef);
	reco::ElectronSeed::CaloClusterRef caloClusRef(superClusRef);
	eleSeed.setCaloCluster(caloClusRef);
	eleSeed.setCaloCluster(caloClusRef);
	eleSeeds.push_back(eleSeed);  // accept this initial seed as final seed
	return eleSeeds;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ElectronNHitSeedProducerNew);
