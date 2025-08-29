#include <iostream>
#include <string>

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/EgammaReco/interface/EleSeedHostCollection.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"


class ElectronSeedConverter : public edm::global::EDProducer<> {
public:
  explicit ElectronSeedConverter(const edm::ParameterSet&);

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const final;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
    const edm::EDGetTokenT<TrajectorySeedCollection> initialSeedsToken_;
    const edm::EDGetTokenT<reco::EleSeedHostCollection> matchedEleSeedSoAToken_; 
    const edm::EDPutTokenT<reco::ElectronSeedCollection> putToken_;
	edm::EDGetTokenT<std::vector<reco::SuperClusterRef>> superClustersTokens_;
};


void ElectronSeedConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("initialSeeds", {"hltElePixelSeedsCombined"});
    desc.add<edm::InputTag>("eleSeedsSoA", edm::InputTag{"ElectronNHitSeedAlpakaProducer"});
  	desc.add<edm::InputTag>("superClusters", {"hltEgammaSuperClustersToPixelMatch"});
    descriptions.add("electronNHitSeedConverter", desc);
}

ElectronSeedConverter::ElectronSeedConverter(const edm::ParameterSet& pset)
    :initialSeedsToken_(consumes(pset.getParameter<edm::InputTag>("initialSeeds"))),
    matchedEleSeedSoAToken_(consumes<reco::EleSeedHostCollection>(pset.getParameter<edm::InputTag>("eleSeedsSoA"))),    
    putToken_{produces<reco::ElectronSeedCollection>()} {
	    superClustersTokens_ = consumes(pset.getParameter<edm::InputTag>("superClusters"));
    }

void ElectronSeedConverter::produce(edm::StreamID, edm::Event& event, const edm::EventSetup& iSetup) const {

    auto const& view = event.get(matchedEleSeedSoAToken_).const_view();

    reco::ElectronSeedCollection eleSeeds{};
    // auto eleSeeds = std::make_unique<reco::ElectronSeedCollection>();
    // eleSeeds->reserve(eleSeedSoAView.metadata().size());

    std::map<int, reco::SuperClusterRef> superClusterRefMap_;
    std::map<int, TrajectorySeed> seedRefMap_;

    int i=0;
	for (auto& superClusRef : event.get(superClustersTokens_)) {
		superClusterRefMap_[i] = superClusRef;
		++i;
    }

    i=0;
    for (auto& initialSeedRef : event.get(initialSeedsToken_)) {
        seedRefMap_[i] = initialSeedRef;  
        ++i;
    }

    for (int i = 0; i < view.metadata().size(); ++i) {
        if (view[i].isMatched() > 0) {
            int matchedScID = view[i].matchedScID();
            auto scIter = superClusterRefMap_.find(matchedScID);
            // std::cout << "  matchedScID: " << view[i].matchedScID() << std::endl;

            if (scIter != superClusterRefMap_.end()) {
                const reco::SuperClusterRef& superClusRef = scIter->second;
                auto seedIter = seedRefMap_.find(view[i].id());
                if (seedIter != seedRefMap_.end()) {
                    const TrajectorySeed& matchedSeed = seedIter->second;
                    reco::ElectronSeed eleSeed(matchedSeed);
                    reco::ElectronSeed::CaloClusterRef caloClusRef(superClusRef);
                    eleSeed.setCaloCluster(caloClusRef);
                    eleSeeds.emplace_back(eleSeed);
                }
            } else {
                std::cerr << "No SuperCluster found for SC ID " << matchedScID << std::endl;
            }
        }
    }
    std::cout << "New eleSeeds size " << eleSeeds.size() << std::endl;
    superClusterRefMap_.clear();
    seedRefMap_.clear();  

    event.emplace(putToken_, std::move(eleSeeds));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ElectronSeedConverter);
