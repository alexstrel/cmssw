#include <iostream>
#include <string>

#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EgammaReco/interface/EleSeedHostCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

class EleSeedSoAToLegacy : public edm::stream::EDProducer<> {
public:
    explicit EleSeedSoAToLegacy(edm::ParameterSet const& ps);
    ~EleSeedSoAToLegacy() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
    void produce(edm::Event&, edm::EventSetup const&) override;
    const edm::EDGetTokenT<reco::EleSeedHostCollection> inputEleSeedToken_; 
    const edm::EDPutTokenT<reco::ElectronSeedCollection> outputEleSeedLegacyToken_;
};

void EleSeedSoAToLegacy::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("eleSeedsSoA", edm::InputTag{"ElectronNHitSeedAlpakaProducer"});
    descriptions.addWithDefaultLabel(desc);
}


EleSeedSoAToLegacy::EleSeedSoAToLegacy(const edm::ParameterSet& ps)
    : inputEleSeedToken_{consumes<reco::EleSeedHostCollection>(ps.getParameter<edm::InputTag>("eleSeedsSoA"))},
      outputEleSeedLegacyToken_{produces<reco::ElectronSeedCollection>()} {}


void EleSeedSoAToLegacy::produce(edm::Event& event, edm::EventSetup const& setup) {

    // Populate the legacy collection
    auto eleSeedsLegacy = std::make_unique<reco::ElectronSeedCollection>();
    auto const& eleSeedSoAView = event.get(inputEleSeedToken_).const_view();
    eleSeedsLegacy->reserve(eleSeedSoAView.metadata().size());

    reco::ElectronSeedCollection electronSeeds;

    for (auto i = 0; i < eleSeedSoAView.metadata().size(); i++) {
        auto const& seed = eleSeedSoAView[i];        
        reco::ElectronSeed seedLeg; 
        eleSeedsLegacy->emplace_back(seedLeg);
    }

    event.put(outputEleSeedLegacyToken_, std::move(eleSeedsLegacy));
}

DEFINE_FWK_MODULE(EleSeedSoAToLegacy);

