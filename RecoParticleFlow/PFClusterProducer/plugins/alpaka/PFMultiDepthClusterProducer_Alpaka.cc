#include "HeterogeneousCore/AlpakaInterface/interface/alpaka/AlpakaCore.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "RecoParticleFlow/PFClusterProducer/interface/InitialClusteringStepBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterBuilderBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterEnergyCorrectorBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/RecHitTopologicalCleanerBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/SeedFinderBase.h"
#include "CondFormats/DataRecord/interface/HcalPFCutsRcd.h"
#include "CondTools/Hcal/interface/HcalPFCutsHandler.h"

#include <memory>

#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthClusterizer_Alpaka.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringVarsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringVarsHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitDeviceCollection.h"

class PFMultiDepthClusterSoAProducer : public stream::SynchronizingEDProducer<> {
  public:
      PFMultiDepthClusterSoAProducer(edm::ParameterSet const& config);

      void beginRun(const edm::Run&, const edm::EventSetup&) override;
      void acquire(device::Event const& event, device::EventSetup const&) override;
      void produce(device::Event& event, device::EventSetup const&) override;
      //
      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
  private:
      edm::EDPutTokenT<reco::PFClusterHostCollection> outputClustersHostToken_;

      const bool synchronise_;
      //
      edm::ParameterSet conf_;  
  
      std::optional<PFMultiDepthClusterizer_Alpaka> clusterizer_;  
      std::optional<reco::PFMultiDepthClusteringVarsDeviceCollection> mdpfClusteringVars;
      //
      // extra:
      std::unique_ptr<PFCPositionCalculatorBase> _allCellsPosCalc;
      // options
      // the actual algorithm
      std::unique_ptr<PFClusterBuilderBase> _pfClusterBuilder;
      std::unique_ptr<PFClusterEnergyCorrectorBase> _energyCorrector;      
      // inputs
      edm::EDGetTokenT<reco::PFClusterCollection> _clustersLabel;
      edm::ESGetToken<HcalPFCuts, HcalPFCutsRcd> hcalCutsToken_;
      //
      bool cutsFromDB;
      HcalPFCuts const* paramPF = nullptr;
  };
  
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(PFMultiDepthClusterSoAProducer);

void PFMultiDepthClusterSoAProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("clustersSource", {});
  desc.add<edm::ParameterSetDescription>("energyCorrector", {});
  {
    edm::ParameterSetDescription pset0;
    pset0.add<std::string>("algoName", "PFMultiDepthClusterizer");
    {
      edm::ParameterSetDescription pset1;
      pset1.add<std::string>("algoName", "Basic2DGenericPFlowPositionCalc");
      {
        edm::ParameterSetDescription psd;
        psd.add<std::vector<int>>("depths", {});
        psd.add<std::string>("detector", "");
        psd.add<std::vector<double>>("logWeightDenominator", {});
        pset1.addVPSet("logWeightDenominatorByDetector", psd, {});
      }
      pset1.add<double>("minAllowedNormalization", 1e-09);
      pset1.add<double>("minFractionInCalc", 1e-09);
      pset1.add<int>("posCalcNCrystals", -1);
      pset1.add<edm::ParameterSetDescription>("timeResolutionCalcBarrel", {});
      pset1.add<edm::ParameterSetDescription>("timeResolutionCalcEndcap", {});
      pset0.add<edm::ParameterSetDescription>("allCellsPositionCalc", pset1);
    }
    pset0.add<edm::ParameterSetDescription>("positionCalc", {});
    pset0.add<double>("minFractionToKeep", 1e-07);
    pset0.add<double>("nSigmaEta", 2.0);
    pset0.add<double>("nSigmaPhi", 2.0);
    desc.add<edm::ParameterSetDescription>("pfClusterBuilder", pset0);
  }
  desc.add<edm::ParameterSetDescription>("positionReCalc", {});
  desc.add<bool>("usePFThresholdsFromDB", false);
  descriptions.addWithDefaultLabel(desc);
}

PFMultiDepthClusterSoAProducer::PFMultiDepthClusterSoAProducer(const edm::ParameterSet& config)
  : //inputPFRecHitDeviceToken_(consumes(config.getParameter<edm::InputTag>("pfRecHitsDevice"))),
    //outputClustersHostToken_(produces<reco::PFClusterHostCollection>()),
    //outputPFClusterDeviceToken_(produces()),
    synchronise_(config.getParameter<bool>("synchronise")),
    conf_(config) {

  _clustersLabel = consumes<reco::PFClusterCollection>(conf.getParameter<edm::InputTag>("clustersSource"));

  cutsFromDB = conf.getParameter<bool>("usePFThresholdsFromDB");

  const edm::ParameterSet& pfcConf = conf.getParameterSet("pfClusterBuilder");

  edm::ConsumesCollector&& cc = consumesCollector();

  const auto& acConf = conf.getParameterSet("allCellsPositionCalc");
  if (!acConf.empty()) {
    const std::string& algoac = acConf.getParameter<std::string>("algoName");
    if (!algoac.empty())
      _allCellsPosCalc = PFCPositionCalculatorFactory::get()->create(algoac, acConf, cc);
  }

  if (cutsFromDB) {
    hcalCutsToken_ = esConsumes<HcalPFCuts, HcalPFCutsRcd, edm::Transition::BeginRun>(edm::ESInputTag("", "withTopo"));
  }

  if (!pfcConf.empty()) {
    const std::string& pfcName = pfcConf.getParameter<std::string>("algoName");
    if (!pfcName.empty())
      _pfClusterBuilder = PFClusterBuilderFactory::get()->create(pfcName, pfcConf, cc);
  }
  // see if new need to apply corrections, setup if there.
  const edm::ParameterSet& cConf = conf.getParameterSet("energyCorrector");

  if (!cConf.empty()) {
    const std::string& cName = cConf.getParameter<std::string>("algoName");
    if (!cName.empty())
      _energyCorrector = PFClusterEnergyCorrectorFactory::get()->create(cName, cConf);
  }
  outputClustersHostToken_ = produces<reco::PFClusterHostCollection>();
}

void PFMultiDepthClusterSoAProducer::acquire(device::Event const& event, device::EventSetup const&) {
  edm::Handle<reco::PFClusterCollection> inputClustersHandle;
  event.getByToken(_clustersLabel, inputClustersHandle);

  const auto& inputClusters = *inputClustersHandle;

  const auto nClusters = inputClusters.size();
  //const auto nClusters = inputClusters.nSeed(); 

  int nRHFracs = 0;

  for (unsigned int i = 0; i < nClusters; ++i) nRHFracs += inputClusters[i].recHitFractions().size();

  PFMultiDepthClusteringVarsHostCollection mdpfClusteringVarsHost(nClusters, cms::alpakatools::host());
  PFRecHitHostCollection pfRecHitsHost(nRHFracs, cms::alpakatools::host());
  //

  auto mdpfcvars_view = mdpfClusteringVarsHost.view();
  auto rhits_view     = pfRecHitsHost.view();

  mdpfcvars_view.size() = nClusters;

  int rhfOffset = 0;

  for (unsigned int i = 0; i < nClusters; ++i) {
    const auto& cluster = inputClusters[i];

    mdpfcvars_view[i].depth()  = cluster.depth();
    mdpfcvars_view[i].energy() = cluster.energy();

    double etaSum = 0.0;
    double phiSum = 0.0;

    auto const& crep = cluster.positionREP();
  
    mdpfcvars_view[i].eta() = crep.eta();
    mdpfcvars_view[i].phi() = crep.phi();

    int idx = 0;
    for (const auto& frac : cluster.recHitFractions()) {
      auto const& h = *frac.recHitRef();
      auto const& rep = h.positionREP();

      etaSum += (frac.fraction() * h.energy()) * std::abs(rep.eta() - crep.eta());
      phiSum += (frac.fraction() * h.energy()) * std::abs(deltaPhi(rep.phi(), crep.phi()));

      rhits_view[rhfOffset+idx].energy() = h.energy();
      rhits_view[rhfOffset+idx].detId()  = h.detId();

      idx += 1;
    }

    mdpfcvars_view[i].rhfracOffset() = rhfOffset;
    mdpfcvars_view[i].rhfracSize()   = idx;

    rhfOffset += idx;

    //protection for single line : assign ~ tower
    const double etaRMS2_ = std::max(etaSum / cluster.energy(), 0.1);
    mdpfcvars_view[i].etaRMS2() = etaRMS2_ * etaRMS2_;
    const double phiRMS2_ = std::max(phiSum / cluster.energy(), 0.1);
    mdpfcvars_view[i].phiRMS2() = phiRMS2_ * phiRMS2_;
  }

  mdpfClusteringVars.emplace(nClusters, event.queue());
  pfRecHits.emplace(nClusters, event.queue());

  alpaka::memcpy(event.queue(), mdpfClusteringVars.view(), mdpfClusteringVarsHost.view());
  alpaka::memcpy(event.queue(), pfRecHits.view(), pfRecHitsHost.view());

  alpaka::wait(event.queue());

  if (!clusterizer_) {
    // Initialize clusterizer at first event
    clusterizer_.emplace(event.queue(), conf_.getParameterSet("pfClusterBuilder"));
  }

  if (nClusters > 0) {
    clusterizer_->apply(event.queue(), mdpfClusteringVars, pfRecHits);
  }
}

void PFMultiDepthClusterSoAProducer::beginRun(const edm::Run& run, const edm::EventSetup& es) {
  if (cutsFromDB) {
    paramPF = &es.getData(hcalCutsToken_);
  }
  _pfClusterBuilder->update(es);
}

void PFMultiDepthClusterSoAProducer::produce(device::Event& event, const device::EventSetup& eventSetup) {
  edm::Handle<reco::PFClusterCollection> inputClustersHandle;
  event.getByToken(_clustersLabel, inputClustersHandle);

  const auto& inputClusters = *inputClustersHandle;

  const auto nClusters = inputClusters.size();
  //const auto nClusters = inputClusters.nSeed();
  reco::PFMultiDepthClusteringVarsHostCollection mdpfClusteringVarsHost(nClusters, cms::alpakatools::host());

  alpaka::memcpy(event.queue(), mdpfClusteringVarsHost.view(), mdpfClusteringVars.const_view());

  alpaka::wait(event.queue());

  const auto nTopos = mdpfClusteringVarsHost.mdpf_nTopos();

  PFClusterHostCollection pfClusters(nTopos, cms::alpakatools::host());

  for (int c = 0; c < nTopos; c++) {
    const auto comp_offset  = mdpfClusteringVarsHost[c].mdpf_componentIndex();
    const auto comp_size    = mdpfClusteringVarsHost[c+1].mdpf_componentIndex() - comp_offset;
    //
    const auto comp_rep_idx = mdpfClusteringVarsHost[comp_offset].mdpf_component();

    reco::PFCluster rep_cluster = inputClusters[comp_rep_idx];

    float best_energy = mdpfClusteringVarsHost[comp_offset].mdpf_componentEnergy();

    auto comp_seed = rep_cluster.seed();

    for (int i = 1; i < comp_size; i++) {
      const auto idx = comp_offset + i;
      const auto comp_cluster_idx = mdpfClusteringVarsHost[idx].mdpf_component();

      const reco::PFCluster& added_cluster = inputClusters[comp_cluster_idx];

      for (const auto& fraction : added_cluster.recHitFractions()) rep_cluster.addRecHitFraction(fraction);

      const float comp_energy = mdpfClusteringVarsHost[idx].mdpf_componentEnergy();

      if (comp_energy > best_energy) {
        comp_seed   = added_cluster.seed();
        best_energy = comp_energy;
      }
    }

    rep_cluster.addSeed(comp_seed);

    _allCellsPosCalc->calculateAndSetPosition(rep_cluster, paramPF);

    pfClusters.push_back(rep_cluster);
  }

  if (_energyCorrector) {
    _energyCorrector->correctEnergies(*pfClusters);
  }
  event.emplace(outputClustersHostToken_, std::move(pfClusters));
}
