#ifndef L1Trigger_Phase2L1ParticleFlow_L1TCorrelatorLayer1PatternFileWriter_h
#define L1Trigger_Phase2L1ParticleFlow_L1TCorrelatorLayer1PatternFileWriter_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "L1Trigger/DemonstratorTools/interface/BoardDataWriter.h"
#include "L1Trigger/DemonstratorTools/interface/utilities.h"

#include "DataFormats/L1TParticleFlow/interface/layer1_emulator.h"

class L1TCorrelatorLayer1PatternFileWriter {
public:
  L1TCorrelatorLayer1PatternFileWriter(const edm::ParameterSet& iConfig, const l1ct::Event& eventTemplate);
  ~L1TCorrelatorLayer1PatternFileWriter();

  static edm::ParameterSetDescription getParameterSetDescription();

  void write(const l1ct::Event& event);
  void flush();

private:
  enum class Partition { Barrel, HGCal, HGCalNoTk, HF };

  Partition partition_;
  const unsigned int tmuxFactor_;
  bool writeInputs_, writeOutputs_, writeDebugs_;
  std::map<l1t::demo::LinkId, std::vector<size_t>> channelIdsInput_, channelIdsOutput_;
  std::map<std::string, l1t::demo::ChannelSpec> channelSpecsInput_, channelSpecsOutput_;

  // The TmuxFactor represents the Tmux of the inputs. The LinksFactor is the number of links per sector.
  // They are not configurable in the current architecture.
  const unsigned int tfTmuxFactor_ = 18, tfLinksFactor_ = 1;
  const unsigned int hgcTmuxFactor_ = 18, hgcLinksFactor_ = 4;
  const unsigned int gctEmTmuxFactor_ = 6, gctEmLinksFactor_ = 1;
  const unsigned int gctHadTmuxFactor_ = 6, gctHadLinksFactor_ = 1;
  const unsigned int gmtTmuxFactor_ = 18, gmtLinksFactor_ = 1;
  const unsigned int gttTmuxFactor_ = 6, gttLinksFactor_ = 1;
  const unsigned int tfTimeslices_, hgcTimeslices_, gctEmTimeslices_, gctHadTimeslices_, gmtTimeslices_, gttTimeslices_;
  uint32_t gctLinksEcal_, gctLinksHad_;
  bool gctSingleLink_;
  uint32_t gmtNumberOfMuons_;
  uint32_t gttNumberOfPVs_;
  uint32_t tfNumberOfTracks_;  // if this messes up Serenity, change to int32 and only apply if not negative?
  uint32_t gctNumberOfEMs_;
  uint32_t gctNumberOfHads_;
  uint32_t gttLatency_;

  std::vector<uint32_t> outputRegions_, outputLinksPuppi_;
  unsigned int nPuppiFramesPerRegion_;
  int32_t outputBoard_, outputLinkEgamma_;
  uint32_t nEgammaObjectsOut_;

  // for debug
  const uint32_t nPFInTrack_;
  const uint32_t nPFInEmCalo_;
  const uint32_t nPFInHadCalo_;
  const uint32_t nPFInMuon_;
  const uint32_t nPFOutCharged_;
  const uint32_t nPFOutPhoton_;
  const uint32_t nPFOutNeutral_;
  const uint32_t nPFOutMuon_;

  // Common stuff related to the format
  uint32_t nInputFramesPerBX_, nOutputFramesPerBX_;
  const std::string fileFormat_;

  // final helper
  const uint32_t eventsPerFile_;
  uint32_t eventIndex_;
  std::unique_ptr<l1t::demo::BoardDataWriter> inputFileWriter_, outputFileWriter_, debugFileWriter_;

  static Partition parsePartition(const std::string& partition);

  static std::unique_ptr<edm::ParameterDescriptionNode> describeTF();
  static std::unique_ptr<edm::ParameterDescriptionNode> describeGCTEm();
  static std::unique_ptr<edm::ParameterDescriptionNode> describeGCTHad();
  static std::unique_ptr<edm::ParameterDescriptionNode> describeHGC();
  static std::unique_ptr<edm::ParameterDescriptionNode> describeGMT();
  static std::unique_ptr<edm::ParameterDescriptionNode> describeGTT();
  static std::unique_ptr<edm::ParameterDescriptionNode> describePuppi();
  static std::unique_ptr<edm::ParameterDescriptionNode> describeEG();

  void configTimeSlices(const edm::ParameterSet& iConfig,
                        const std::string& prefix,
                        unsigned int nSectors,
                        unsigned int nTimeSlices,
                        unsigned int linksFactor);
  static std::unique_ptr<edm::ParameterDescriptionNode> describeTimeSlices(const std::string& prefix);
  void configSectors(const edm::ParameterSet& iConfig,
                     const std::string& prefix,
                     unsigned int nSectors,
                     unsigned int linksFactor);
  static std::unique_ptr<edm::ParameterDescriptionNode> describeSectors(const std::string& prefix);
  void configLinks(const edm::ParameterSet& iConfig,
                   const std::string& prefix,
                   unsigned int linksFactor,
                   unsigned int offset);
  static std::unique_ptr<edm::ParameterDescriptionNode> describeLinks(const std::string& prefix);

  void writeTF(const l1ct::Event& event, l1t::demo::EventData& out);
  void writeGCTEm(const l1ct::Event& event, l1t::demo::EventData& out);
  void writeGCTHad(const l1ct::Event& event, l1t::demo::EventData& out);
  void writeHGC(const l1ct::Event& event, l1t::demo::EventData& out);
  void writeGMT(const l1ct::Event& event, l1t::demo::EventData& out);
  void writeGTT(const l1ct::Event& event, l1t::demo::EventData& out);
  void writePuppi(const l1ct::Event& event, l1t::demo::EventData& out);
  void writeEgamma(const l1ct::OutputBoard& egboard, std::vector<ap_uint<64>>& out);
  void writeEgamma(const l1ct::Event& event, l1t::demo::EventData& out);
  void writeDebugs(const l1ct::Event& event, l1t::demo::EventData& out);
};

#endif
