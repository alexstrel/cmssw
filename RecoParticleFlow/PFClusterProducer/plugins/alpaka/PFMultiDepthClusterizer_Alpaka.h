#ifndef PFClusterProducer_plugins_alpaka_PFMultiDepthClusterizer_Alpaka_h
#define PFClusterProducer_plugins_alpaka_PFMultiDepthClusterizer_Alpaka_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "HeterogeneousCore/AlpakaCore/interface/MoveToDeviceCache.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFClusterSoAProducerKernel.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/PFRecHitTopologyRecord.h"

#include "HeterogeneousCore/AlpakaMath/interface/deltaPhi.h"

/**
 * @class PFMultiDepthClusterizer_Alpaka
 * @brief Alpaka clusterizer algorithm for multi-depth particle flow clusters.
 * 
 * This class manages the execution of the full multi-stage particle flow clustering pipeline
 * using Alpaka, including link building, adjacency graph construction,
 * connected component detection (ECL-CC), and postprocessing.
 */

namespace ALPAKA_ACCELERATOR_NAMESPACE {


    class PFMultiDepthClusterizer_Alpaka {
        public:
            PFMultiDepthClusterizer_Alpaka(Queue& queue, const edm::ParameterSet& conf) :   nSigmaEta_(cms::alpakatools::make_device_buffer<double>(queue)),
                                                                                            nSigmaPhi_(cms::alpakatools::make_device_buffer<double>(queue)) { 
                const double _nSigmaEta = pow(conf.getParameter<double>("nSigmaEta"), 2);
                const double _nSigmaPhi = pow(conf.getParameter<double>("nSigmaPhi"), 2); 
                //
                alpaka::memcpy(queue, nSigmaEta_, &_nSigmaEta, sizeof(double));
                alpaka::memcpy(queue, nSigmaPhi_, &_nSigmaPhi, sizeof(double)); 
            }
      
            PFMultiDepthClusterizer_Alpaka(const PFMultiDepthClusterizer_Alpaka&)            = delete;
            PFMultiDepthClusterizer_Alpaka& operator=(const PFMultiDepthClusterizer_Alpaka&) = delete;

            void clusterize(Queue& queue, 
                            reco::PFMultiDepthClusteringVarsDeviceCollection& mdpfClusteringVars, 
                            const reco::PFRecHitDeviceCollection& pfRecHits);

        private:     
            cms::alpakatools::device_buffer<Device, double> nSigmaEta_; 
            cms::alpakatools::device_buffer<Device, double> nSigmaPhi_; 
    };
}

#endif

