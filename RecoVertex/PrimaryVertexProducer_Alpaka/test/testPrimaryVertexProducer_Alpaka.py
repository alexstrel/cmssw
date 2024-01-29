import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3

process = cms.Process('PV',Run3)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('HLTrigger.Configuration.HLT_GRun_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.load('Configuration.StandardSequences.Accelerators_cff')
process.load('HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi')

# Input files
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_13_3_0_pre4/RelValTTbar_14TeV/GEN-SIM-RECO/PU_133X_mcRun4_realistic_v1_13_3_0_pre_4_RV217_D98-v1/2590000/03a3903f-b0d8-4a49-a0b8-c7a0742c2bb3.root'),
    secondaryFileNames = cms.untracked.vstring(),
)

# Number of events to run
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2),
)

# Production metadata
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('PV nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition
process.FEVToutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RECO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('test.root'), # output file name
    outputCommands = cms.untracked.vstring('drop *', 'keep *_tracksSoA_*_*', 'keep *_beamSpotSoA_*_*', 'keep *_vertexSoA_*_*', 'keep *_vertexAoS_*_*'),# I.e., just drop everything and keep things in this module
    splitLevel = cms.untracked.int32(0)
)

# Endpath and output
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVToutput_step = cms.EndPath(process.FEVToutput)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)

import sys
import argparse
parser = argparse.ArgumentParser(prog=f"{sys.argv[0]} {sys.argv[1]} --", description='Test and validation of PrimaryVertexProducer_Alpaka')
parser.add_argument('-b', '--backend', type=str, default='auto',
                    help='Alpaka backend. Possible options: CPU, GPU, auto. Default: auto')
args = parser.parse_args()

alpaka_backends = {
    "cpu": "alpaka_serial_sync::%s",  # Execute on CPU
    "gpu": "alpaka_cuda_async::%s",   # Execute using CUDA / TODO: set up so it also checks ROCm
    "cuda": "alpaka_cuda_async::%s",  # Execute using CUDA
    "auto": "%s@alpaka"               # Let framework choose
}
alpaka_backend_str = "alpaka_serial_sync::%s" #alpaka_backends[args.backend]

################################
## Now the plugins themselves ##
################################

# Convert reco::Track to portable Track
process.tracksSoA = cms.EDProducer(alpaka_backend_str % "PortableTrackSoAProducer",
    TrackLabel    = cms.InputTag("generalTracks"),
    BeamSpotLabel = cms.InputTag("offlineBeamSpot"),
    # For historical reasons, parameters are set up as in the legacy code of RecoVertex/PrimaryVertexProducer/python/OfflinePrimaryVertices_cfi.py
    
    TkFilterParameters = cms.PSet(
        maxNormalizedChi2 = cms.double(10.0),
        minPixelLayersWithHits=cms.int32(2),
        minSiliconLayersWithHits = cms.int32(5),
        maxD0Significance = cms.double(4.0), 
        maxD0Error = cms.double(1.0), 
        maxDzError = cms.double(1.0), 
        minPt = cms.double(0.0),
        maxEta = cms.double(2.4),
        trackQuality = cms.string("any"),
        vertexSize = cms.double(0.006),
        d0CutOff   = cms.double(3.)
    )
)

# Convert reco::BeamSpot to portable BeamSpot
process.beamSpotSoA = cms.EDProducer(alpaka_backend_str % "PortableBeamSpotSoAProducer",
    BeamSpotLabel = cms.InputTag("offlineBeamSpot")
)

process.vertexSoA = cms.EDProducer(alpaka_backend_str % "PrimaryVertexProducer_Alpaka",
    TrackLabel = cms.InputTag("tracksSoA"),
    BeamSpotLabel = cms.InputTag("beamSpotSoA"),
    blockOverlap = cms.double(0.50),
    blockSize    = cms.int32(512),
    TkFitterParameters = cms.PSet(
        chi2cutoff = cms.double(2.5),
        minNdof=cms.double(0.0),
        useBeamSpotConstraint = cms.bool(True),
        maxDistanceToBeam = cms.double(1.0)
    ),
    TkClusParameters = cms.PSet(    
        coolingFactor = cms.double(0.6),  # moderate annealing speed
        zrange = cms.double(4.),          # consider only clusters within 4 sigma*sqrt(T) of a track
        delta_highT = cms.double(1.e-2),  # convergence requirement at high T
        delta_lowT = cms.double(1.e-3),   # convergence requirement at low T
        convergence_mode = cms.int32(0),  # 0 = two steps, 1 = dynamic with sqrt(T)
        Tmin = cms.double(2.0),           # end of vertex splitting
        Tpurge = cms.double(2.0),         # cleaning
        Tstop = cms.double(0.5),          # end of annealing
        vertexSize = cms.double(0.006),   # added in quadrature to track-z resolutions
        d0CutOff = cms.double(3.),        # downweight high IP tracks
        dzCutOff = cms.double(3.),        # outlier rejection after freeze-out (T<Tmin)
        zmerge = cms.double(1e-2),        # merge intermediat clusters separated by less than zmerge
        uniquetrkweight = cms.double(0.8),# require at least two tracks with this weight at T=Tpurge
        uniquetrkminp = cms.double(0.0),  # minimal a priori track weight for counting unique tracks
    ) 
)

process.vertexAoS = cms.EDProducer("SoAToRecoVertexProducer",
    soaVertex = cms.InputTag("vertexSoA"),
    srcTrack  = cms.InputTag("generalTracks")
)

## TODO: Vertex producer

## TODO: Conversion back to reco::vertex


###################################
## Last, organize paths and exec ##
###################################

process.vertexing_task = cms.EndPath(process.tracksSoA + process.beamSpotSoA + process.vertexSoA + process.vertexAoS)
process.schedule = cms.Schedule(process.vertexing_task)
process.schedule.extend([process.endjob_step,process.FEVToutput_step])
