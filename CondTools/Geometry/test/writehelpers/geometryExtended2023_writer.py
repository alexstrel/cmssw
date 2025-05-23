import FWCore.ParameterSet.Config as cms
import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Generate XML geometry.')
parser.add_argument("--tag", help="global tag to use", type=str)
args = parser.parse_args()


import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_DDD_cff import Run3_DDD

process = cms.Process("GeometryWriter", Run3_DDD)

process.load('CondCore.CondDB.CondDB_cfi')

process.load('Configuration.Geometry.GeometryExtended2023_cff')
process.load('Geometry.CaloEventSetup.CaloGeometryDBWriter_cfi')
process.load('CondTools.Geometry.HcalParametersWriter_cff')
process.load("Geometry.MuonNumbering.muonGeometryConstants_cff")
process.load("Geometry.ForwardGeometry.ZdcGeometry_cfi")

process.CaloGeometryBuilder = cms.ESProducer("CaloGeometryBuilder",
    SelectedCalos = cms.vstring(
        'HCAL',
        'ZDC',
        'EcalBarrel',
        'EcalEndcap',
        'EcalPreshower',
        'TOWER'
    )
)

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

# This reads the big XML file and the only way to fill the
# nonreco part of the database is to read this file.
process.XMLGeometryWriter = cms.EDAnalyzer("XMLGeometryBuilder",
                                           XMLFileName = cms.untracked.string("./geSingleBigFile.xml"),
                                           ZIP = cms.untracked.bool(True)
                                           )

process.TrackerGeometryWriter = cms.EDAnalyzer("PGeometricDetBuilder",fromDD4hep=cms.bool(False))
process.TrackerParametersWriter = cms.EDAnalyzer("PTrackerParametersDBBuilder",fromDD4hep=cms.bool(False))

process.CaloGeometryWriter = cms.EDAnalyzer("PCaloGeometryBuilder",fromDD4hep = cms.untracked.bool(False))

process.CSCGeometryWriter = cms.EDAnalyzer("CSCRecoIdealDBLoader",fromDD4hep = cms.untracked.bool(False))

process.DTGeometryWriter = cms.EDAnalyzer("DTRecoIdealDBLoader",fromDD4hep = cms.untracked.bool(False))

process.RPCGeometryWriter = cms.EDAnalyzer("RPCRecoIdealDBLoader",fromDD4hep = cms.untracked.bool(False))

process.GEMGeometryWriter = cms.EDAnalyzer("GEMRecoIdealDBLoader",fromDD4hep = cms.untracked.bool(False))

process.CondDB.timetype = cms.untracked.string('runnumber')
process.CondDB.connect = cms.string('sqlite_file:myfile.db')
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDB,
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('GeometryFileRcd'),tag = cms.string('XMLFILE_Geometry_'+args.tag+'_Extended2023_mc')),
                                                            cms.PSet(record = cms.string('IdealGeometryRecord'),tag = cms.string('TKRECO_Geometry_'+args.tag)),
                                                            cms.PSet(record = cms.string('PTrackerParametersRcd'),tag = cms.string('TKParameters_Geometry_'+args.tag)),
                                                            cms.PSet(record = cms.string('PEcalBarrelRcd'),   tag = cms.string('EBRECO_Geometry_'+args.tag)),
                                                            cms.PSet(record = cms.string('PEcalEndcapRcd'),   tag = cms.string('EERECO_Geometry_'+args.tag)),
                                                            cms.PSet(record = cms.string('PEcalPreshowerRcd'),tag = cms.string('EPRECO_Geometry_'+args.tag)),
                                                            cms.PSet(record = cms.string('PHcalRcd'),         tag = cms.string('HCALRECO_Geometry_'+args.tag)),
                                                            cms.PSet(record = cms.string('HcalParametersRcd'), tag = cms.string('HCALParameters_Geometry_'+args.tag)),
                                                            cms.PSet(record = cms.string('PCaloTowerRcd'),    tag = cms.string('CTRECO_Geometry_'+args.tag)),
                                                            cms.PSet(record = cms.string('PZdcRcd'),          tag = cms.string('ZDCRECO_Geometry_'+args.tag)),
                                                            cms.PSet(record = cms.string('PCastorRcd'),       tag = cms.string('CASTORRECO_Geometry_'+args.tag)),
                                                            cms.PSet(record = cms.string('CSCRecoGeometryRcd'),tag = cms.string('CSCRECO_Geometry_'+args.tag)),
                                                            cms.PSet(record = cms.string('CSCRecoDigiParametersRcd'),tag = cms.string('CSCRECODIGI_Geometry_'+args.tag)),
                                                            cms.PSet(record = cms.string('DTRecoGeometryRcd'),tag = cms.string('DTRECO_Geometry_'+args.tag)),
                                                            cms.PSet(record = cms.string('RPCRecoGeometryRcd'),tag = cms.string('RPCRECO_Geometry_'+args.tag)),
                                                            cms.PSet(record = cms.string('GEMRecoGeometryRcd'),tag = cms.string('GEMRECO_Geometry_'+args.tag))
                                                        )
                                    )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.p1 = cms.Path(process.XMLGeometryWriter+process.TrackerGeometryWriter+process.TrackerParametersWriter+process.CaloGeometryWriter+process.HcalParametersWriter+process.CSCGeometryWriter+process.DTGeometryWriter+process.RPCGeometryWriter+process.GEMGeometryWriter)
