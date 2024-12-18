// Check that ALPAKA_HOST_ONLY is not defined during device compilation:
#ifdef ALPAKA_HOST_ONLY
#error ALPAKA_HOST_ONLY defined in device compilation
#endif

#include <alpaka/alpaka.hpp>
#include <Eigen/Core>

#include "DataFormats/EgammaReco/interface/alpaka/EleSeedDeviceCollection.h"
#include "DataFormats/EgammaReco/interface/alpaka/SuperclusterDeviceCollection.h"
#include "MagneticField/ParametrizedEngine/interface/ParabolicParametrizedMagneticField.h"

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoEgamma/EgammaElectronAlgos/interface/alpaka/helixBarrelPlaneCrossingByCircle.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/alpaka/ftsFromVertexToPointPortable.h"

#include "PixelMatchingAlgo.h"
#include "DataFormats/EgammaReco/interface/EleRelPointPairPortable.h"


using Vector3f = Eigen::Matrix<double, 3, 1>;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

	ALPAKA_FN_ACC ALPAKA_FN_INLINE float getZVtxFromExtrapolation(const Vector3f& primeVtxPos,
											const Vector3f& hitPos,
											const Vector3f& candPos) {
		auto sq = [](float x) { return x * x; };
		auto calRDiff = [sq](const Vector3f& p1, const Vector3f& p2) {
			return sqrt(sq(p2(0) - p1(0)) + sq(p2(1) - p1(1)));
		};
		const double r1Diff = calRDiff(primeVtxPos, hitPos);
		const double r2Diff = calRDiff(hitPos, candPos);
		return hitPos(2) - r1Diff * (candPos(2) - hitPos(2)) / r2Diff;
	}

	ALPAKA_FN_ACC ALPAKA_FN_INLINE float getCutValue(const float et, 
	                                                 const float highEt, 
													 const float highEtThres, 
													 const float lowEtGrad) {
		return highEt + std::min(0.f, et - highEtThres) * lowEtGrad;
	}

	//--- Kernel for printing the SC SoA
	class printSCSoA {
	    public:
	        template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
	    	ALPAKA_FN_ACC void operator()(TAcc const& acc,
									      reco::SuperclusterDeviceCollection::View view,
									      int32_t size) const 
			{
				//const int32_t thread = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];
				for (int32_t i : uniform_elements(acc, size)) 
				{
					printf("For SC i=%d Energy is :%f , theta is :%f , and r is %lf and phi is %lf,  \n",i,view[i].scEnergy(),view[i].scSeedTheta(), view[i].scR(),view[i].scPhi() ) ;
					const float x = view[i].scR() * alpaka::math::sin(acc,view[i].scSeedTheta()) * alpaka::math::cos(acc,view[i].scPhi());
					const float y = view[i].scR() * alpaka::math::sin(acc,view[i].scSeedTheta()) * alpaka::math::sin(acc,view[i].scPhi());
					const float z = view[i].scR() * alpaka::math::cos(acc,view[i].scSeedTheta());
					printf("x: %lf,  y: %lf,  z %lf",x,z,y);
					Vector3f position{x,y,z};
					printf("Value of perp2 %lf \n",x*x+y*y);
					printf("Calculate the magnetic field with the parabolic approximation at the SC position : %f\n",magneticFieldParabolicPortable::magneticFieldAtPoint(position));
				}
		}
	};

   	//--- Kernel for printing the electron seeds SoA
	class printElectronSeedSoA {
		public:
			template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
			ALPAKA_FN_ACC void operator()(TAcc const& acc,
									      reco::EleSeedDeviceCollection::View view,
									      int32_t size) const 
			{
				//const int32_t thread = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];
				printf("Printed from device : \n");
				for (int32_t i : uniform_elements(acc, size)) 
				{
					printf("For electron seed %d number of hits is %d \n ",i,view[i].nHits());
					printf("Hit 0 position is (%lf , %lf , %lf ) \n", view[i].hitPosX().x(),view[i].hitPosY().x(),view[i].hitPosZ().x());
					printf("Hit 1 position is (%lf , %lf , %lf ) \n", view[i].hitPosX().y(),view[i].hitPosY().y(),view[i].hitPosZ().y());
					if(view[i].nHits()>2)
						printf("Hit 2 position is (%lf , %lf , %lf ) \n ", view[i].hitPosX().z(),view[i].hitPosY().z(),view[i].hitPosZ().z());
				}
			}
	};

	//--- Kernel for performing the seed to SC match
	//--- Still dummy and just testing out some things
	class SeedToSuperClusterMatcher {
	public:
		template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
		ALPAKA_FN_ACC void operator()(TAcc const& acc,
										reco::EleSeedDeviceCollection::View viewEleSeeds,
										int32_t sizeEleSeeds,
										reco::SuperclusterDeviceCollection::View viewSCs,
										int32_t sizeSCs,
										double vtx_x,
										double vtx_y,
										double vtx_z) const 
		{

			//const int32_t thread = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];

			Vector3f vertex = {vtx_x,vtx_y,vtx_z}; 

			for (int32_t i : uniform_elements(acc,sizeEleSeeds)) 
			{

				if(!(viewEleSeeds[i].isValid().x()))
					continue;

				// Access first hit information
				Vector3f hitPosition  = {viewEleSeeds[i].hitPosX().x(),viewEleSeeds[i].hitPosY().x(),viewEleSeeds[i].hitPosZ().x()};
				Vector3f surfPosition = {viewEleSeeds[i].surfPosX().x(),viewEleSeeds[i].surfPosY().x(),viewEleSeeds[i].surfPosZ().x()};
				Vector3f surfRotation = {viewEleSeeds[i].surfRotX().x(),viewEleSeeds[i].surfRotY().x(),viewEleSeeds[i].surfRotZ().x()};

				Vector3f hit2Position  = {viewEleSeeds[i].hitPosX().y(),viewEleSeeds[i].hitPosY().y(),viewEleSeeds[i].hitPosZ().y()};
				Vector3f surf2Position = {viewEleSeeds[i].surfPosX().y(),viewEleSeeds[i].surfPosY().y(),viewEleSeeds[i].surfPosZ().y()};
				Vector3f surf2Rotation = {viewEleSeeds[i].surfRotX().y(),viewEleSeeds[i].surfRotY().y(),viewEleSeeds[i].surfRotZ().y()};

				for(int32_t j : uniform_elements(acc,sizeSCs)) 
				{
					float x = viewSCs[j].scR() * alpaka::math::sin(acc,viewSCs[j].scSeedTheta()) * alpaka::math::cos(acc,viewSCs[j].scPhi());
					float y = viewSCs[j].scR() * alpaka::math::sin(acc,viewSCs[j].scSeedTheta()) * alpaka::math::sin(acc,viewSCs[j].scPhi());
					float z = viewSCs[j].scR() * alpaka::math::cos(acc,viewSCs[j].scSeedTheta());
					float et = viewSCs[j].scEnergy() * alpaka::math::sin(acc, viewSCs[j].scSeedTheta());
					Vector3f positionSC = {x,y,z};

					for (int charge : {1,-1}) 
					{
						auto newfreeTS = ftsFromVertexToPointPortable::ftsFromVertexToPoint(positionSC,vertex, viewSCs[j].scEnergy(),charge,magneticFieldParabolicPortable::magneticFieldAtPoint(positionSC));									
						Vector3f position = {newfreeTS.position(0),newfreeTS.position(1),newfreeTS.position(2)};
						Vector3f momentum = {newfreeTS.momentum(0),newfreeTS.momentum(1),newfreeTS.momentum(2)};
						
						auto transverseCurvature = [](const Vector3f& p, int charge, const float& magneticFieldZ) -> float {
   			 				return -2.99792458e-3f * (charge / sqrt(p(0) * p(0) + p(1) * p(1))) * magneticFieldZ;  
						};

						double s=0; 
						bool theSolExists = false; 
						Vector3f propagatedPos = {0,0,0};
						Vector3f propagatedMom = {0,0,0}; 
						double rho = transverseCurvature(momentum,charge,magneticFieldParabolicPortable::magneticFieldAtPoint(positionSC));
						Propagators::helixBarrelPlaneCrossing(position,momentum,rho,Propagators::oppositeToMomentum,surfPosition,surfRotation,theSolExists,propagatedPos,propagatedMom,s);
						if(!theSolExists)
							continue;
						// Momentum should be renormalized - might want to add this in propagator ?
						propagatedMom.normalize(); 
						propagatedMom*= momentum.norm();

						// Construct relative point-pair struct for applying quality cuts
						EleRelPointPairPortable::EleRelPointPair<Vector3f> pair(hitPosition,propagatedPos,vertex);
						float dPhiMax = getCutValue(et, 0.05, 20., -0.002);
						float dRZMax = getCutValue(et, 9999., 0., 0);
						float dRZ = pair.dZ();
						if(viewEleSeeds[i].detectorID().x() != 1)
							dRZ = pair.dPerp(); 
						if ((dPhiMax >= 0 && std::abs(pair.dPhi()) > dPhiMax) || (dRZMax >= 0 && std::abs(dRZ) > dRZMax))
							continue;
						double zVertex = getZVtxFromExtrapolation(vertex, hitPosition,positionSC);
						Vector3f vertexUpdated(vertex(0),vertex(1), zVertex);

						printf("Position : (%lf,%lf,%lf) and direction : (%lf,%lf,%lf) and path length : %lf \n",propagatedPos(0),propagatedPos(1),propagatedPos(2),propagatedMom(0),propagatedMom(1),propagatedMom(2),s);
						printf("Vertex Z updated %lf \n",zVertex);

						// Move to the second hit of the seed
						if(!(viewEleSeeds[i].isValid().y()))
							continue;	
						auto firstMatchFreeTraj = ftsFromVertexToPointPortable::ftsFromVertexToPoint(hitPosition,vertexUpdated,viewSCs[j].scEnergy(),charge,magneticFieldParabolicPortable::magneticFieldAtPoint(hitPosition));									
						Vector3f position2 = {firstMatchFreeTraj.position(0),firstMatchFreeTraj.position(1),firstMatchFreeTraj.position(2)};
						Vector3f momentum2 = {firstMatchFreeTraj.momentum(0),firstMatchFreeTraj.momentum(1),firstMatchFreeTraj.momentum(2)};
						rho = transverseCurvature(momentum2,charge,magneticFieldParabolicPortable::magneticFieldAtPoint(hitPosition));
						theSolExists = false; 
						propagatedPos = {0,0,0};
						propagatedMom = {0,0,0}; 
						Propagators::helixBarrelPlaneCrossing(position2,momentum2,rho,Propagators::alongMomentum,surf2Position,surf2Rotation,theSolExists,propagatedPos,propagatedMom,s);
						propagatedMom.normalize(); 
						propagatedMom*= momentum.norm();
						if(!theSolExists)
							continue;
						printf("Position : (%lf,%lf,%lf) and direction : (%lf,%lf,%lf) and path length : %lf \n",propagatedPos(0),propagatedPos(1),propagatedPos(2),propagatedMom(0),propagatedMom(1),propagatedMom(2),s);
						EleRelPointPairPortable::EleRelPointPair<Vector3f> pair2(hit2Position,propagatedPos,vertexUpdated);
						dPhiMax = getCutValue(et, 0.003, 0., 0.);
						dRZMax = getCutValue(et, 0.05, 30., -0.002);
						dRZ = pair2.dZ();
						if(viewEleSeeds[i].detectorID().y() != 1)
							dRZ = pair2.dPerp(); 
						if ((dPhiMax >= 0 && std::abs(pair2.dPhi()) > dPhiMax) || (dRZMax >= 0 && std::abs(dRZ) > dRZMax))
							continue;
					}
				}
			}
		}
	};


	//---- Kernel launch for printing the SC SoA collection
	void PixelMatchingAlgo::printEleSeeds(Queue& queue, reco::EleSeedDeviceCollection& collection) const {
		uint32_t items = 32;
		uint32_t groups = divide_up_by(collection->metadata().size(), items);
		auto workDiv = make_workdiv<Acc1D>(groups, items);
		alpaka::exec<Acc1D>(queue, workDiv, printElectronSeedSoA{}, collection.view(), collection->metadata().size());
	}

	//---- Kernel launch for printing the SC SoA collection
	void PixelMatchingAlgo::printSCs(Queue& queue, reco::SuperclusterDeviceCollection& collection) const {
		uint32_t items = 32;
		uint32_t groups = divide_up_by(collection->metadata().size(), items);
		auto workDiv = make_workdiv<Acc1D>(groups, items);
		alpaka::exec<Acc1D>(queue, workDiv, printSCSoA{}, collection.view(), collection->metadata().size());
	}

	//---- Kernel launch for SC and seed matching
	void PixelMatchingAlgo::matchSeeds(Queue& queue, reco::EleSeedDeviceCollection& collection, reco::SuperclusterDeviceCollection& collectionSCs, double vtx_X, double vtx_Y, double vtx_Z) const 
	{
		uint32_t items = 32;
		uint32_t groups = divide_up_by(collection->metadata().size(), items);
		auto workDiv = make_workdiv<Acc1D>(groups, items);
		alpaka::exec<Acc1D>(queue, workDiv, SeedToSuperClusterMatcher{},collection.view(),collection->metadata().size(),collectionSCs.view(),collectionSCs->metadata().size(),vtx_X, vtx_Y,vtx_Z);
	}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
