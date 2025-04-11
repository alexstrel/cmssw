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
#include "RecoEgamma/EgammaElectronAlgos/interface/alpaka/helixArbitraryPlaneCrossing.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/alpaka/helixForwardPlaneCrossing.h"

#include "PixelMatchingAlgo.h"
#include "DataFormats/EgammaReco/interface/EleRelPointPairPortable.h"
#include "DataFormats/EgammaReco/interface/alpaka/Plane.h"

using Vector3d = Eigen::Matrix<double, 3, 1>;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

	ALPAKA_FN_ACC ALPAKA_FN_INLINE float getZVtxFromExtrapolation(const Vector3d& primeVtxPos,
											const Vector3d& hitPos,
											const Vector3d& candPos) {
		auto sq = [](float x) { return x * x; };
		auto calRDiff = [sq](const Vector3d& p1, const Vector3d& p2) {
			return sqrt(sq(p2(0) - p1(0)) + sq(p2(1) - p1(1)));
		};
		const double r1Diff = calRDiff(primeVtxPos, hitPos);
		const double r2Diff = calRDiff(hitPos, candPos);
		float zvtx = hitPos(2) - r1Diff * (candPos(2) - hitPos(2)) / r2Diff;
		return zvtx;
	}

	ALPAKA_FN_ACC ALPAKA_FN_INLINE float getCutValue(float et, float highEt, float highEtThres, float lowEtGrad) {
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
				if(i>=size)
				    break;

				auto sc = view[i];

				printf("For SC i=%d Energy is :%f , theta is :%f , and r is %lf and phi is %lf,  \n",i,sc.scEnergy(),sc.scSeedTheta(), sc.scR(),sc.scPhi() ) ;
				float x = sc.scR() * alpaka::math::sin(acc,sc.scSeedTheta()) * alpaka::math::cos(acc,sc.scPhi());
				float y = sc.scR() * alpaka::math::sin(acc,sc.scSeedTheta()) * alpaka::math::sin(acc,sc.scPhi());
				float z = sc.scR() * alpaka::math::cos(acc,sc.scSeedTheta());
				printf("x: %lf,  y: %lf,  z %lf ",x,z,y);
				Vector3d position{x,y,z};
				printf("  Value of perp2 %lf \n",x*x+y*y);
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

			for (int32_t i : uniform_elements(acc, size)) 
			{

				if(i>=size)
    				break;

				auto seed = view[i];

				printf("For electron seed %d number of hits is %d \n ",i,seed.nHits());
				printf("Seed id: %d is matched? %d and the matchesSC id %d \n",seed.id(),seed.isMatched(),seed.matchedScID());
				printf("Hit 0 is valid: %d\n",seed.hit0isValid());
				printf("Hit 0 detId is : %d\n",seed.hit0detectorID());
				printf("Hit 1 is valid: %d\n",seed.hit1isValid());
				printf("Hit 1 detId is : %d\n",seed.hit1detectorID());
				printf("Hit 0 position is (%lf , %lf , %lf ) \n", seed.hit0Pos().x(),seed.hit0Pos().y(),seed.hit0Pos().z());
				printf("Hit 0 surface is (%lf , %lf , %lf ) \n", seed.surf0Pos().x(),seed.surf0Pos().y(),seed.surf0Pos().z());
				printf("Hit 0 rotation is (%lf , %lf , %lf ) \n", seed.surf0Rot().x(),seed.surf0Rot().y(),seed.surf0Rot().z());
				printf("Hit 1 position is (%lf , %lf , %lf ) \n", seed.hit1Pos().x(),seed.hit1Pos().y(),seed.hit1Pos().z());
				printf("Hit 1 surface is (%lf , %lf , %lf ) \n", seed.surf1Pos().x(),seed.surf1Pos().y(),seed.surf1Pos().z());
				printf("Hit 1 rotation is (%lf , %lf , %lf ) \n", seed.surf1Rot().x(),seed.surf1Rot().y(),seed.surf1Rot().z());
				if(seed.nHits()>2){
					printf("Hit 2 is valid: %d",seed.hit2isValid());
					printf("Hit 2 detId is : %d",seed.hit2detectorID());
					printf("Hit 2 position is (%lf , %lf , %lf ) \n ", seed.hit2Pos().x(),seed.hit2Pos().y(),seed.hit2Pos().z());
					printf("Hit 2 surface is (%lf , %lf , %lf ) \n", seed.surf2Pos().x(),seed.surf2Pos().y(),seed.surf2Pos().z());
					printf("Hit 2 rotation is (%lf , %lf , %lf ) \n", seed.surf2Rot().x(),seed.surf2Rot().y(),seed.surf2Rot().z());
				}
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

			Vector3d vertex = {vtx_x,vtx_y,vtx_z}; 

			for (int32_t i : uniform_elements(acc,sizeEleSeeds)) 
			{

				auto eleSeed = viewEleSeeds[i];

				if(!(eleSeed.hit0isValid()))
					continue;

				// Access first hit information
				Vector3d hitPosition  = {eleSeed.hit0Pos().x(),eleSeed.hit0Pos().y(),eleSeed.hit0Pos().z()};
				Vector3d surfPosition = {eleSeed.surf0Pos().x(),eleSeed.surf0Pos().y(),eleSeed.surf0Pos().z()};
				Vector3d surfRotation = {eleSeed.surf0Rot().x(),eleSeed.surf0Rot().y(),eleSeed.surf0Rot().z()};

				Vector3d hit2Position  = {eleSeed.hit1Pos().x(),eleSeed.hit1Pos().y(),eleSeed.hit1Pos().z()};
				Vector3d surf2Position = {eleSeed.surf1Pos().x(),eleSeed.surf1Pos().y(),eleSeed.surf1Pos().z()};
				Vector3d surf2Rotation = {eleSeed.surf1Rot().x(),eleSeed.surf1Rot().y(),eleSeed.surf1Rot().z()};

				for(int32_t j=0; j<sizeSCs; ++j)
				{
					float x = viewSCs[j].scR() * alpaka::math::sin(acc,viewSCs[j].scSeedTheta()) * alpaka::math::cos(acc,viewSCs[j].scPhi());
					float y = viewSCs[j].scR() * alpaka::math::sin(acc,viewSCs[j].scSeedTheta()) * alpaka::math::sin(acc,viewSCs[j].scPhi());
					float z = viewSCs[j].scR() * alpaka::math::cos(acc,viewSCs[j].scSeedTheta());
					float et = viewSCs[j].scEnergy() * alpaka::math::sin(acc, viewSCs[j].scSeedTheta());
					Vector3d positionSC = {x,y,z};

					for (int charge : {1,-1}) 
					{
						auto newfreeTS = ftsFromVertexToPointPortable::ftsFromVertexToPoint(positionSC,vertex, viewSCs[j].scEnergy(),charge,magneticFieldParabolicPortable::magneticFieldAtPoint(positionSC));									
						Vector3d position = {newfreeTS.position(0),newfreeTS.position(1),newfreeTS.position(2)};
						Vector3d momentum = {newfreeTS.momentum(0),newfreeTS.momentum(1),newfreeTS.momentum(2)};
						
						auto transverseCurvature = [](const Vector3d& p, int charge, const float& magneticFieldZ) -> float {
   			 				return -2.99792458e-3f * (charge / sqrt(p(0) * p(0) + p(1) * p(1))) * magneticFieldZ;  
						};

						double s=0; 
						bool theSolExists = false; 
						Vector3d propagatedPos = {0,0,0};
						Vector3d propagatedMom = {0,0,0}; 
						double rho = transverseCurvature(momentum,charge,magneticFieldParabolicPortable::magneticFieldAtPoint(positionSC));

						PlanePortable::Plane<Vector3d> plane{surfPosition,surfRotation};
						constexpr float small = 1.e-6; 
						auto u = plane.normalVector();
						if (std::abs(u(2)) < small) {
							Propagators::helixBarrelPlaneCrossing(position,momentum,rho,Propagators::oppositeToMomentum,surfPosition,surfRotation,theSolExists,propagatedPos,propagatedMom,s);
						} 
						else if ((std::abs(u(0)) < small) && (std::abs(u(1)) < small)) 
						{
							Propagators::helixForwardPlaneCrossing(position,momentum,rho,Propagators::oppositeToMomentum,plane,s,propagatedPos,propagatedMom,theSolExists);
						} 
						else {
							Propagators::helixBarrelPlaneCrossing(position,momentum,rho,Propagators::oppositeToMomentum,surfPosition,surfRotation,theSolExists,propagatedPos,propagatedMom,s);
							// Propagators::helixArbitraryPlaneCrossing(position,momentum,rho,Propagators::oppositeToMomentum,plane,s,propagatedPos,propagatedMom,theSolExists);
						}
						
						if(!theSolExists)
							continue;
						// Momentum should be renormalized - might want to add this in propagator ?
						propagatedMom.normalize(); 
						propagatedMom*= momentum.norm();

						// Construct relative point-pair struct for applying quality cuts
						EleRelPointPairPortable::EleRelPointPair<Vector3d> pair(hitPosition,propagatedPos,vertex);
						float dPhiMax = getCutValue(et, 0.05, 20., -0.002);
						float dRZMax = getCutValue(et, 9999., 0., 0);
						float dRZ = pair.dZ();
						if(eleSeed.hit0detectorID() != 1)
							dRZ = pair.dPerp(); 
						if ((dPhiMax >= 0 && std::abs(pair.dPhi()) > dPhiMax) || (dRZMax >= 0 && std::abs(dRZ) > dRZMax))
							continue;
						double zVertex = getZVtxFromExtrapolation(vertex, hitPosition,positionSC);
						Vector3d vertexUpdated(vertex(0),vertex(1), zVertex);
						// printf("Position : (%lf,%lf,%lf) and direction : (%lf,%lf,%lf) and path length : %lf \n",propagatedPos(0),propagatedPos(1),propagatedPos(2),propagatedMom(0),propagatedMom(1),propagatedMom(2),s);

						// Move to the second hit of the seed
						if(!(eleSeed.hit1isValid()))
							continue;	
						auto firstMatchFreeTraj = ftsFromVertexToPointPortable::ftsFromVertexToPoint(hitPosition,vertexUpdated,viewSCs[j].scEnergy(),charge,magneticFieldParabolicPortable::magneticFieldAtPoint(hitPosition));									
						Vector3d position2 = {firstMatchFreeTraj.position(0),firstMatchFreeTraj.position(1),firstMatchFreeTraj.position(2)};
						Vector3d momentum2 = {firstMatchFreeTraj.momentum(0),firstMatchFreeTraj.momentum(1),firstMatchFreeTraj.momentum(2)};
						rho = transverseCurvature(momentum2,charge,magneticFieldParabolicPortable::magneticFieldAtPoint(hitPosition));
						theSolExists = false; 
						propagatedPos = {0,0,0};
						propagatedMom = {0,0,0}; 
						PlanePortable::Plane<Vector3d> plane2{surf2Position,surf2Rotation};
						u = plane2.normalVector();

						if (std::abs(u(2)) < small) {
							Propagators::helixBarrelPlaneCrossing(position2,momentum2,rho,Propagators::alongMomentum,surf2Position,surf2Rotation,theSolExists,propagatedPos,propagatedMom,s);
						}
						else if ((std::abs(u(0)) < small) && (std::abs(u(1)) < small)) 
						{
							Propagators::helixForwardPlaneCrossing(position2,momentum2,rho,Propagators::alongMomentum,plane2,s,propagatedPos,propagatedMom,theSolExists);
						} 
						else {
							Propagators::helixBarrelPlaneCrossing(position2,momentum2,rho,Propagators::alongMomentum,surf2Position,surf2Rotation,theSolExists,propagatedPos,propagatedMom,s);
							// Propagators::helixArbitraryPlaneCrossing(position2,momentum2,rho,Propagators::alongMomentum,plane2,s,propagatedPos,propagatedMom,theSolExists);
						}								

						propagatedMom.normalize(); 
						propagatedMom*= momentum.norm();

						if(!theSolExists)
							continue;
						
						// printf("Position : (%lf,%lf,%lf) and direction : (%lf,%lf,%lf) and path length : %lf \n",propagatedPos(0),propagatedPos(1),propagatedPos(2),propagatedMom(0),propagatedMom(1),propagatedMom(2),s);
						
						EleRelPointPairPortable::EleRelPointPair<Vector3d> pair2(hit2Position,propagatedPos,vertexUpdated);
						dPhiMax = getCutValue(et, 0.003, 0., 0.);
						dRZMax = getCutValue(et, 0.05, 30., -0.002);
						dRZ = pair2.dZ();
						if(eleSeed.hit1detectorID()!= 1)
							dRZ = pair2.dPerp(); 
						if ((dPhiMax >= 0 && std::abs(pair2.dPhi()) > dPhiMax) || (dRZMax >= 0 && std::abs(dRZ) > dRZMax))
							continue;	
						eleSeed.matchedScID() = viewSCs[j].id();
						eleSeed.isMatched() = 1;

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
