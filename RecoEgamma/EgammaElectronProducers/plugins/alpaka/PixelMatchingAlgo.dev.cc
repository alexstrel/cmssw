// Check that ALPAKA_HOST_ONLY is not defined during device compilation:
#ifdef ALPAKA_HOST_ONLY
#error ALPAKA_HOST_ONLY defined in device compilation
#endif

#include <alpaka/alpaka.hpp>

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
#include "DataFormats/EgammaReco/interface/alpaka/EleRelPointPairPortable.h"
#include "DataFormats/EgammaReco/interface/alpaka/Plane.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

        template<typename TAcc, typename T>
	ALPAKA_FN_ACC ALPAKA_FN_INLINE T getZVtxFromExtrapolation(TAcc const& acc,
								  const Vec3d& primeVtxPos,
								  const Vec3d& hitPos,
								  const Vec3d& candPos) {
		auto sq = [](T x) { return x * x; };

		auto calRDiff2 = [sq](const Vec3d& p1, const Vec3d& p2) {
		  return sq(p2[0] - p1[0]) + sq(p2[1] - p1[1]);
		};
		const T r1Diff = alpaka::math::sqrt(acc, calRDiff2(primeVtxPos, hitPos));
		const T r2Diff = alpaka::math::sqrt(acc, calRDiff2(hitPos, candPos));

		T zvtx = hitPos[2] - r1Diff * (candPos[2] - hitPos[2]) / r2Diff;
		return zvtx;
	}

	template<typename TAcc, typename T>
	ALPAKA_FN_ACC ALPAKA_FN_INLINE T getCutValue(TAcc const& acc, const T et, const T highEt, const T highEtThres, const T lowEtGrad) {
		return highEt + alpaka::math::min(acc, static_cast<T>(0.), et - highEtThres) * lowEtGrad;
	}

	//--- Kernel for printing the SC SoA
	class printSCSoA {
	public:
	template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
	ALPAKA_FN_ACC void operator()(TAcc const& acc,
									reco::SuperclusterDeviceCollection::View view,
									int32_t size) const 
		{
#if 0
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
				printf("Calculate the magnetic field with the parabolic approximation at the SC position : %f\n",magneticFieldParabolicPortable::magneticFieldAtPoint(acc, position));			
			}
#endif
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
					      const int32_t sizeEleSeeds,
					      reco::SuperclusterDeviceCollection::View viewSCs,
					      const int32_t sizeSCs,
					      const double vtx_x,
					      const double vtx_y,
					      const double vtx_z) const 
		{

			const Vec3d vertex(vtx_x,vtx_y,vtx_z); 

			for (int i : uniform_elements(acc,sizeEleSeeds)) 
			{
				if(i>=sizeEleSeeds)
					break;

				auto eleSeed = viewEleSeeds[i];

				if(!(eleSeed.hit0isValid()))
					continue;

				// Access first hit information
				Vec3d hitPosition(eleSeed.hit0Pos().x(),eleSeed.hit0Pos().y(),eleSeed.hit0Pos().z());
				Vec3d surfPosition(eleSeed.surf0Pos().x(),eleSeed.surf0Pos().y(),eleSeed.surf0Pos().z());
				Vec3d surfRotation(eleSeed.surf0Rot().x(),eleSeed.surf0Rot().y(),eleSeed.surf0Rot().z());

				Vec3d hit2Position(eleSeed.hit1Pos().x(),eleSeed.hit1Pos().y(),eleSeed.hit1Pos().z());
				Vec3d surf2Position(eleSeed.surf1Pos().x(),eleSeed.surf1Pos().y(),eleSeed.surf1Pos().z());
				Vec3d surf2Rotation(eleSeed.surf1Rot().x(),eleSeed.surf1Rot().y(),eleSeed.surf1Rot().z());

				for(int j = 0; j < sizeSCs; ++j)
				{

					if(j >= sizeSCs) break;

			 		const double x = viewSCs[j].scR() * alpaka::math::sin(acc,viewSCs[j].scSeedTheta()) * alpaka::math::cos(acc,viewSCs[j].scPhi());				
					const double y = viewSCs[j].scR() * alpaka::math::sin(acc,viewSCs[j].scSeedTheta()) * alpaka::math::sin(acc,viewSCs[j].scPhi());
			 		const double  z = viewSCs[j].scR() * alpaka::math::cos(acc,viewSCs[j].scSeedTheta());
			 		
					const float et = viewSCs[j].scEnergy() * alpaka::math::sin(acc, viewSCs[j].scSeedTheta());
					const float e = viewSCs[j].scEnergy();

			 		Vec3d positionSC(x,y,z);
	
			 		for (int charge : {1,-1}) 
			 		{
						const float c = (charge == 1 ? -2.99792458e-3f : +2.99792458e-3f);

			 			auto newfreeTS = ftsFromVertexToPointPortable::ftsFromVertexToPoint(acc,positionSC,vertex,e,charge,magneticFieldParabolicPortable::magneticFieldAtPoint(acc, positionSC));	
												
			 			const Vec3d position(newfreeTS.get_position());
			 			const Vec3d momentum(newfreeTS.get_momentum());

			 			double s=0; 
			 			bool theSolExists = false; 

			 			Vec3d propagatedPos(0);
			 			Vec3d propagatedMom(0);

			 			double rho = (c * magneticFieldParabolicPortable::magneticFieldAtPoint(acc, positionSC)) /  momentum.partial_norm<TAcc, 2>(acc);
					
			 			PlanePortable::Plane<typename Vec3d::value_type> plane(surfPosition,surfRotation);

			 			constexpr float small = 1.e-6; 

			 			auto u = plane.normalVector();
			 			if (alpaka::math::abs(acc,u[2]) < small) {
			 				Propagators::helixBarrelPlaneCrossing<TAcc, Propagators::PropagationDirection::oppositeToMomentum>(acc,position,momentum,rho,surfPosition,surfRotation,theSolExists,propagatedPos,propagatedMom,s);
			 			} 
						else if ((alpaka::math::abs(acc, u[0]) < small) && (alpaka::math::abs(acc, u[1]) < small)) 
			 			{
			 				//Propagators::helixForwardPlaneCrossing(position,momentum,rho,Propagators::oppositeToMomentum,plane,s,propagatedPos,propagatedMom,theSolExists);
			 				Propagators::helixBarrelPlaneCrossing<TAcc, Propagators::PropagationDirection::oppositeToMomentum>(acc,position,momentum,rho,surfPosition,surfRotation,theSolExists,propagatedPos,propagatedMom,s);

			 			} 
			 			else {
			 				Propagators::helixBarrelPlaneCrossing<TAcc, Propagators::PropagationDirection::oppositeToMomentum>(acc,position,momentum,rho,surfPosition,surfRotation,theSolExists,propagatedPos,propagatedMom,s);
			 				// Propagators::helixArbitraryPlaneCrossing(position,momentum,rho,Propagators::oppositeToMomentum,plane,s,propagatedPos,propagatedMom,theSolExists);
			 			}
	
						if(!theSolExists) continue;

						// Momentum should be renormalized - might want to add this in propagator ?
						const double scale = momentum.norm(acc) / propagatedMom.norm(acc);

						propagatedMom *= scale; 

						// Construct relative point-pair struct for applying quality cuts
						EleRelPointPairPortable::EleRelPointPair<typename Vec3d::value_type> pair(hitPosition,propagatedPos,vertex);

						const float dPhiMax = getCutValue(acc, et, 0.05f, 20.f, -0.002f);
						const float dRZMax  = getCutValue(acc, et, 9999.f, 0.f, 0.f);
						const float dRZ     = eleSeed.hit0detectorID() ? pair.dPerp(acc) : pair.dZ();

						if ((dPhiMax >= 0 && alpaka::math::abs(acc, pair.dPhi(acc)) > dPhiMax) || (dRZMax >= 0 && alpaka::math::abs(acc,dRZ) > dRZMax))
							continue;
						
						const double zVertex = getZVtxFromExtrapolation<TAcc, typename Vec3d::value_type>(acc, vertex, hitPosition,positionSC);
						Vec3d vertexUpdated(vertex[0],vertex[1], zVertex);
					
						// Move to the second hit of the seed
						if(!(eleSeed.hit1isValid()))
							continue;	

						auto firstMatchFreeTraj = ftsFromVertexToPointPortable::ftsFromVertexToPoint(acc, hitPosition,vertexUpdated,e,charge,magneticFieldParabolicPortable::magneticFieldAtPoint(acc, hitPosition));									
						Vec3d position2(firstMatchFreeTraj.get_position());
						Vec3d momentum2(firstMatchFreeTraj.get_momentum());
						
						rho = (c * magneticFieldParabolicPortable::magneticFieldAtPoint(acc, hitPosition)) /  momentum2.partial_norm<TAcc, 2>(acc);
						
						theSolExists = false; 
						propagatedPos = Vec3d(0);
						propagatedMom = Vec3d(0); 

			 			PlanePortable::Plane<typename Vec3d::value_type> plane2(surf2Position,surf2Rotation);
			 			u = plane2.normalVector();

			 			if (alpaka::math::abs(acc, u[2]) < small) {
			 				Propagators::helixBarrelPlaneCrossing<TAcc, Propagators::PropagationDirection::alongMomentum>(acc,position2,momentum2,rho,surf2Position,surf2Rotation,theSolExists,propagatedPos,propagatedMom,s);
			 			}
			 			else if ((alpaka::math::abs(acc, u[0]) < small) && (alpaka::math::abs(acc,u[1]) < small)) 
			 			{
			 				Propagators::helixBarrelPlaneCrossing<TAcc, Propagators::PropagationDirection::alongMomentum>(acc,position2,momentum2,rho,surf2Position,surf2Rotation,theSolExists,propagatedPos,propagatedMom,s);
							// Propagators::helixForwardPlaneCrossing(position2,momentum2,rho,Propagators::alongMomentum,plane2,s,propagatedPos,propagatedMom,theSolExists);
			 			} 
			 			else {
			 				Propagators::helixBarrelPlaneCrossing<TAcc, Propagators::PropagationDirection::alongMomentum>(acc, position2,momentum2,rho,surf2Position,surf2Rotation,theSolExists,propagatedPos,propagatedMom,s);
			 				// Propagators::helixArbitraryPlaneCrossing(position2,momentum2,rho,Propagators::alongMomentum,plane2,s,propagatedPos,propagatedMom,theSolExists);
			 			}								

						const double scale2 = momentum.norm(acc) / propagatedMom.norm(acc);
						propagatedMom *= scale2;

						//propagatedMom.normalize(); 
						//propagatedMom*= momentum.norm();

						if(!theSolExists)
							continue;
								
						EleRelPointPairPortable::EleRelPointPair<typename Vec3d::value_type> pair2(hit2Position,propagatedPos,vertexUpdated);

						const float dPhiMax2 = getCutValue(acc, et, 0.003f, 0.f, 0.f);
						const float dRZMax2  = getCutValue(acc, et, 0.05f, 30.f, -0.002f);
						const float dRZ2     = eleSeed.hit1detectorID()!= 1 ? pair2.dPerp(acc) : pair2.dZ();

					
						if ((dPhiMax2 >= 0 && alpaka::math::abs(acc,pair2.dPhi(acc)) > dPhiMax2) || (dRZMax2 >= 0 && alpaka::math::abs(acc,dRZ2) > dRZMax2))
							continue;	

						eleSeed.matchedScID() = static_cast<int16_t>(viewSCs[j].id());
						eleSeed.isMatched()   = static_cast<int16_t>(1);

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
	    auto nSeeds = static_cast<uint32_t>(collection->metadata().size());
		uint32_t groups = divide_up_by(nSeeds, items);

		if(groups<1)
			return;
		auto workDiv = make_workdiv<Acc1D>(groups, items);
		alpaka::exec<Acc1D>(queue, workDiv, SeedToSuperClusterMatcher{},collection.view(),collection->metadata().size(),collectionSCs.view(),collectionSCs->metadata().size(),vtx_X, vtx_Y,vtx_Z);
	}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
