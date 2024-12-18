/**
 Description: Function to propagate from a helix to a plane on the GPU
*/

#ifndef RecoEgamma_EgammaElectronAlgos_interface_alpaka_helixArbitraryPlaneCrossing2Order_h
#define RecoEgamma_EgammaElectronAlgos_interface_alpaka_helixArbitraryPlaneCrossing2Order_h

#include <Eigen/Dense>
#include "DataFormats/EgammaReco/interface/Plane.h"
#include <cmath>
#include <cfloat>

using Vector3f = Eigen::Matrix<double, 3, 1>;//Vector3f -> Vector3d

namespace ALPAKA_ACCELERATOR_NAMESPACE {

	namespace Propagators {

		namespace PlaneCrossing2Order {

			ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE double smallestPathLength(const double firstPathLength, const double secondPathLength) {
				return fabs(firstPathLength) < fabs(secondPathLength) ? firstPathLength : secondPathLength;
			}

			ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE Vector3f positionInDouble(const double theRho,
				                                                          const double s, 
																		  const Vector3f &point, 
				                                                          const double cosPhi0, 
																		  const double sinPhi0, 
																		  const double cosTheta, 
																		  const double sinThetaI) {
				const double st = s / sinThetaI;
				return Vector3f(point(0) + (cosPhi0 - (st * 0.5 * theRho) * sinPhi0) * st,
								point(1) + (sinPhi0 + (st * 0.5 * theRho) * cosPhi0) * st,
								point(2) + st * cosTheta * sinThetaI);
			}

			ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE Vector3f directionInDouble(const double theRho,
				                                                           const double s, 
																		   const double cosPhi0, 
																		   const double sinPhi0, 
																		   const double cosTheta, 
																		   const double sinThetaI) {
				const double dph = s * theRho / sinThetaI;
				return Vector3f(cosPhi0 - (sinPhi0 + 0.5 * dph * cosPhi0) * dph,
								sinPhi0 + (cosPhi0 - 0.5 * dph * sinPhi0) * dph,
								cosTheta * sinThetaI);
			}
            // may slightly improve performance (by avoiding creation of std::pair) if called frequently 
			template<PropagationDirection propDir>
			ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool solutionByDirection(
				const double dS1,
				const double dS2,
				double &path) {
					
				bool is_valid = false;

				if constexpr (propDir == anyDirection) {
					path = smallestPathLength(dS1, dS2);
					is_valid = std::isfinite(path) ? true : false;
				} else {
					const double propSign = (propDir == alongMomentum) ? 1 : -1;
					double s1(propSign * dS1);
					double s2(propSign * dS2);
					if (s1 > s2){
						//std::swap(s1, s2);
						double tmp = s1;
						s1 = s2;
						s2 = tmp;
					}
					if ((s1 < 0) & (s2 >= 0)) {
						is_valid = true;
						path = propSign * s2;
					} else if (s1 >= 0) {
						is_valid = true;
						path = propSign * s1;
					} else {
						path = 0;
					}
				}
				return is_valid;
			}
		} //namespace PlaneCrossing2Order
        
		template<PropagationDirection propDir>
		ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void helixArbitraryPlaneCrossing2Order(
			const Vector3f& point,
			const Vector3f& direction,
			const float curvature,
			const PlanePortable::Plane<Vector3f> plane,
			double& pathLength,
			bool& validPath,
			Vector3f& position,
			Vector3f& directionOut) {

            const double px = direction(0);
            const double py = direction(1); 
            const double pz = direction(2);
            const double pt2 = px * px + py * py;
            const double p2 = pt2 + pz * pz;
            const double pI = 1.0 / std::sqrt(p2);
            const double ptI = 1.0 / std::sqrt(pt2);

            const double theCosPhi0 = px * ptI;
            const double theSinPhi0 = py * ptI;
            const double theCosTheta = pz * pI;
            const double theSinThetaI = pt2 * ptI * pI;

			// Get normal vector of the plane
			const Vector3f normalToPlane = plane.normalVector();

			const double nPx = normalToPlane(0);
			const double nPy = normalToPlane(1);
			const double nPz = normalToPlane(2);
			const double cP = plane.localZ(point);

			// Coefficients of the 2nd order equation
			const double ceq1 = curvature * (nPx * theSinPhi0 - nPy * theCosPhi0);
			const double ceq2 = nPx * theCosPhi0 + nPy * theSinPhi0 + nPz * theCosTheta * theSinThetaI;
			const double ceq3 = cP;

			//
			// Check for degeneration to linear equation (zero
  			//   curvature, forward plane or direction perp. to plane)
  			//
			
			double dS1, dS2;
			if(std::abs(ceq1) > FLT_MIN) {
				const double deq1 = ceq2 * ceq2;
				const double deq2 = ceq1 * ceq3;
	//A.S. : another magic number..			
				if (std::abs(deq1) < FLT_MIN || std::abs(deq2 / deq1) > 1.e-6) {
					//
					// Standard solution for quadratic equations
					//
					const double deq = deq1 + 2 * deq2;
					if(deq < 0.)
						validPath = false;
					const double ceq = ceq2 + std::copysign(std::sqrt(deq), ceq2);
					dS1 = (ceq / ceq1) * theSinThetaI;
					dS2 = -2. * (ceq3 / ceq) * theSinThetaI;
				} else {
					const double ceq = (ceq2 / ceq1) * theSinThetaI;
					double deq = deq2 / deq1;
					deq *= (1 - 0.5 * deq);
					dS1 = -ceq * deq;
					dS2 = ceq * (2 + deq);
				}
			} else {
				//
				// Special case: linear equation
				//
				dS1 = dS2 = -(ceq3 / ceq2) * theSinThetaI;
			}
        	// Choose solution based on direction
			validPath = PlaneCrossing2Order::solutionByDirection<propDir>(dS1, dS2, pathLength);

			if (validPath) {
				// Calculate position and direction
				position = PlaneCrossing2Order::positionInDouble(curvature, pathLength, point, theCosPhi0, theSinPhi0, theCosTheta, theSinThetaI);
				directionOut = PlaneCrossing2Order::directionInDouble(curvature, pathLength, theCosPhi0, theSinPhi0, theCosTheta, theSinThetaI);
			}
		}

	} // namespace Propagators

} // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
