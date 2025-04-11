/**
 Description: Function to propagate from a helix to a plane on the GPU
*/

#ifndef RecoEgamma_EgammaElectronAlgos_interface_alpaka_helixArbitraryPlaneCrossing2Order_h
#define RecoEgamma_EgammaElectronAlgos_interface_alpaka_helixArbitraryPlaneCrossing2Order_h

#include <Eigen/Dense>
#include "DataFormats/EgammaReco/interface/alpaka/Plane.h"
#include <cmath>
#include <cfloat>

using Vector3f = Eigen::Matrix<double, 3, 1>;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

	namespace Propagators {

		namespace PlaneCrossing2Order {

			ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE double smallestPathLength(const double firstPathLength, const double secondPathLength) {
				return fabs(firstPathLength) < fabs(secondPathLength) ? firstPathLength : secondPathLength;
			}

			ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE Vector3f positionInDouble(double theRho,
				double s, double x0, double y0, double z0,
				double cosPhi0, double sinPhi0, double cosTheta, double sinThetaI) {
				double st = s / sinThetaI;
				return Vector3f(x0 + (cosPhi0 - (st * 0.5 * theRho) * sinPhi0) * st,
								y0 + (sinPhi0 + (st * 0.5 * theRho) * cosPhi0) * st,
								z0 + st * cosTheta * sinThetaI);
			}

			ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE Vector3f directionInDouble(double theRho,
				double s, double cosPhi0, double sinPhi0, double cosTheta, double sinThetaI) {
				double dph = s * theRho / sinThetaI;
				return Vector3f(cosPhi0 - (sinPhi0 + 0.5 * dph * cosPhi0) * dph,
								sinPhi0 + (cosPhi0 - 0.5 * dph * sinPhi0) * dph,
								cosTheta * sinThetaI);
			}

			ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE std::pair<bool, double> solutionByDirection(
				const double dS1,
				const double dS2,
				PropagationDirection propDir) {

				bool valid = false;
				double path = 0;
				if (propDir == anyDirection) {
					valid = true;
					path = smallestPathLength(dS1, dS2);
				} else {
					double propSign = (propDir == alongMomentum) ? 1 : -1;
					double s1(propSign * dS1);
					double s2(propSign * dS2);
					if (s1 > s2){
						//std::swap(s1, s2);
						double tmp = s1;
						s1 = s2;
						s2 = tmp;
					}
					if ((s1 < 0) & (s2 >= 0)) {
						valid = true;
						path = propSign * s2;
					} else if (s1 >= 0) {
						valid = true;
						path = propSign * s1;
					}
				}

				if(!(std::isfinite(path)))
					valid = false;

				return std::pair<bool, double>(valid, path);
			}
		};

		ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void helixArbitraryPlaneCrossing2Order(
			const Vector3f& point,
			const Vector3f& direction,
			const float curvature,
			PropagationDirection propDir,
			const PlanePortable::Plane<Vector3f> plane,
			double& pathLength,
			bool& validPath,
			Vector3f& position,
			Vector3f& directionOut) {

			double theX0 = point(0);
			double theY0 = point(1);
			double theZ0 = point(2);
            double px = direction(0);
            double py = direction(1); 
            double pz = direction(2);
            double pt2 = px * px + py * py;
            double p2 = pt2 + pz * pz;
            double pI = 1.0 / std::sqrt(p2);
            double ptI = 1.0 / std::sqrt(pt2);
            double theCosPhi0 = px * ptI;
            double theSinPhi0 = py * ptI;
            double theCosTheta = pz * pI;
            double theSinThetaI = pt2 * ptI * pI;

			// Get normal vector of the plane
			Vector3f normalToPlane = plane.normalVector();
			double nPx = normalToPlane(0);
			double nPy = normalToPlane(1);
			double nPz = normalToPlane(2);
			double cP = plane.localZ(point);

			// Coefficients of the 2nd order equation
			double ceq1 = curvature * (nPx * theSinPhi0 - nPy * theCosPhi0);
			double ceq2 = nPx * theCosPhi0 + nPy * theSinPhi0 + nPz * theCosTheta * theSinThetaI;
			double ceq3 = cP;

			//
			// Check for degeneration to linear equation (zero
  			//   curvature, forward plane or direction perp. to plane)
  			//
			
			double dS1, dS2;
			if(std::abs(ceq1) > FLT_MIN) {
				double deq1 = ceq2 * ceq2;
				double deq2 = ceq1 * ceq3;
				if (std::abs(deq1) < FLT_MIN || std::abs(deq2 / deq1) > 1.e-6) {
					//
					// Standard solution for quadratic equations
					//
					double deq = deq1 + 2 * deq2;
					if(deq < 0.)
						validPath = false;
					double ceq = ceq2 + std::copysign(std::sqrt(deq), ceq2);
					dS1 = (ceq / ceq1) * theSinThetaI;
					dS2 = -2. * (ceq3 / ceq) * theSinThetaI;
				} else {
					double ceq = (ceq2 / ceq1) * theSinThetaI;
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
			std::pair<bool, double> solution = PlaneCrossing2Order::solutionByDirection(dS1, dS2, propDir);
			validPath = solution.first;
			pathLength = solution.second;

			if (validPath) {
				// Calculate position and direction
				position = PlaneCrossing2Order::positionInDouble(curvature, pathLength, theX0, theY0, theZ0, theCosPhi0, theSinPhi0, theCosTheta, theSinThetaI);
				directionOut = PlaneCrossing2Order::directionInDouble(curvature, pathLength, theCosPhi0, theSinPhi0, theCosTheta, theSinThetaI);
			}
		}

	} // namespace Propagators

} // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
