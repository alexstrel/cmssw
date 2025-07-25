/**
 Description: Function to propagate a helix from a point to a plane
*/

#ifndef RecoEgamma_EgammaElectronAlgos_interface_alpaka_helixForwardPlaneCrossing_h
#define RecoEgamma_EgammaElectronAlgos_interface_alpaka_helixForwardPlaneCrossing_h

#include <Eigen/Dense>
#include "DataFormats/EgammaReco/interface/alpaka/Plane.h"
#include <cmath>
#include <cfloat>
#include <vdt/vdtMath.h>

namespace ALPAKA_ACCELERATOR_NAMESPACE {

    namespace Propagators {

        constexpr Vec3d positionInHelix(const bool select,
					const double s, 
					const Vec3d& point, 
					const double rho,
                                        const double cosPhi0, 
					const double sinPhi0,
					const double cosTheta, 
					const double sinTheta,
                                        const double cachedS, 
					const double cachedSDPhi, 
					const double cachedCDPhi) {

            if (select) {
                const double o = 1.0 / rho;
                return Vec3d(point[0] + (-sinPhi0 * (1.0 - cachedCDPhi) + cosPhi0 * cachedSDPhi) * o,
                             point[1] + (cosPhi0 * (1.0 - cachedCDPhi) + sinPhi0 * cachedSDPhi) * o,
                             point[2] + s * cosTheta);
            } else {
                const double st = cachedS * sinTheta;
                return Vec3d(point[0] + (cosPhi0 - st * 0.5 * rho * sinPhi0) * st,
                             point[1] + (sinPhi0 + st * 0.5 * rho * cosPhi0) * st,
                             point[2] + st * cosTheta / sinTheta);
            }
        }

        constexpr Vec3d directionInHelix(const bool select,
					 const double s, 
					 const double rho,
                                         const double cosPhi0, 
					 const double sinPhi0, 
					 const double cosTheta,
					 const double sinTheta,
					 const double cachedSDPhi, 
					 const double cachedCDPhi) 
        {
            if (select) {
                return Vec3d(cosPhi0 * cachedCDPhi - sinPhi0 * cachedSDPhi,
                             sinPhi0 * cachedCDPhi + cosPhi0 * cachedSDPhi,
                             cosTheta / sinTheta);
            } else {
                const double dph = s * rho * sinTheta;
                return Vec3d(cosPhi0 - (sinPhi0 + 0.5 * cosPhi0 * dph) * dph,
                             sinPhi0 + (cosPhi0 - 0.5 * sinPhi0 * dph) * dph,
                             cosTheta / sinTheta);
            }
        }

	template<typename TAcc, PropagationDirection propDir>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void helixForwardPlaneCrossing( TAcc const &acc,
            								    const Vec3d& point,
            								    const Vec3d& direction,
            								    const float curvature,
									    const PlanePortable::Plane<Vec3d> plane,
            								    double& pathLength,
            							            Vec3d& position,
            								    Vec3d& dir,
            								    bool& solExists) {
            double cachedS = 0.; 
            double cachedDPhi = 0.;
            double cachedSDPhi = 0.;
            double cachedCDPhi = 1.;

            const double px = direction[0];
            const double py = direction[1];
            const double pz = direction[2];
            const double pt2 = px * px + py * py;
            const double p2 = pt2 + pz * pz;
            const double pI = 1.0 / alpaka::math::sqrt(acc, p2);
            const double ptI = 1.0 / alpaka::math::sqrt(acc, pt2);
            const double cosPhi0 = px * ptI;
            const double sinPhi0 = py * ptI;
            const double cosTheta = pz * pI;
            const double sinTheta = pt2 * ptI * pI;

            // Calculate path length to the plane
            const bool min_cosTheta_flag = ( alpak::math::abs(acc, cosTheta) < std::numeric_limits<float>::min() ); //TODO: replace with alpaka version
            
            pathLength = min_cosTheta_flag ? 0.0 : (plane.pos(2)- point[2]) / cosTheta;

	    const bool validSolution = !min_cosTheta_flag && !(((propDir == PropagationDirection::alongMomentum) && (pathLength < 0.)) || ((propDir == PropagationDirection::oppositeToMomentum) && (pathLength > 0.)) || !alpaka::math::isfinite(acc, pathLength));
	    //
	    if (!validSolution) {
              solExists = false;
              pathLength = 0.0;
              return;
            } 

            if (pathLength != cachedS) {
              cachedS = pathLength;
              cachedDPhi = cachedS * rho * sinTheta;
              cachedSDPhi = alpaka::math::sin(acc,cachedDPhi);
              cachedCDPhi = alpaka::math::cos(acc,cachedDPhi);
            }

	    const bool cachedDPhi_flag = alpaka::math::abs(acc, cachedDPhi) > 1.e-4;

            position = positionInHelix(cachedDPhi_flag, pathLength, point, curvature, cosPhi0, sinPhi0, cosTheta, sinTheta, cachedS, cachedSDPhi, cachedCDPhi);
            dir      = directionInHelix(cachedDPhi_flag, pathLength, curvature, cosPhi0, sinPhi0, cosTheta, sinTheta, cachedSDPhi, cachedCDPhi);
            
	    solExists = true;
        }

    } // namespace Propagators

} // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif // RecoEgamma_EgammaElectronAlgos_interface_alpaka_helixForwardPlaneCrossing_h
