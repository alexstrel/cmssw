/**
 Description: Function to propagate a helix from a point to a plane
*/

#ifndef RecoEgamma_EgammaElectronAlgos_interface_alpaka_helixForwardPlaneCrossing_h
#define RecoEgamma_EgammaElectronAlgos_interface_alpaka_helixForwardPlaneCrossing_h

#include <Eigen/Dense>
#include "DataFormats/EgammaReco/interface/Plane.h"
#include <cmath>
#include <cfloat>
#include <vdt/vdtMath.h>

using Vector3f = Eigen::Matrix<double, 3, 1>;//Vector3f -> Vector3d

namespace ALPAKA_ACCELERATOR_NAMESPACE {

    namespace Propagators {

        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE Vector3f positionInHelix(const double s, 
                                                                     const Vector3f& point, 
                                                                     const double rho,
                                                                     const double cosPhi0, 
                                                                     const double sinPhi0, 
                                                                     const double cosTheta, 
                                                                     const double sinTheta,
                                                                     double& cachedS, 
                                                                     double& cachedDPhi, 
                                                                     double& cachedSDPhi, 
                                                                     double& cachedCDPhi) 
        {
            if (s != cachedS) {
                cachedS = s;
                cachedDPhi = cachedS * rho * sinTheta;
                vdt::fast_sincos(cachedDPhi, cachedSDPhi, cachedCDPhi);
            }
// curious : so high threshold for dp var? would be good to avoid "magic" numbers... 
            if (std::abs(cachedDPhi) > 1.e-4) {
                const double o = 1.0 / rho;
                return Vector3f(point(0) + (-sinPhi0 * (1.0 - cachedCDPhi) + cosPhi0 * cachedSDPhi) * o,
                                point(1) + (cosPhi0 * (1.0 - cachedCDPhi) + sinPhi0 * cachedSDPhi) * o,
                                point(2) + s * cosTheta);
            } else {
                const double st = cachedS * sinTheta;
                return Vector3f(point(0) + (cosPhi0 - st * 0.5 * rho * sinPhi0) * st,
                                point(1) + (sinPhi0 + st * 0.5 * rho * cosPhi0) * st,
                                point(2) + st * cosTheta / sinTheta);
            }
        }

        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE Vector3f directionInHelix(const double s, 
                                                                      const Vector3f& point, 
                                                                      const double rho,
                                                                      const double cosPhi0, 
                                                                      const double sinPhi0, 
                                                                      const double cosTheta, 
                                                                      const double sinTheta,
                                                                      double& cachedS, 
                                                                      double& cachedDPhi, 
                                                                      double& cachedSDPhi, 
                                                                      double& cachedCDPhi) 
        {
            if (s != cachedS) {
                cachedS = s;
                cachedDPhi = cachedS * rho * sinTheta;
                vdt::fast_sincos(cachedDPhi, cachedSDPhi, cachedCDPhi);
            }

            if (std::abs(cachedDPhi) > 1.e-4) {
                return Vector3f(cosPhi0 * cachedCDPhi - sinPhi0 * cachedSDPhi,
                                sinPhi0 * cachedCDPhi + cosPhi0 * cachedSDPhi,
                                cosTheta / sinTheta);
            } else {
                double dph = s * rho * sinTheta;
                return Vector3f(cosPhi0 - (sinPhi0 + 0.5 * cosPhi0 * dph) * dph,
                                sinPhi0 + (cosPhi0 - 0.5 * sinPhi0 * dph) * dph,
                                cosTheta / sinTheta);
            }
        }

        template<PropagationDirection propDir>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool calculatePathLength(const PlanePortable::Plane<Vector3f>& plane, 
                                                                     const double z0, 
                                                                     const double cosTheta, 
                                                                     double& pathLength) {
            if (std::abs(cosTheta) < std::numeric_limits<float>::min()) {
                pathLength = 0.0;
                return false;
            }
            const double pos_z = plane.pos()(2);
            pathLength = (pos_z- z0) / cosTheta;
            return !(((propDir == alongMomentum) && (dS < 0.)) || ((propDir == oppositeToMomentum) && (dS > 0.)) || !std::isfinite(dS));
        }

        template<PropagationDirection propDir>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void helixForwardPlaneCrossing(
            const Vector3f& point,
            const Vector3f& direction,
            float curvature,
			const PlanePortable::Plane<Vector3f> &plane,//REF
            double& pathLength,
            Vector3f& position,
            Vector3f& dir,
            bool& solExists) 
        {
            double cachedS     = 0.; 
            double cachedDPhi  = 0.;
            double cachedSDPhi = 0.;
            double cachedCDPhi = 1.;

            const double px = direction(0);
            const double py = direction(1);
            const double pz = direction(2);
            const double pt2 = px * px + py * py;
            const double p2 = pt2 + pz * pz;
            const double pI = 1.0 / std::sqrt(p2);
            const double ptI = 1.0 / std::sqrt(pt2);
            const double cosPhi0 = px * ptI;
            const double sinPhi0 = py * ptI;
            const double cosTheta = pz * pI;
            const double sinTheta = pt2 * ptI * pI;

            // Calculate path length to the plane
            solExists = calculatePathLength<propDir>(plane, point(2), cosTheta, pathLength);
            if (solExists == false) {
                pathLength = 0.0;
                return;
            }

            position = positionInHelix(pathLength, point, curvature, cosPhi0, sinPhi0, cosTheta, sinTheta, cachedS, cachedDPhi, cachedSDPhi, cachedCDPhi);
            dir = directionInHelix(pathLength, point, curvature, cosPhi0, sinPhi0, cosTheta, sinTheta, cachedS, cachedDPhi, cachedSDPhi, cachedCDPhi);
            solExists = true;
        }

    } // namespace Propagators

} // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif // RecoEgamma_EgammaElectronAlgos_interface_alpaka_helixForwardPlaneCrossing_h
