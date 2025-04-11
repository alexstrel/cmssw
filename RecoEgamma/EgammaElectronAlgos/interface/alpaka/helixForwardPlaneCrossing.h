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

using Vector3f = Eigen::Matrix<double, 3, 1>;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

    namespace Propagators {

        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE Vector3f positionInHelix(double s, const Vector3f& point, double rho,
                                                                     double cosPhi0, double sinPhi0, double cosTheta, double sinTheta,
                                                                     double& cachedS, double& cachedDPhi, double& cachedSDPhi, double& cachedCDPhi) 
        {
            if (s != cachedS) {
                cachedS = s;
                cachedDPhi = cachedS * rho * sinTheta;
                //vdt::fast_sincos(cachedDPhi, cachedSDPhi, cachedCDPhi);
                cachedSDPhi = std::sin(cachedDPhi);
                cachedCDPhi = std::cos(cachedDPhi);
            }

            if (std::abs(cachedDPhi) > 1.e-4) {
                double o = 1.0 / rho;
                return Vector3f(point(0) + (-sinPhi0 * (1.0 - cachedCDPhi) + cosPhi0 * cachedSDPhi) * o,
                                point(1) + (cosPhi0 * (1.0 - cachedCDPhi) + sinPhi0 * cachedSDPhi) * o,
                                point(2) + s * cosTheta);
            } else {
                double st = cachedS * sinTheta;
                return Vector3f(point(0) + (cosPhi0 - st * 0.5 * rho * sinPhi0) * st,
                                point(1) + (sinPhi0 + st * 0.5 * rho * cosPhi0) * st,
                                point(2) + st * cosTheta / sinTheta);
            }
        }

        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE Vector3f directionInHelix(double s, const Vector3f& point, double rho,
                                                                     double cosPhi0, double sinPhi0, double cosTheta, double sinTheta,
                                                                     double& cachedS, double& cachedDPhi, double& cachedSDPhi, double& cachedCDPhi) 
        {
            if (s != cachedS) {
                cachedS = s;
                cachedDPhi = cachedS * rho * sinTheta;
                //vdt::fast_sincos(cachedDPhi, cachedSDPhi, cachedCDPhi);
                cachedSDPhi = std::sin(cachedDPhi);
                cachedCDPhi = std::cos(cachedDPhi);
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

        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE std::pair<bool, double> calculatePathLength(const PlanePortable::Plane<Vector3f>& plane, double z0, double cosTheta, PropagationDirection propDir) {
            if (std::abs(cosTheta) < std::numeric_limits<float>::min())
                return std::make_pair(false, 0.0);

            Vector3f planePos = plane.pos();
            double dS = (planePos(2)- z0) / cosTheta;
            bool validSolution = !(((propDir == alongMomentum) && (dS < 0.)) || ((propDir == oppositeToMomentum) && (dS > 0.)) || !std::isfinite(dS));
            return std::make_pair(validSolution,dS);
        }

        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void helixForwardPlaneCrossing(
            const Vector3f& point,
            const Vector3f& direction,
            float curvature,
            PropagationDirection propDir,
			const PlanePortable::Plane<Vector3f> plane,
            double& pathLength,
            Vector3f& position,
            Vector3f& dir,
            bool& solExists) 
        {
            double cachedS = 0.; 
            double cachedDPhi = 0.;
            double cachedSDPhi = 0.;
            double cachedCDPhi = 1.;

            double px = direction(0);
            double py = direction(1);
            double pz = direction(2);
            double pt2 = px * px + py * py;
            double p2 = pt2 + pz * pz;
            double pI = 1.0 / std::sqrt(p2);
            double ptI = 1.0 / std::sqrt(pt2);
            double cosPhi0 = px * ptI;
            double sinPhi0 = py * ptI;
            double cosTheta = pz * pI;
            double sinTheta = pt2 * ptI * pI;

            // Calculate path length to the plane
            auto pathResult = calculatePathLength(plane, point(2), cosTheta, propDir);
            if (!pathResult.first) {
                solExists = false;
                pathLength = 0.0;
                return;
            }

            pathLength = pathResult.second;
            position = positionInHelix(pathLength, point, curvature, cosPhi0, sinPhi0, cosTheta, sinTheta, cachedS, cachedDPhi, cachedSDPhi, cachedCDPhi);
            dir = directionInHelix(pathLength, point, curvature, cosPhi0, sinPhi0, cosTheta, sinTheta, cachedS, cachedDPhi, cachedSDPhi, cachedCDPhi);
            solExists = true;
        }

    } // namespace Propagators

} // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif // RecoEgamma_EgammaElectronAlgos_interface_alpaka_helixForwardPlaneCrossing_h
