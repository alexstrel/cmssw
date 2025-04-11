/**
 Description: Function to propagate from a point to a plane on the GPU
*/

#ifndef RecoEgamma_EgammaElectronAlgos_interface_alpaka_helixArbitraryPlaneCrossing_h
#define RecoEgamma_EgammaElectronAlgos_interface_alpaka_helixArbitraryPlaneCrossing_h


#include <Eigen/Dense>
#include <alpaka/alpaka.hpp>
#include <cmath>
#include <utility>
#include <iostream>
#include <atomic>
#include <vdt/vdtMath.h> 

#include "DataFormats/EgammaReco/interface/alpaka/Plane.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/alpaka/helixArbitraryPlaneCrossing2Order.h"

using Vector3f = Eigen::Matrix<double, 3, 1>;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

    namespace Propagators {

        const float theNumericalPrecision = 5.e-7f;
        const float theMaxDistToPlane = 1.e-4f;

        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE Vector3f positionInDouble(double s, const Vector3f& point, double rho, 
                                                                     double cosPhi0,  double sinPhi0, double cosTheta, double sinTheta, double sinThetaI,
                                                                     double& theCachedS, double& theCachedDPhi, double& theCachedSDPhi, double& theCachedCDPhi) 
        {
            if(s != theCachedS) {
                theCachedS = s;
                theCachedDPhi = theCachedS * rho * sinTheta;
                //vdt::fast_sincos(theCachedDPhi, theCachedSDPhi, theCachedCDPhi);
                theCachedSDPhi = std::sin(theCachedDPhi);
                theCachedCDPhi = std::cos(theCachedDPhi);
            }

            if (std::abs(theCachedDPhi) > 1.e-4) {
                // "standard" helix formula
                double o = 1. / rho;
                return Vector3f(point(0) + (-sinPhi0 * (1.0 - theCachedCDPhi) + cosPhi0 * theCachedSDPhi) * o,
                                point(1) + (cosPhi0 * (1.0 - theCachedCDPhi) + sinPhi0 * theCachedSDPhi) * o,
                                point(2) + s * cosTheta);
            }        
            else {
                double st = theCachedS / sinThetaI;
                return Vector3f(point(0)  + (cosPhi0 - (st * 0.5 * rho) * sinPhi0) * st,
                                point(1) + (sinPhi0 + (st * 0.5 * rho) * cosPhi0) * st,
                                point(2) + st * cosTheta * sinThetaI);
            }
        }


        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE Vector3f directionInDouble(double s, const Vector3f& point, double rho, 
                                                                       double cosPhi0,  double sinPhi0, double cosTheta, double sinTheta, double sinThetaI,
                                                                       double& theCachedS, double& theCachedDPhi, double& theCachedSDPhi, double& theCachedCDPhi)
            {
            //
            // Calculate delta phi (if not already available)
            //
            if(s != theCachedS) { // very very unlikely!
                theCachedS = s;
                theCachedDPhi = theCachedS * rho * sinTheta;
                //vdt::fast_sincos(theCachedDPhi, theCachedSDPhi, theCachedCDPhi);
                theCachedSDPhi = std::sin(theCachedDPhi);
                theCachedCDPhi = std::cos(theCachedDPhi);
            }

            if (std::abs(theCachedDPhi) > 1.e-4) {
                // full helix formula
                return Vector3f(cosPhi0 * theCachedCDPhi - sinPhi0 * theCachedSDPhi,
                                        sinPhi0 * theCachedCDPhi + cosPhi0 * theCachedSDPhi,
                                        cosTheta / sinTheta);
            } else {
                // 2nd order
                double dph = s * rho / sinThetaI;
                return Vector3f(cosPhi0 - (sinPhi0 + 0.5 * dph * cosPhi0) * dph,
                                sinPhi0 + (cosPhi0 - 0.5 * dph * sinPhi0) * dph,
                                cosTheta * sinThetaI);
            }

        }

        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool notAtSurface(const PlanePortable::Plane<Vector3f>& plane, const Vector3f& point, float maxDist) {
            float dz = plane.localZ(point);
            return std::abs(dz) > maxDist;
        }

        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void helixArbitraryPlaneCrossing(
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
            double theCachedS = 0.; 
            double theCachedDPhi = 0.;
            double theCachedSDPhi = 0.;
            double theCachedCDPhi = 1.;

            constexpr int maxIterations = 20;
            float maxNumDz = theNumericalPrecision * plane.pos().norm();
            float safeMaxDist = (theMaxDistToPlane > maxNumDz) ? theMaxDistToPlane : maxNumDz;

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
            double sinThetaI = p2 * pI * ptI;  //  (1/(pt/p)) = p/pt = p*ptI and p = p2/p = p2*pI

            //
            // Prepare internal value of the propagation direction and position / direction vectors for iteration
            //

            float dz = plane.localZ(point);
            if (std::abs(dz) < safeMaxDist) {
                pathLength = 0.0;
                solExists = true;
                position = point;
                dir = direction;
                return;
            }

            // Use existing 2nd order object at first pass
            double pathLength2O = 0;
			bool validPath2O = false;
			Vector3f position2O = {0,0,0};
			Vector3f directionOut2O = {0,0,0};
		    helixArbitraryPlaneCrossing2Order(point,direction,curvature,propDir,plane,pathLength2O,validPath2O,position2O,directionOut2O);
			
            if (!validPath2O) {
                solExists = false;
                pathLength = pathLength2O;
                return;
            }

            Vector3f xnew = positionInDouble(pathLength2O, point, curvature, cosPhi0, sinPhi0, cosTheta, sinTheta, sinThetaI,
                                            theCachedS,theCachedDPhi,theCachedSDPhi,theCachedCDPhi);

            auto currentPropDir = propDir;
            auto newDir = pathLength2O >= 0 ? alongMomentum : oppositeToMomentum;
            if (currentPropDir == anyDirection) {
                currentPropDir = newDir;
            } else {
                if (newDir != currentPropDir) {
                    solExists = false;
                    return;
                }
            }

            //
            // Prepare iterations: count and total pathlength
            //

            pathLength = pathLength2O;
            auto iteration = maxIterations;
            while (notAtSurface(plane, xnew, safeMaxDist)) 
            {
                if (--iteration == 0) {
                    //LogDebug("HelixArbitraryPlaneCrossing") << "pathLength : no convergence";
                    solExists = false;
                    return;
                }

            
                Vector3f pnew = directionInDouble(pathLength, point, curvature, cosPhi0, sinPhi0, cosTheta, sinTheta, sinThetaI,
                                                theCachedS,theCachedDPhi,theCachedSDPhi,theCachedCDPhi);

                double tmpPathLength = 0.;
                bool tmpValidPath = false;
                Vector3f tmpPosition = {0,0,0};
                Vector3f tmpDirectionOut = {0,0,0};

                // Originally it passes the theSinTheta
                helixArbitraryPlaneCrossing2Order(xnew,pnew,curvature,anyDirection,plane,tmpPathLength,tmpValidPath,tmpPosition,tmpDirectionOut);
                /////////////////////////

                if (!tmpValidPath) {
                    solExists = false;
                    return;
                }

                pathLength += tmpPathLength;

                newDir = pathLength >= 0 ? alongMomentum : oppositeToMomentum;
                if (currentPropDir == anyDirection) {
                    currentPropDir = newDir;
                } else {
                    if (newDir != currentPropDir) {
                        solExists = false;
                        return;
                    }
                }
                xnew = positionInDouble(pathLength, point, curvature, cosPhi0, sinPhi0, cosTheta, sinTheta, sinThetaI,
                                            theCachedS,theCachedDPhi,theCachedSDPhi,theCachedCDPhi);
            }

            solExists = true;
            position = xnew;
            dir = directionInDouble(pathLength, point, curvature, cosPhi0, sinPhi0, cosTheta, sinTheta, sinThetaI,
                                    theCachedS,theCachedDPhi,theCachedSDPhi,theCachedCDPhi);
        }

    } // namespace Propagators

} // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
