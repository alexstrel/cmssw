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

#include "DataFormats/EgammaReco/interface/Plane.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/alpaka/helixArbitraryPlaneCrossing2Order.h"

// A.S.: Vector3f -> Vector3d
using Vector3f = Eigen::Matrix<double, 3, 1>;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

    namespace Propagators {
// A.S.: constexpr is preferable
        constexpr float theNumericalPrecision = 5.e-7f;
        constexpr float theMaxDistToPlane     = 1.e-4f;

        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE Vector3f positionInDouble(const double s, 
                                                                      const Vector3f& point, 
                                                                      const double rho, 
                                                                      const double cosPhi0, 
                                                                      const double sinPhi0, 
                                                                      const double cosTheta, 
                                                                      const double sinTheta, 
                                                                      const double sinThetaI,
                                                                      double& theCachedS, 
                                                                      double& theCachedDPhi, 
                                                                      double& theCachedSDPhi, 
                                                                      double& theCachedCDPhi) 
        {
            if(s != theCachedS) {
                theCachedS = s;
                theCachedDPhi = theCachedS * rho * sinTheta;
                vdt::fast_sincos(theCachedDPhi, theCachedSDPhi, theCachedCDPhi);
            }
//A.S.: "magic" number
            if (std::abs(theCachedDPhi) > 1.e-4) {
                // "standard" helix formula
                const double o = 1. / rho;
                return Vector3f(point(0) + (-sinPhi0 * (1.0 - theCachedCDPhi) + cosPhi0 * theCachedSDPhi) * o,
                                point(1) + (cosPhi0 * (1.0 - theCachedCDPhi) + sinPhi0 * theCachedSDPhi) * o,
                                point(2) + s * cosTheta);
            }        
            else {
                const double st = theCachedS / sinThetaI;
                return Vector3f(point(0)  + (cosPhi0 - (st * 0.5 * rho) * sinPhi0) * st,
                                point(1) + (sinPhi0 + (st * 0.5 * rho) * cosPhi0) * st,
                                point(2) + st * cosTheta * sinThetaI);
            }
        }


        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE Vector3f directionInDouble(const double s, 
                                                                       const Vector3f& point, 
                                                                       const double rho, 
                                                                       const double cosPhi0, 
                                                                       const double sinPhi0, 
                                                                       const double cosTheta, 
                                                                       const double sinTheta, 
                                                                       const double sinThetaI,
                                                                       double& theCachedS, 
                                                                       double& theCachedDPhi, 
                                                                       double& theCachedSDPhi, 
                                                                       double& theCachedCDPhi)
        {
            //
            // Calculate delta phi (if not already available)
            //
            if(s != theCachedS) { // very very unlikely!
                theCachedS = s;
                theCachedDPhi = theCachedS * rho * sinTheta;
                vdt::fast_sincos(theCachedDPhi, theCachedSDPhi, theCachedCDPhi);
            }
//A.S.: "magic" number
            if (std::abs(theCachedDPhi) > 1.e-4) {
                // full helix formula
                return Vector3f(cosPhi0 * theCachedCDPhi - sinPhi0 * theCachedSDPhi,
                                        sinPhi0 * theCachedCDPhi + cosPhi0 * theCachedSDPhi,
                                        cosTheta / sinTheta);
            } else {
                // 2nd order
                const double dph = s * rho / sinThetaI;
                return Vector3f(cosPhi0 - (sinPhi0 + 0.5 * dph * cosPhi0) * dph,
                                sinPhi0 + (cosPhi0 - 0.5 * dph * sinPhi0) * dph,
                                cosTheta * sinThetaI);
            }

        }

        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool notAtSurface(const PlanePortable::Plane<Vector3f>& plane, const Vector3f& point, const float maxDist) {
            const float dz = plane.localZ(point);
            return std::abs(dz) > maxDist;
        }

        template<PropagationDirection propDir>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool helixArbitraryPlaneCrossing(        
            const Vector3f& point,
            const Vector3f& direction,
            float curvature,
            const PlanePortable::Plane<Vector3f> plane,
            double& pathLength,
            Vector3f& position,
            Vector3f& dir) 
        { 
            bool solExists = false;
            //
            const float maxNumDz    = theNumericalPrecision * plane.pos().norm();
            const float safeMaxDist = (theMaxDistToPlane > maxNumDz) ? theMaxDistToPlane : maxNumDz;
            
            const float dz = plane.localZ(point);
            if (std::abs(dz) < safeMaxDist) {
                pathLength = 0.0;
                position = point;
                dir = direction;
                solExists = true;   
                return solExists;//true
            }            
            //
            // Prepare internal value of the propagation direction and position / direction vectors for iteration
            //
            // Use existing 2nd order object at first pass
            double pathLength2O = 0;
	        bool validPath2O = false;
	        //
	        Vector3f position2O = {0,0,0};
	        Vector3f directionOut2O = {0,0,0};
	    
	        helixArbitraryPlaneCrossing2Order<propDir>(point,direction,curvature,plane,pathLength2O,validPath2O,position2O,directionOut2O);
			
            if (!validPath2O) {
                pathLength = pathLength2O;
                return solExists;//false
            }
                       
            auto currentPropDir = propDir;
            auto newDir = pathLength2O >= 0 ? alongMomentum : oppositeToMomentum;
            
            if (currentPropDir == anyDirection) {
                currentPropDir = newDir;
            } else {
                if (newDir != currentPropDir) return solExists;//false
            }
            //
            // Prepare iterations: count and total pathlength
            //
            constexpr int maxIterations = 20;
            
            pathLength = pathLength2O;
            auto iteration = maxIterations;
            
            double pathLengthX = pathLength2O;

            double theCachedS     = 0.; 
            double theCachedDPhi  = 0.;
            double theCachedSDPhi = 0.;
            double theCachedCDPhi = 1.;
            //
            const double pt2 = direction(0) * direction(0) + direction(1) * direction(1);
            const double p2 = pt2 + direction(2) * direction(2);
            const double pI = 1.0 / std::sqrt(p2);
            const double ptI = 1.0 / std::sqrt(pt2);
            const double cosPhi0 = direction(0) * ptI;
            const double sinPhi0 = direction(2) * ptI;
            const double cosTheta = direction(2) * pI;
            const double sinTheta = pt2 * ptI * pI;
            const double sinThetaI = p2 * pI * ptI;  //  (1/(pt/p)) = p/pt = p*ptI and p = p2/p = p2*pI 
             
            do {
                const Vector3f xnew = positionInDouble(pathLengthX, point, curvature, cosPhi0, sinPhi0, cosTheta, sinTheta, sinThetaI,
                                                 theCachedS,theCachedDPhi,theCachedSDPhi,theCachedCDPhi);
                                                 
                solExists = notAtSurface(plane, xnew, safeMaxDist); 
                
                if (solExists) {
                    position = xnew;
                    dir = directionInDouble(pathLength, point, curvature, cosPhi0, sinPhi0, cosTheta, sinTheta, sinThetaI,
                                      theCachedS,theCachedDPhi,theCachedSDPhi,theCachedCDPhi);                
                    //return solExists;//direct return from the loop
                    break;
                }                                                
            
                const Vector3f pnew = directionInDouble(pathLength, point, curvature, cosPhi0, sinPhi0, cosTheta, sinTheta, sinThetaI,
                                                theCachedS,theCachedDPhi,theCachedSDPhi,theCachedCDPhi);

                double tmpPathLength = 0.;
                bool tmpValidPath = false;
                Vector3f tmpPosition = {0,0,0};
                Vector3f tmpDirectionOut = {0,0,0};

                // Originally it passes the theSinTheta
                helixArbitraryPlaneCrossing2Order<anyDirection>(xnew,pnew,curvature,plane,tmpPathLength,tmpValidPath,tmpPosition,tmpDirectionOut);
                /////////////////////////
                if (!tmpValidPath) break;

                pathLength += tmpPathLength;

                newDir = pathLength >= 0 ? alongMomentum : oppositeToMomentum;
                if (currentPropDir == anyDirection) {
                    currentPropDir = newDir;
                } else {
                    if (newDir != currentPropDir) break;
                }
                pathLengthX = pathLength;
                
            } while(--iteration > 0);
                                    
            return solExists;//false or true                                    
        }

    } // namespace Propagators

} // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
