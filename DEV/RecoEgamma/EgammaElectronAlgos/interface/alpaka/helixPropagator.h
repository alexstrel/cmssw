
/**
 Description: Function to propagate from a point to a surface on the GPU
*/

#ifndef RecoEgamma_EgammaElectronAlgos_interface_alpaka_helixPropagator_h
#define RecoEgamma_EgammaElectronAlgos_interface_alpaka_helixPropagator_h

#include <iostream>
#include <cmath>
#include <Eigen/Dense>


using Vector3D = Eigen::Matrix<float, 3, 1>;
using Point3D  = Eigen::Matrix<float, 3, 1>;
enum Solution { bothSol, bestSol, onlyPos };

namespace ALPAKA_ACCELERATOR_NAMESPACE {

	namespace Propagators {

		template <typename T>
		ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T sqr(const T& t) {
			return t * t;
		};

		template <typename T>
		ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float perp2(const T& t) {
			return t(0)*t(0)+t(1)*t(1);
		};

		// For solving quad eq.
		struct QuadEquationSolver {

			double first;
			double second;
			bool hasSolution;

			ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE QuadEquationSolver(const double A, const double B, const double C) {
				const double D = B * B - 4 * A * C;
				if (D < 0)
					hasSolution = false;
				else {
					hasSolution = true;
					auto q = -0.5 * (B + std::copysign(std::sqrt(D), B));
					first = q / A;
					second = C / q;
				}
			}
		};
        
		template<int propDir>
		ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE Vector3D chooseSolution(
			const Vector3D& d1,
			const Vector3D& d2,
			//const Vector3D& startingPos,//has no effect...
			const Vector3D& startingDir,
			int& theActualDir,
			bool& theSolExists) {

			Vector3D theD;

			const auto momProj1 = startingDir(0) * d1(0) + startingDir(1) * d1(1);
			const auto momProj2 = startingDir(0) * d2(0) + startingDir(1) * d2(1);

			if constexpr (propDir == 0) {
				theSolExists = true;
				if (perp2(d1) < perp2(d2)) {
					theD = d1;
					theActualDir = (momProj1 > 0) ? 1 : -1;
				} else {
					theD = d2;
					theActualDir = (momProj2 > 0) ? 1 : -1;
				}
			} else {
				int propSign = (propDir == 1) ? 1 : -1;
				if (momProj1 * momProj2 < 0) {
					theSolExists = true;
					theD = (momProj1 * propSign > 0) ? d1 : d2;
					theActualDir = propSign;
				} else if (momProj1 * propSign > 0) {
					theSolExists = true;
					theD = (perp2(d1) < perp2(d2)) ? d1 : d2;
					theActualDir = propSign;
				} else {
					theSolExists = false;
				}
			}				
			return theD;
		};

        template<int propDir>
    	ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void helixBarrelCylinderCrossing(
                                    const Vector3D& startingPos, const Vector3D& startingDir, 
                                    double rho, double radius, bool& theSolExists,
                                    Vector3D& x, Vector3D& p, double& s){

			// Cylinder must be barrel (x,y = 0,0)
// is sol supposed to be updated?			
			Solution sol = bothSol;
			Vector3D thePos;
			Vector3D theDir;
			double theS;
			Vector3D thePos1;
			Vector3D thePos2;

			double R = radius;
			const double sraightLineCutoff = 1.e-7;

			if (fabs(rho) * R < sraightLineCutoff && fabs(rho) * startingPos.head(2).norm() < sraightLineCutoff) {
				printf("STRAIGHT LINE"); 
				theSolExists = false;
				return;
				// Should add implementation 
			}

			double R2cyl = R * R;
			double pt = sqrt(perp2(startingDir));

			// center of helix in global coords?
			double center_x = startingPos(0) - startingDir(1) / (pt * rho);
			double center_y = startingPos(1) + startingDir(0) / (pt * rho);
		
			double p2 = perp2(startingPos);
			bool solveForX;
			double B, C, E, F;

			if (fabs(center_x) > fabs(center_y)) {
				solveForX = false;
				E = (R2cyl - p2) / (2. * center_x);
				F = center_y / center_x;
				B = 2. * (startingPos(1) - F * startingPos(0) - E * F);
				C = 2. * E * startingPos(0) + E * E + p2 - R2cyl;
			} else {
				solveForX = true;
				E = (R2cyl - p2) / (2. * center_y);
				F = center_x / center_y;
				B = 2. * (startingPos(0) - F * startingPos(1) - E * F);
				C = 2. * E * startingPos(1) + E * E + p2 - R2cyl;
			}

			QuadEquationSolver eq(1 + F * F, B, C);
			if (!eq.hasSolution) {
				theSolExists = false;
				return;
			}

			double d1_x,d1_y, d2_x,d2_y;

			if (solveForX) {
				d1_x = eq.first;
				d1_y = E - F * eq.first;
				d2_x = eq.second;
				d2_y = E - F * eq.second;

			} else {
				d1_x = E - F * eq.first;
				d1_y = eq.first;
				d2_x = E - F * eq.second;
				d2_y = eq.second;
			}

			Vector3D theD;
			int theActualDir;
			Vector3D d1(d1_x,d1_y,0.0f);
			Vector3D d2(d2_x,d2_y,0.0f);

			theD = chooseSolution<propDir>(d1, d2, startingPos, startingDir,theActualDir,theSolExists);
			
			if (!theSolExists)
				return;

			float ipabs = 1.f / startingDir.norm();
			float sinTheta = float(pt) * ipabs;
			float cosTheta = startingDir(2) * ipabs;

			// -------

			auto dMag = theD.norm();
			float tmp = 0.5f * float(dMag * rho);
			if (std::abs(tmp) > 1.f)
				tmp = std::copysign(1.f, tmp);
			theS = theActualDir * 2.f * std::asin(tmp) / (float(rho) * sinTheta);
			thePos = Point3D(startingPos(0) + theD(0), startingPos(1) + theD(1), startingPos(2) + theS * cosTheta);


			if (sol == onlyPos)
    			return;

			if (theS < 0)
				tmp = -tmp;
			auto sinPhi = 2.f * tmp * sqrt(1.f - tmp * tmp);
			auto cosPhi = 1.f - 2.f * tmp * tmp;
			theDir = Vector3D(startingDir(0) * cosPhi - startingDir(1) * sinPhi,
								startingDir(0) * sinPhi + startingDir(1) * cosPhi,
								startingDir(2));

			if (sol != bothSol)
    			return;
			
			s = theS;
			x = thePos;
			p = theDir;

			if (sol != bothSol)
				return;

			const int theActualDir1 = propDir == 1 ? 1 : -1;
			const int theActualDir2 = propDir == 1 ? 1 : -1;

			const auto dMag1 = d1.norm();
			auto tmp1 = 0.5f * dMag1 * float(rho);
			if (std::abs(tmp1) > 1.f)
				tmp1 = std::copysign(1.f,tmp1);
			const auto theS1 = theActualDir1 * 2.f * std::asin(tmp1) / (rho * sinTheta);
			thePos1 = Vector3D(startingPos(0) + d1(0), startingPos(1) + d1(1), startingPos(2) + theS1 * cosTheta);

			const auto dMag2 = d2.norm();
			auto tmp2 = 0.5f * dMag2 * float(rho);
			if (std::abs(tmp2) > 1.f)
				tmp2 = std::copysign(1.f, tmp2);
			const auto theS2 = theActualDir2 * 2.f * std::asin(tmp2) / (float(rho) * sinTheta);
			thePos2 = Vector3D(startingPos(0) + d2(0), startingPos(1) + d2(1), startingPos(2) + theS2 * cosTheta);

		}

	} //Propagator_namespace

}// ALPAKA_ACCELERATOR_NAMESPACE

#endif
