/**
 Description: Function to propagate from a point to a plane on the GPU
*/

#ifndef RecoEgamma_EgammaElectronAlgos_interface_alpaka_helixBarrelPlaneCrossing_h
#define RecoEgamma_EgammaElectronAlgos_interface_alpaka_helixBarrelPlaneCrossing_h

#include <iostream>
#include <cmath>
#include <Eigen/Dense>

#include "DataFormats/EgammaReco/interface/Plane.h"

using Vector3f = Eigen::Matrix<double, 3, 1>;


namespace ALPAKA_ACCELERATOR_NAMESPACE {

	namespace Propagators {

		enum PropagationDirection { alongMomentum, oppositeToMomentum, anyDirection };
// same as in helixPropagator.h...
		struct QuadEquationSolver {

			double first;
			double second;
			bool hasSolution;

			ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE QuadEquationSolver(double A, double B, double C) {
				double D = B * B - 4 * A * C;
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

		ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE Vector3f chooseSolution(
			const Vector3f& d1,
			const Vector3f& d2,
			const Vector3f& startingPos,// has no effect
			const Vector3f& startingDir,
			PropagationDirection propDir,//const
			int& theActualDir,
			bool& theSolExists) {

			Vector3f theD;

			const auto momProj1 = startingDir(0) * d1(0) + startingDir(1) * d1(1);
			const auto momProj2 = startingDir(0) * d2(0) + startingDir(1) * d2(1);

			if (propDir == anyDirection) {
				theSolExists = true;
				if (d1.squaredNorm() < d2.squaredNorm()) {
					theD = d1;
					theActualDir = (momProj1 > 0) ? 1 : -1;
				} else {
					theD = d2;
					theActualDir = (momProj2 > 0) ? 1 : -1;
				}
			} else {
				double propSign = (propDir == alongMomentum) ? 1 : -1;
				if (momProj1 * momProj2 < 0) {
					theSolExists = true;
					theD = (momProj1 * propSign > 0) ? d1 : d2;
					theActualDir = propSign;
				} else if (momProj1 * propSign > 0) {
					theSolExists = true;
					theD = (d1.squaredNorm() < d2.squaredNorm()) ? d1 : d2;
					theActualDir = propSign;
				} else {
					theSolExists = false;
				}
			}

			return theD;
		}

		ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void helixBarrelPlaneCrossing(
		const Vector3f& startingPos,
		const Vector3f& startingDir,
		double rho,
		PropagationDirection propDir,
		Vector3f& surfPosition,
		Vector3f& surfRotation,            
		bool& theSolExists,
		Vector3f& position,
		Vector3f& direction,
		double& s) 
		{

		constexpr double sraightLineCutoff = 1.e-7;

		const double pt = startingDir.head(2).norm();
		const double ipabs = 1. / startingDir.norm();

		const double sinTheta = pt * ipabs;
		const double cosTheta = startingDir(2) * ipabs;

		// === TO FIX SHOULD ADD STRAIGHT LINE CASE === //
		//bool useStraightLine;
		if (fabs(rho) < sraightLineCutoff && fabs(rho) * startingPos.head(2).norm() < sraightLineCutoff) {
			//useStraightLine = true;
			// Add implementation
			//StraightLinePlaneCrossing slc(startingPos, startingDir, propDir);
			//auto pl = slc.pathLength(plane);
			//if (pl.first) {
			//	theSolExists = true;
			//	s = pl.second;
			//	position = slc.position(s);
			//	direction = startingDir;
			//} else {
				theSolExists = false;
			//}
			//return; // all needed data members have been set
		}

		const double o = 1. / (pt * rho);//div by zero check?
		const double theXCenter = startingPos(0) - startingDir(1) * o;
		const double theYCenter = startingPos(1) + startingDir(0) * o;

		// This is default when there curvature is non zero

		const PlanePortable::Plane<Vector3f> plane{surfPosition,surfRotation};

		const Vector3f n = plane.normalVector();
		const double distToPlane = -plane.localZ(startingPos);
		const double nx = n(0);
		const double ny = n(1);
		const double distCx = startingPos(0) - theXCenter;
		const double distCy = startingPos(1) - theYCenter;

		double nfac, dfac;
		double A, B, C;
		bool solveForX;

		if (fabs(nx) > fabs(ny)) {
			solveForX = false;
			nfac = ny / nx;
			dfac = distToPlane / nx;
			B = distCy - nfac * distCx; // only part of B
			C = (2. * distCx + dfac) * dfac;
		} else {
			solveForX = true;
			nfac = nx / ny;
			dfac = distToPlane / ny;
			B = distCx - nfac * distCy; // only part of B
			C = (2. * distCy + dfac) * dfac;
		}

		B -= nfac * dfac;
		B *= 2; // the rest of B
		A = 1. + nfac * nfac;

		QuadEquationSolver eq(A, B, C);
		if (!eq.hasSolution) {
			theSolExists = false;
			return;
		}

		Vector3f d1, d2;
		if (solveForX) {
			d1 = Vector3f(eq.first, dfac - nfac * eq.first, 0.0);
			d2 = Vector3f(eq.second, dfac - nfac * eq.second, 0.0);
		} else {
			d1 = Vector3f(dfac - nfac * eq.first, eq.first, 0.0);
			d2 = Vector3f(dfac - nfac * eq.second, eq.second, 0.0);
		}

		Vector3f theD;
		int theActualDir;
		theD = chooseSolution(d1, d2, startingPos, startingDir, propDir, theActualDir, theSolExists);
		if (!theSolExists)
			return;

		const auto dMag = theD.norm();

		double sinAlpha = 0.5 * dMag * rho;
		if (std::abs(sinAlpha) > 1.0)
			 sinAlpha = std::copysign(1.0, sinAlpha);
		// Path length
		s = theActualDir * 2.0 * std::asin(sinAlpha) / (rho * sinTheta);

		// Position
		position = Vector3f(startingPos(0) + theD(0), startingPos(1) + theD(1), startingPos(2) + s * cosTheta);

		// Direction
    	const double tmp    = 0.5 * dMag * rho * (s < 0 ? -1.0 : 1.0);
	    const double tmp2   = tmp*tmp;

		const double sinPhi = (tmp2 < 1 ? 0. : 2.0 * tmp * sqrt(1-tmp2);
		const double cosPhi = 1.0 - 2.0 * tmp2;

		direction = Vector3f(startingDir(0) * cosPhi - startingDir(1) * sinPhi,
							  startingDir(0) * sinPhi + startingDir(1) * cosPhi,
							  startingDir(2));
		}

	} // namespace Propagators

} // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
