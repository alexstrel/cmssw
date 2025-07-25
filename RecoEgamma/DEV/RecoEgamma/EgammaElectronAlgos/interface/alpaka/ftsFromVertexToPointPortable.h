#ifndef RecoEgamma_EgammaElectronAlgos_interface_alpaka_ftsFromVertexToPointPortable_h
#define RecoEgamma_EgammaElectronAlgos_interface_alpaka_ftsFromVertexToPointPortable_h

#include <cmath>

#include <HeterogeneousCore/AlpakaInterface/interface/VecArray.h>

using Vec3d = cms::alpakatools::VecArray<double, 3>;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

    namespace ftsFromVertexToPointPortable {

        // FreeTrajectoryState template structure
        template <typename Vec3 = Vec3d >
        class FreeTrajectoryState {
	  private:
            Vec3 momentum;  // 3D momentum vector
            Vec3 position;  // 3D position vector
            const int charge;     // Particle charge
          public:
            // Constructor
            constexpr FreeTrajectoryState(const Vec3& p, const Vec3& pos, const int q) 
                : momentum(p), position(pos), charge(q) {}

	    constexpr Vec3 momentum() const {return momentum;}
	    constexpr Vec3 position() const {return position;}
	    constexpr Vec3 charge() const {return charge;}
        };

        // Function to calculate the FreeTrajectoryState from vertex to point
        template <typename TAcc, typename Vec3 = Vec3d>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE FreeTrajectoryState<Vec3> ftsFromVertexToPoint(
	    TAcc const& acc,
            const Vec3& xmeas,    // Measured point
            const Vec3& xvert,    // Vertex point
            const float momentum,       // Magnitude of momentum
            const int charge,           // Charge of the particle
            const float BInTesla        // Magnetic field (in Tesla)
        ) {
	    using T = Vec3::value_t;	
            // Calculate the difference between measurement and vertex positions
            const Vec3 xdiff = cms::alpakatools::xmy(xmeas, xvert) ; //= xmeas - xvert;

            // Normalize xdiff and scale by momentum to get the momentum vector
	    const auto xdiff_norm2 = xdiff.norm2();

            // Normalize xdiff and scale by momentum to get the momentum vector:
	    const T scale = momentum / alpaka::math::sqrt(acc, xdiff_norm2);

            const Vec3 mom = cms::alpakatools::ax(scale, xdiff);

            // Transverse momentum (perpendicular to the z-axis)
            const T pt = mom.template partial_norm<TAcc, 2>(acc);
            const T pz = mom[2];

            const T pxOld = mom[0];
            const T pyOld = mom[1];

            // Calculate the curvature (assuming charge is either +1 or -1)
            const T curv = (BInTesla * 0.29979 * 0.01) / pt;

            // Calculate the sine and cosine of the rotation angle
            const T sa = 0.5 * xdiff.template partial_norm<TAcc, 2>(acc) * curv * float(charge);
            const T ca = alpaka::math::sqrt(acc,1. - sa * sa);

            // Rotate momentum vector in the xy-plane
            const T pxNew = ca * pxOld + sa * pyOld;
            const T pyNew = -sa * pxOld + ca * pyOld;
	    //
            const Vec3 pNew(pxNew,pyNew,pz); 

            return FreeTrajectoryState<Vec3>(pNew, xmeas, charge);
        }

    }  // namespace ftsFromVertexToPointPortable

} // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif // RecoEgamma_EgammaElectronAlgos_interface_alpaka_ftsFromVertexToPointPortable_h

