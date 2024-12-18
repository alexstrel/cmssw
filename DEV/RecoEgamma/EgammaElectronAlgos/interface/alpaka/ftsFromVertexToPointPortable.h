#ifndef RecoEgamma_EgammaElectronAlgos_interface_alpaka_ftsFromVertexToPointPortable_h
#define RecoEgamma_EgammaElectronAlgos_interface_alpaka_ftsFromVertexToPointPortable_h

#include <cmath>
#include <Eigen/Core>

namespace ALPAKA_ACCELERATOR_NAMESPACE {

    namespace ftsFromVertexToPointPortable {

        // FreeTrajectoryState template structure
        template <typename Vec3>
        struct FreeTrajectoryState {
            Vec3 momentum;  // 3D momentum vector
            Vec3 position;  // 3D position vector
            int charge;     // Particle charge

            // Constructor
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE FreeTrajectoryState(const Vec3 &p, const Vec3 &pos, const int q) //A.S. : avoid unnecessary copies
                : momentum(p), position(pos), charge(q) {}
        };

        // Function to calculate the FreeTrajectoryState from vertex to point
        template <typename Vec3>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE FreeTrajectoryState<Vec3> ftsFromVertexToPoint(
            const Vec3& xmeas,    // Measured point
            const Vec3& xvert,    // Vertex point
            const float momentum,       // Magnitude of momentum
            const int charge,           // Charge of the particle
            const float BInTesla        // Magnetic field (in Tesla)
        ) {
            // Calculate the difference between measurement and vertex positions
            const Vec3 xdiff = xmeas - xvert;

            // Normalize xdiff and scale by momentum to get the momentum vector
            const Vec3 mom = momentum * (xdiff / sqrt(xdiff(0) * xdiff(0) + xdiff(1) * xdiff(1) + xdiff(2) * xdiff(2)));

            // Transverse momentum (perpendicular to the z-axis)
            const float pt = sqrt(mom(0) * mom(0) + mom(1) * mom(1));
            const float pz = mom(2);

            const float pxOld = mom(0);
            const float pyOld = mom(1);

            // Calculate the curvature (assuming charge is either +1 or -1)
            const float curv = (BInTesla * 0.29979f * 0.01f) / pt;

            // Calculate the sine and cosine of the rotation angle
            const float sa = 0.5f * sqrt(xdiff(0) * xdiff(0) + xdiff(1) * xdiff(1)) * curv * float(charge);
            const float ca = sqrt(1.f - sa * sa);

            // Rotate momentum vector in the xy-plane
            const float pxNew = ca * pxOld + sa * pyOld;
            const float pyNew = -sa * pxOld + ca * pyOld;
            
            const Vec3 pNew(pxNew, pyNew, pz);

            return FreeTrajectoryState<Vec3>(pNew, xmeas, charge);
        }

    }  // namespace ftsFromVertexToPointPortable

} // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif // RecoEgamma_EgammaElectronAlgos_interface_alpaka_ftsFromVertexToPointPortable_h

