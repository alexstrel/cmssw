#ifndef DataFormats_EgammaReco_interface_alpaka_EleRelPointPairPortable_h
#define DataFormats_EgammaReco_interface_alpaka_EleRelPointPairPortable_h

#include <Eigen/Dense>
#include <cmath>
#include <algorithm>

namespace ALPAKA_ACCELERATOR_NAMESPACE {

    namespace EleRelPointPairPortable {

        template <typename Vec3>
        struct EleRelPointPair {
            Vec3 relP1; // Relative point 1
            Vec3 relP2; // Relative point 2

            // Constructor to compute relative points
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE EleRelPointPair(const Vec3& p1, const Vec3& p2, const Vec3& origin)
                : relP1(relativePosition(p1, origin)), relP2(relativePosition(p2, origin)) {}

            // Calculate differences
            //constexpr auto dEta() const { return relative_eta(relP1, relP2); }
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto dZ() const { return (relP1.z() - relP2.z()); }
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto dPerp() const { return (relP1.head(2).norm() - relP2.head(2).norm()); }

            
            // Helper function to compute relative position
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE Vec3 relativePosition(const Vec3& point, const Vec3& origin) {
                return Vec3(point(0) - origin(0), point(1) - origin(1), point(2) - origin(2));
            }

            // Calculate relative eta
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static double relative_eta(const Vec3& p, const Vec3& origin) {
                return (p - origin).eta();
            }

            // Calculate relative phi
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static double relative_phi(const Vec3& p1, const Vec3& p2) {
                auto phi = std::atan2(p1(1),p1(0)) - std::atan2(p2(1),p2(0));
                return reduceRange(phi);
            }
            
            // Normalize phi to the range [-pi, pi]
            template <typename T>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr T reduceRange(T x) {
                constexpr T o2pi = 1. / (2. * M_PI);
                if (std::abs(x) <= T(M_PI))
                    return x;
                return x - std::floor(x * o2pi + (x < 0 ? -0.5 : 0.5)) * 2. * M_PI;
            }

            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto dPhi() const { return relative_phi(relP1, relP2); }

        };

    }  // namespace EleRelPointPairPortable

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif // DataFormats_EgammaReco_interface_alpaka_EleRelPointPairPortable_h
