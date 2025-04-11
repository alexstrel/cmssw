#ifndef DataFormats_EgammaReco_interface_alpaka_Plane_h
#define DataFormats_EgammaReco_interface_alpaka_Plane_h

#include <cmath>
#include <Eigen/Core>  

namespace ALPAKA_ACCELERATOR_NAMESPACE {

    namespace PlanePortable {

        template <typename Vec3>
        struct Plane {
            Vec3 position;
            Vec3 rotation; // z coordinate of rotation matrix

            // Constructor
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE  Plane(Vec3 pos, Vec3 rot) : position(pos), rotation(rot) {}

            // Returns the position of the plane
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE  Vec3 pos() const {
                return position;
            }

            // Returns the normal vector of the plane
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE  Vec3 normalVector() const {
                return rotation;
            }

            // Fast access to distance from plane for a point
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE  float localZ(const Vec3& vp) const {
                return normalVector().dot(vp - position);
            }

            // Clamped distance from plane for a point
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE  float localZclamped(const Vec3& vp) const {
                auto d = localZ(vp);
                return std::abs(d) > 1e-7f ? d : 0;
            }

            // Fast access to distance from plane for a vector
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE  float distanceFromPlaneVector(const Vec3& gv) const {
                return normalVector().dot(gv);
            }
        };

    }  // namespace PlanePortable

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif // DataFormats_EgammaReco_interface_Plane_h
