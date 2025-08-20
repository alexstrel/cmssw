#ifndef DataFormats_EgammaReco_interface_alpaka_Plane_h
#define DataFormats_EgammaReco_interface_alpaka_Plane_h

#include <cmath>
#include <DataFormats/EgammaReco/interface/alpaka/Phys3DVector.h>

namespace ALPAKA_ACCELERATOR_NAMESPACE {

    namespace PlanePortable {

        template <typename T = double>
        class Plane {
          public:		
	    using Vec3 = cms::alpakatools::math::Phys3DVector<T>;

            // Constructor
            constexpr  Plane(const Vec3& pos, const Vec3& rot) : position(pos), rotation(rot) {}

            // Returns the position of the plane
            constexpr inline Vec3 pos() const {
                return position;
            }

           // Returns a specific component of the position of the plane
            constexpr inline T pos(const unsigned int x) const {
                return position[x];
            }

	    template <typename TAcc>
	    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T pos_norm(TAcc const& acc) const {
                return position.norm(acc);
            }

            // Returns the normal vector of the plane
            constexpr  inline Vec3 normalVector() const {
                return rotation;
            }

            // Fast access to distance from plane for a point
            constexpr inline T localZ(const Vec3& vp) const {
		return cms::alpakatools::math::diff_dot(rotation, vp, position);
            }

            // Clamped distance from plane for a point
	    template <typename TAcc>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T localZclamped(TAcc const& acc, const Vec3& vp) const {
                const T d = localZ(vp);
                return alpaka::math::abs(acc, d) > 1e-7f ? d : 0;
            }

            // Fast access to distance from plane for a vector
            constexpr inline T distanceFromPlaneVector(const Vec3& gv) const {
                return cms::alpakatools::math::dot(rotation, gv);
            }

	  private:

	    const Vec3 position;
            const Vec3 rotation;  
        };

    }  // namespace PlanePortable

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif // DataFormats_EgammaReco_interface_Plane_h
