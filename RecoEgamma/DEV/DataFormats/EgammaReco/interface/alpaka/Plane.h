#ifndef DataFormats_EgammaReco_interface_alpaka_Plane_h
#define DataFormats_EgammaReco_interface_alpaka_Plane_h

#include <cmath>
#include <HeterogeneousCore/AlpakaInterface/interface/VecArray.h>

namespace ALPAKA_ACCELERATOR_NAMESPACE {

    namespace PlanePortable {

        template <typename T = double>
        class Plane {
          public:		
	    constexpr unsigned int N = 3;	

	    using Vec3 = cms::alpakatools::VecArray<T, N>;

            Vec3 position;
            Vec3 rotation; // z coordinate of rotation matrix

            // Constructor
            constexpr  Plane(Vec3& pos, Vec3& rot) : position(pos), rotation(rot) {}

          private:	    
            // Returns the position of the plane
            constexpr inline Vec3 pos() const {
                return position;
            }

           // Returns a specific component of the position of the plane
            constexpr inline Vec3 pos(const unsigned int x) const {
                return position[x];
            }

	    template <typename TAcc>
	    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T pos_norm(TAcc const& acc) const {
		T res{0.};

		CMS_UNROLL_LOOP
                for (unsigned int i = 0; i < N; i++){
                  res[i] += position[i] * position[i];
                }

                return alpaka::math::sqrt(acc, res);
            }

            // Returns the normal vector of the plane
            constexpr  inline Vec3 normalVector() const {
                return rotation;
            }

            // Fast access to distance from plane for a point
            constexpr inline T localZ(const Vec3& vp) const {
		T diff_dot = static_cast<T>(0.);
                CMS_UNROLL_LOOP
		for (unsigned int i = 0; i < N; i++){
		  diff_dot[i] += rotation[i] * (vp[i] - position[i]);	
		}
                return diff_dot;
            }

            // Clamped distance from plane for a point
	    template <typename TAcc>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T localZclamped(TAcc const& acc, const Vec3& vp) const {
                const T d = localZ(vp);
                return alpaka::math::abs(acc, d) > 1e-7f ? d : 0;
            }

            // Fast access to distance from plane for a vector
            constexpr inline T distanceFromPlaneVector(const Vec3& gv) const {
		T res = static_cast<T>(0.);
                CMS_UNROLL_LOOP		
	        for (unsigned int i = 0; i < N; i++){
	          res += rotation[i] * gv[i];		
		}	
                return res;
            }
        };

    }  // namespace PlanePortable

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif // DataFormats_EgammaReco_interface_Plane_h
