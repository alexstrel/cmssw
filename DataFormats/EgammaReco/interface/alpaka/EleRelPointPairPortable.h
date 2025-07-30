#ifndef DataFormats_EgammaReco_interface_alpaka_EleRelPointPairPortable_h
#define DataFormats_EgammaReco_interface_alpaka_EleRelPointPairPortable_h

#include <cmath>
//#include <algorithm>
#include <HeterogeneousCore/AlpakaInterface/interface/VecArray.h>

namespace ALPAKA_ACCELERATOR_NAMESPACE {

    namespace EleRelPointPairPortable {

        template <typename T = double>
        class EleRelPointPair {

	  public: 
	    using Vec3 = cms::alpakatools::VecArray<T, 3>; 

            // Constructor to compute relative points
            constexpr EleRelPointPair(const Vec3& p1, const Vec3& p2, const Vec3& origin)
                : relP1(relativePosition(p1, origin)), relP2(relativePosition(p2, origin)) {}

            // Calculate differences
            //constexpr auto dEta() const { return relative_eta(relP1, relP2); }
            constexpr inline T dZ() const { return (relP1[2] - relP2[2]); }

	    template <typename TAcc>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T dPerp(TAcc const& acc) const {
		    const T relP1_2dnorm = relP1.template partial_norm<TAcc, 2>(acc);
		    const T relP2_2dnorm = relP2.template partial_norm<TAcc, 2>(acc);

		    return (relP1_2dnorm - relP2_2dnorm); 
	    }

            
            // Helper function to compute relative position
            constexpr Vec3 relativePosition(const Vec3& point, const Vec3& origin) const {
                return cms::alpakatools::xmy(point, origin);
            }

            // Calculate  relative eta
	    template <typename TAcc>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T relative_eta(TAcc const& acc, const Vec3& p, const Vec3& origin) const {

		const T tmp  = cms::alpakatools::diff2(p, origin);
                const T pdiff = alpaka::math::sqrt(acc,tmp);
		const T z     = p[2] - origin[2];

	        return 0.5 * alpaka::math::log(acc, (pdiff + z) / (pdiff - z) );
            }

            // Calculate relative phi
	    template <typename TAcc>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T relative_phi(TAcc const& acc, const Vec3& p1, const Vec3& p2) const {
                const T phi = alpaka::math::atan2(acc, p1[1],p1[0]) - alpaka::math::atan2(acc, p2[1],p2[0]);
                return reduceRange(acc, phi);
            }
            
            // Normalize phi to the range [-pi, pi]
            template <typename TAcc>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T reduceRange(TAcc const& acc, const T x) const {
                constexpr T o2pi = 1. / (2. * M_PI);
                if (alpaka::math::abs(acc, x) <= T(M_PI))
                    return x;
                return x - alpaka::math::floor(acc, x * o2pi + (x < 0 ? -0.5 : 0.5)) * 2. * M_PI;
            }

	    template <typename TAcc>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto dPhi(TAcc const& acc) const { return relative_phi(acc, relP1, relP2); }

	  private:
            Vec3 relP1; // Relative point 1
            Vec3 relP2; // Relative point 2

        };

    }  // namespace EleRelPointPairPortable

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif // DataFormats_EgammaReco_interface_alpaka_EleRelPointPairPortable_h
