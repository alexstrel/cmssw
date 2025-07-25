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
	    constexpr unsigned int N = 3;	
	    using Vec3 = cms::alpakatools::VecArray<T, N>; 
           	    
            Vec3 relP1; // Relative point 1
            Vec3 relP2; // Relative point 2

            // Constructor to compute relative points
            constexpr EleRelPointPair(const Vec3& p1, const Vec3& p2, const Vec3& origin)
                : relP1(relativePosition(p1, origin)), relP2(relativePosition(p2, origin)) {}

	  private:  
            // Calculate differences
            //constexpr auto dEta() const { return relative_eta(relP1, relP2); }
            constexpr inline T dZ() const { return (relP1[N-1] - relP2[N-1]); }

	    template <typename TAcc>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static T dPerp(TAcc const& acc) const {
		    const T relP1_2dnorm = relP1.template partial_norm<Tacc, 2>();
		    const T relP2_2dnorm = relP2.template partial_norm<Tacc, 2>();

		    return (relP1_2dnorm - relP2_2dnorm); 
	    }

            
            // Helper function to compute relative position
            constexpr Vec3 relativePosition(const Vec3& point, const Vec3& origin) {
                return cms::alpakatools::xmy(point - origin);
            }

            // Calculate  relative eta
	    template <typename TAcc>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static T relative_eta(TAcc const& acc, const Vec3& p, const Vec3& origin) const {
		T res{0};

		CMS_UNROLL_LOOP
		for(unsigned int i = 0; i < N; i++) {
		  const T tmp = p[i] - origin[i];	
		  res += tmp*tmp;	
		}

                const T tmp2 = alpaka::math::sqrt(acc,res);
		const T z    = p[N-1] - origin[N-1];

	        const T eta  = 0.5 * alpaka::math::log(acc, (tmp + z) / (tmp - z) );

                return eta;
            }

            // Calculate relative phi
	    template <typename TAcc>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static T relative_phi(TAcc const& acc, const Vec3& p1, const Vec3& p2) {
                const T phi = alpaka::math::atan2(acc, p1[1],p1[0]) - alpaka::math::atan2(acc, p2[1],p2[0]);
                return reduceRange(phi);
            }
            
            // Normalize phi to the range [-pi, pi]
            template <typename TAcc, typename T>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static T reduceRange(TAcc const& acc, const T x) {
                constexpr T o2pi = 1. / (2. * M_PI);
                if (alpaka::math::abs(acc, x) <= T(M_PI))
                    return x;
                return x - alpaka::math::floor(acc, x * o2pi + (x < 0 ? -0.5 : 0.5)) * 2. * M_PI;
            }

            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto dPhi() const { return relative_phi(relP1, relP2); }

        };

    }  // namespace EleRelPointPairPortable

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif // DataFormats_EgammaReco_interface_alpaka_EleRelPointPairPortable_h
