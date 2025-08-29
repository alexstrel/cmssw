#ifndef DataFormats_EgammaReco_interface_alpaka_Plane_h
#define DataFormats_EgammaReco_interface_alpaka_Plane_h

#include <cmath>
#include <DataFormats/EgammaReco/interface/alpaka/Phys3DVector.h>

//#define EIGEN_DONT_VECTORIZE

#include <Eigen/Core>

#define CRASH_IT

namespace ALPAKA_ACCELERATOR_NAMESPACE {

    namespace PlanePortable {

        template <typename T = double>
        class Plane {
          public:		
	    using Vec3      = cms::alpakatools::math::Phys3DVector<T>;
	    using EigenVec3 = Eigen::Vector3d;

            // Constructor
	    ALPAKA_FN_HOST_ACC Plane(const Vec3& pos, const Vec3& rot) {
	       for(int i = 0; i < 3; i++) {
		 position[i] = pos[i];
	         rotation[i] = rot[i];	 
	       }	       
	    }
            constexpr  Plane(const EigenVec3& pos, const EigenVec3& rot) : position(pos), rotation(rot) {}

            // Returns the position of the plane
	    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE Vec3 pos() const {		    
                return Vec3(position[0], position[1], position[2]);
            }

           // Returns a specific component of the position of the plane
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T pos(const unsigned int x) const {
                return position[x];
            }

	    template <typename TAcc>
	    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T pos_norm(TAcc const& acc) const {
                return position.norm();
            }

            // Returns the normal vector of the plane
	    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE Vec3 normalVector() const {
                return Vec3(rotation[0], rotation[1], rotation[2]); 
            }

            // Fast access to distance from plane for a vector
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T distanceFromPlaneVector(const Vec3& gv) const {
                EigenVec3 tmp;
                tmp[0] = gv[0]; tmp[1] = gv[1]; tmp[2] = gv[2];
                return rotation.dot(tmp);
            }

#ifndef CRASH_IT 
	    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T localZ(const Vec3& vp) const {

                T tmp_dot{0};
                CMS_UNROLL_LOOP
                for (unsigned int i = 0; i < 3; i++){
                  tmp_dot += rotation[i] * (vp[i] - position[i]);
                }
                return tmp_dot;
            }

#else	    
            // Fast access to distance from plane for a point
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T localZ(const Vec3& vp) const {
		EigenVec3 tmp;
                tmp[0] = vp[0] - position[0]; tmp[1] = vp[1] - position[1]; tmp[2] = vp[2] - position[2];
		//return rotation.dot(tmp); //normalVector().dot(vp - position);
		return (rotation[0] * tmp[0] + rotation[1] * tmp[1] + rotation[2] * tmp[2]);//this hack does not help
            }
#endif	    

            // Clamped distance from plane for a point
	    template <typename TAcc>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T localZclamped(TAcc const& acc, const Vec3& vp) const {
                const T d = localZ(vp);
                return alpaka::math::abs(acc, d) > 1e-7f ? d : 0;
            }

	  private:

	    EigenVec3 position;
	    EigenVec3 rotation;  
        };

    }  // namespace PlanePortable

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif // DataFormats_EgammaReco_interface_Plane_h
