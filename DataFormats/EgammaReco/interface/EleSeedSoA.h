#ifndef DataFormats_EgammaReco_interface_EleSeedSoA_h
#define DataFormats_EgammaReco_interface_EleSeedSoA_h

#include <Eigen/Core>
#include <cstdint>
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"


namespace reco {

	using Vector3d = Eigen::Matrix<double, 3, 1>;

	GENERATE_SOA_LAYOUT(EleSeedLayout,
		SOA_COLUMN(int32_t, nHits),
		SOA_COLUMN(int16_t, isMatched),
		SOA_COLUMN(int16_t, matchedScID),
		SOA_COLUMN(int16_t, id),
		SOA_COLUMN(int16_t, hit0detectorID),
		SOA_COLUMN(int16_t, hit0isValid),
		SOA_COLUMN(int16_t, hit1detectorID),
		SOA_COLUMN(int16_t, hit1isValid),
		SOA_COLUMN(int16_t, hit2detectorID),
		SOA_COLUMN(int16_t, hit2isValid),
		SOA_EIGEN_COLUMN(Vector3d, hit0Pos),
		SOA_EIGEN_COLUMN(Vector3d, surf0Pos),
		SOA_EIGEN_COLUMN(Vector3d, surf0Rot),
		SOA_EIGEN_COLUMN(Vector3d, hit1Pos),
		SOA_EIGEN_COLUMN(Vector3d, surf1Pos),
		SOA_EIGEN_COLUMN(Vector3d, surf1Rot),
		SOA_EIGEN_COLUMN(Vector3d, hit2Pos),
		SOA_EIGEN_COLUMN(Vector3d, surf2Pos),
		SOA_EIGEN_COLUMN(Vector3d, surf2Rot)
	)
	using EleSeedSoA = EleSeedLayout<>;
}  // namespace reco

#endif
