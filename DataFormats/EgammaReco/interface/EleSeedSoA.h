#ifndef DataFormats_EgammaReco_interface_EleSeedSoA_h
#define DataFormats_EgammaReco_interface_EleSeedSoA_h

#include <Eigen/Core>
#include <cstdint>
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace reco {
	GENERATE_SOA_LAYOUT(EleSeedLayout,
		SOA_COLUMN(int32_t, nHits),
		SOA_EIGEN_COLUMN(Eigen::Vector3d, hitPosX),
		SOA_EIGEN_COLUMN(Eigen::Vector3d, hitPosY),
		SOA_EIGEN_COLUMN(Eigen::Vector3d, hitPosZ),
		SOA_EIGEN_COLUMN(Eigen::Vector3d, surfPosX),
		SOA_EIGEN_COLUMN(Eigen::Vector3d, surfPosY),
		SOA_EIGEN_COLUMN(Eigen::Vector3d, surfPosZ),
		SOA_EIGEN_COLUMN(Eigen::Vector3d, surfRotX),
		SOA_EIGEN_COLUMN(Eigen::Vector3d, surfRotY),
		SOA_EIGEN_COLUMN(Eigen::Vector3d, surfRotZ),
		SOA_EIGEN_COLUMN(Eigen::Vector3d, detectorID),
		SOA_EIGEN_COLUMN(Eigen::Vector3d, isValid),
		SOA_COLUMN(int32_t, isMatched),
		SOA_COLUMN(int32_t, matchedScID),
		SOA_COLUMN(int32_t, id)
	)
	using EleSeedSoA = EleSeedLayout<>;
}  // namespace reco

#endif
