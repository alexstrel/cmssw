#ifndef RecoEgamma_EgammaElectronProducers_test_alpaka_EigenTest_h
#define RecoEgamma_EgammaElectronProducers_test_alpaka_EigenTest_h

//#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include <Eigen/Core>
#include <cstdint>
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"

using Vec3 = Eigen::Matrix<float, 3, 1>;

namespace cmstest {

        GENERATE_SOA_LAYOUT(DummyEigLayout,
                SOA_COLUMN(int, A),//Used
                SOA_COLUMN(int, B),//Used
                SOA_COLUMN(int, C),//Used
                SOA_COLUMN(int, D),//Not used
                SOA_COLUMN(int, E),//Not used
                SOA_COLUMN(int, F),//Not used
                SOA_EIGEN_COLUMN(Vec3, G0),//Used
                SOA_EIGEN_COLUMN(Vec3, H0),//Used
                SOA_EIGEN_COLUMN(Vec3, I0),//Used
                SOA_EIGEN_COLUMN(Vec3, G1),//Used
                SOA_EIGEN_COLUMN(Vec3, H1),//Used
                SOA_EIGEN_COLUMN(Vec3, I1),//Not used
                SOA_EIGEN_COLUMN(Vec3, G2),//Not used
                SOA_EIGEN_COLUMN(Vec3, H2),//Not used
                SOA_EIGEN_COLUMN(Vec3, I2) //Not used
        )
        using DummyEigSoA = DummyEigLayout<>;

        using DummyEigHostCollection = PortableHostCollection<DummyEigSoA>;

}  // namespace cmstest


namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace cmstest {
    using namespace ::cmstest;

    using DummyEigDeviceCollection = PortableCollection<::cmstest::DummyEigSoA>;
  }  // namespace portabletest


  class EigenTest {
  public:
    void runTest(Queue& queue, cmstest::DummyEigDeviceCollection& collection) const;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
