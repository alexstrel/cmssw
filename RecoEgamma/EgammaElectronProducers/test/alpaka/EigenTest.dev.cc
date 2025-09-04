#include <Eigen/Core>
#include <random>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
//
#include "FWCore/Utilities/interface/FileInPath.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/stringize.h"
//
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "EigenTest.h"

#define CRASH

namespace ALPAKA_ACCELERATOR_NAMESPACE {

	class EigenKernel {
	public:
		template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
		ALPAKA_FN_ACC void operator()(TAcc const& acc,
					      cmstest::DummyEigDeviceCollection::View in_view,
					      const int32_t size) const 
		{

			for (int i : cms::alpakatools::uniform_elements(acc,size)) 
			{

				if(i > 0) break;
#ifdef CRASH
				in_view[i].C() = 1;
				if(!(in_view[i].C())) continue;
#endif
				// Access first hit information
				Vec3 s0(in_view[i].G0().x(),in_view[i].G0().y(),in_view[i].G0().z());
				Vec3 s1(in_view[i].H0().x(),in_view[i].H0().y(),in_view[i].H0().z());
				Vec3 s2(in_view[i].I0().x(),in_view[i].I0().y(),in_view[i].I0().z());

				Vec3 s3(in_view[i].G1().x(),in_view[i].G1().y(),in_view[i].G1().z());
				Vec3 s4(in_view[i].H1().x(),in_view[i].H1().y(),in_view[i].H1().z());

				{

			 		{
						in_view[i].B() = 0;
						in_view[i].A() = 1;

					}
				}
			}
		}
	};

	void EigenTest::runTest(Queue& queue, cmstest::DummyEigDeviceCollection& collection ) const 
	{
		uint32_t items = 32;

	        auto n = static_cast<uint32_t>(collection->metadata().size());
		uint32_t groups = cms::alpakatools::divide_up_by(n, items);

		if(groups<1) {
			printf("Skip kernel launch...\n");
			return;
		}

		auto workDiv =cms::alpakatools:: make_workdiv<Acc1D>(groups, items);

		alpaka::exec<Acc1D>(queue, workDiv, EigenKernel{}, collection.view(), collection->metadata().size());
	}

	void launch_test(Queue& queue, const int collectionSize) {

	  EigenTest eig_test_{};
    	  // Create device products :
	  cmstest::DummyEigHostCollection hostProduct{collectionSize, queue};

          std::random_device rd;
          std::mt19937 gen(rd());

          std::normal_distribution<float> distr(0.f, 1.f);

	  auto& viewProduct = hostProduct.view();

          for( int i = 0; i < collectionSize; i++ ) {
	     viewProduct[i].A() = 1;
             viewProduct[i].B() = 1;
             viewProduct[i].C() = 1;
             viewProduct[i].D() = 1;	     
             viewProduct[i].E() = 1;
             viewProduct[i].F() = 1;

             viewProduct[i].G0() = Vec3(distr(gen), distr(gen), distr(gen));
	     viewProduct[i].H0() = Vec3(distr(gen), distr(gen), distr(gen));
	     viewProduct[i].I0() = Vec3(distr(gen), distr(gen), distr(gen));

	     viewProduct[i].G1() = Vec3(distr(gen), distr(gen), distr(gen));
             viewProduct[i].H1() = Vec3(distr(gen), distr(gen), distr(gen));
             viewProduct[i].I1() = Vec3(distr(gen), distr(gen), distr(gen));

	     viewProduct[i].G2() = Vec3(distr(gen), distr(gen), distr(gen));
             viewProduct[i].H2() = Vec3(distr(gen), distr(gen), distr(gen));
             viewProduct[i].I2() = Vec3(distr(gen), distr(gen), distr(gen));

	  }

    	  cmstest::DummyEigDeviceCollection deviceProduct{collectionSize, queue};

          alpaka::memcpy(queue, deviceProduct.buffer(), hostProduct.buffer());

    	  eig_test_.runTest(queue, deviceProduct);

	}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

using namespace edm;
using namespace std;
using namespace ALPAKA_ACCELERATOR_NAMESPACE;

int main() {
  // get the list of devices on the current platform
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cerr << "No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
      "the test will be skipped.\n";
    exit(EXIT_FAILURE);
  }

  // run the test on each device
  for (auto const& device : devices) {
    auto queue = Queue(device);

    const int32_t collectionSize = 1024;

    launch_test(queue, collectionSize);
  }

  return 0;
}








