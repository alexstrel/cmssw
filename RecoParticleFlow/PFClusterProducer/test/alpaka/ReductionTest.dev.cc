#include <random>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"


#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"

#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthClusterizerHelper.h"

namespace cmstest {

  GENERATE_SOA_LAYOUT(TestPFClusterSoALayout,
                      SOA_COLUMN(int, depth),
                      SOA_COLUMN(int, seedRHIdx),
                      SOA_COLUMN(int, topoId),
                      SOA_COLUMN(int, rhfracSize),
                      SOA_COLUMN(int, rhfracOffset),
                      SOA_COLUMN(float, energy),
                      SOA_COLUMN(float, x),
                      SOA_COLUMN(float, y),
                      SOA_COLUMN(float, z),
                      SOA_COLUMN(int, topoRHCount),
                      SOA_SCALAR(int, nTopos),
                      SOA_SCALAR(int, nSeeds),
                      SOA_SCALAR(int, nRHFracs),
                      SOA_SCALAR(int, size)  // nRH
  )
  using TestPFClusterSoA = TestPFClusterSoALayout<>;
  using TestPFClusterHostCollection = PortableHostCollection<TestPFClusterSoA>;
}  //


namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace cmstest {
    using namespace ::cmstest;

    using TestPFClusterDeviceCollection = PortableCollection<::cmstest::TestPFClusterSoA>;
  }  // namespace portabletest


  class PFClusterTest {
  public:
    void runTest(Queue& queue, cmstest::TestPFClusterDeviceCollection& collection) const;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#define PRINT_CASE

namespace ALPAKA_ACCELERATOR_NAMESPACE {

        //  Define operation type: 
        template <bool comp_min>
        struct CompFn {
          ALPAKA_FN_ACC float operator()(float a, float b) const {
          if constexpr (comp_min)
            return a < b ? a : b;
          else
            return a > b ? a : b;
          }
        };

	class PFClusterTestKernel {
	public:
		template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
		ALPAKA_FN_ACC void operator()(TAcc const& acc,
					      cmstest::TestPFClusterDeviceCollection::View in) const 
		{
			const unsigned int nClusters = in.size();
      
                        const unsigned int w_extent = alpaka::warp::getSize(acc);

			for (auto group : ::cms::alpakatools::uniform_groups(acc)) {  //loop over thread blocks
		          // Only single block is active:
                          if (group != 0) continue;
			  const unsigned int boundary = (w_extent - 20);

                          for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, ::cms::alpakatools::round_up_by(nClusters, w_extent))) {
		             if(idx.local == 0) printf("Entry kernel body...\n");
			     const unsigned int active_lanes_mask = alpaka::warp::ballot(acc, idx.global < boundary);

			     if (idx.local >= boundary) continue;

			     const unsigned int lane_idx = idx.local % w_extent;

			     unsigned int val = idx.global % 2;

			     const unsigned int even_lanes_mask = warp::ballot_mask(acc,active_lanes_mask,idx.global % 2 == 0);

			     if(idx.local < 2) printf("Even lanes mask = %u (seen from lane %u)\n", even_lanes_mask, lane_idx );

			     if(is_work_lane(even_lanes_mask,lane_idx,w_extent)){
				val = warp::shfl_mask( acc, even_lanes_mask, val, 0, w_extent );     
			     }

			     warp::syncWarpThreads_mask(acc, active_lanes_mask);

			     const unsigned int odd_lanes_mask = warp::ballot_mask(acc,active_lanes_mask,idx.global % 2 == 1);

                             if(idx.local < 2) printf("Odd lanes mask = %u (seen from lane %u)\n", odd_lanes_mask, lane_idx );

                             if(is_work_lane(odd_lanes_mask,lane_idx,w_extent)){
                                val = warp::shfl_mask( acc, odd_lanes_mask, val, 1, w_extent );
                             }

			     warp::syncWarpThreads_mask(acc, active_lanes_mask);

			     const unsigned int evenodd_mask = warp::match_any_mask(acc, active_lanes_mask, val);

			     if(idx.local < 2) printf("Evenodd lanes mask = %u (seen from lane %u)\n", evenodd_mask, lane_idx );

			     warp::syncWarpThreads_mask(acc, active_lanes_mask);

			     const unsigned int local_offset = in[idx.global].rhfracSize();
			     const unsigned int warp_offsets = warp_sparse_exclusive_sum(acc, evenodd_mask, local_offset, lane_idx);

			     if( is_work_lane(active_lanes_mask, idx.local, w_extent) ) printf("Result %u, \t lane id %u \t offset %u\n", warp_offsets, lane_idx, local_offset);

			     warp::syncWarpThreads_mask(acc, active_lanes_mask);
                             const float energy = in[idx.global].energy(); 
			     const float min_energy = warp_sparse_reduce(acc, active_lanes_mask, lane_idx, energy, CompFn<true>());

			     if( is_work_lane(active_lanes_mask, idx.local, w_extent) ) printf("Current energy = %f (seen from lane %u)\n", energy, lane_idx );

			     warp::syncWarpThreads_mask(acc, active_lanes_mask);

			     const auto winner_mask = warp::ballot_mask(acc,active_lanes_mask,energy == min_energy);

			     if(idx.local == 0) printf("Min energy = %f (seen from lane %u, local energy is %f)\n", min_energy, lane_idx, energy );

			     if( is_work_lane(winner_mask, idx.local, w_extent) ) printf("Orig lane with min energy = %f (seen from lane %u)\n", energy, lane_idx );
			  }
			}
		}
	};

	void PFClusterTest::runTest(Queue& queue, cmstest::TestPFClusterDeviceCollection& collection ) const 
	{
		uint32_t items = 1024;

	        auto n = static_cast<uint32_t>(collection->metadata().size());
		uint32_t groups = cms::alpakatools::divide_up_by(n, items);

		if(groups<1) {
		  printf("Skip kernel launch...\n");
		  return;
		}

		auto workDiv =cms::alpakatools:: make_workdiv<Acc1D>(groups, items);

		alpaka::exec<Acc1D>(queue, workDiv, PFClusterTestKernel{}, collection.view());

		alpaka::wait(queue);
	}

	void launch_test(Queue& queue, const int collectionSize) {

	  PFClusterTest pfcluster_test_{};
    	  // Create device products :
	  cmstest::TestPFClusterHostCollection hostProduct{collectionSize, queue};

          std::random_device rd;
          std::mt19937 gen(rd());

          std::normal_distribution<float> distr(0.f, 1.f);

	  auto& viewProduct = hostProduct.view();

          for( int i = 0; i < collectionSize; i++ ) {
	     viewProduct[i].depth()        = i;
             viewProduct[i].seedRHIdx()    = i;
             viewProduct[i].topoId()       = i;
             viewProduct[i].rhfracSize()   = i + 1;	     
             viewProduct[i].rhfracOffset() = i;

             viewProduct[i].energy() = fabs(distr(gen));

	     viewProduct[i].x() = distr(gen);
	     viewProduct[i].y() = distr(gen);
	     viewProduct[i].z() = distr(gen);

             viewProduct[i].topoRHCount() = collectionSize / (i+1);

	  }

	  viewProduct.nTopos() = collectionSize;
          viewProduct.nSeeds() = collectionSize;
          viewProduct.nRHFracs() = collectionSize;

          viewProduct.size() = collectionSize;

    	  cmstest::TestPFClusterDeviceCollection deviceProduct{collectionSize, queue};

          alpaka::memcpy(queue, deviceProduct.buffer(), hostProduct.buffer());

	  printf("Run kernel: \n");
    	  pfcluster_test_.runTest(queue, deviceProduct);

	  printf("...done\n");

	  alpaka::wait(queue);
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

    const int32_t collectionSize = 100;

    launch_test(queue, collectionSize);
  }

  return 0;
}








