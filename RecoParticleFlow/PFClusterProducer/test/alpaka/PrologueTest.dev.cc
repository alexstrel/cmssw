#include <random>
#include <vector>
#include <cmath>
#include <algorithm>

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

#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthECLCCPrologue.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFMultiDepthClusteringVarsHostCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFMultiDepthClusteringEdgeVarsHostCollection.h"

using namespace reco;

static bool verbose = true;


namespace ALPAKA_ACCELERATOR_NAMESPACE {

	class PrologueTest {
          public:
            void apply(Queue& queue, 
                       reco::PFMultiDepthClusteringEdgeVarsDeviceCollection &pfClusteringEdgeVars,
                       const reco::PFMultiDepthClusteringVarsDeviceCollection &mdpfClusteringVars) const;
        };

	void PrologueTest::apply(Queue& queue,
                             reco::PFMultiDepthClusteringEdgeVarsDeviceCollection &pfClusteringEdgeVars, 
                             const reco::PFMultiDepthClusteringVarsDeviceCollection &mdpfClusteringVars) const 
	{
	  uint32_t items = 1024;

	  auto n = static_cast<uint32_t>(mdpfClusteringVars->metadata().size());
	  uint32_t groups = cms::alpakatools::divide_up_by(n, items);

	  if(groups<1) {
	    printf("Skip kernel launch...\n");
	    return;
	  } else {
            printf("Number of groups :: %d\n", groups);		  
	  }

	  auto workDiv =cms::alpakatools:: make_workdiv<Acc1D>(groups, items);

	  alpaka::exec<Acc1D>(queue, workDiv, ECLCCPrologueKernel{}, pfClusteringEdgeVars.view(), mdpfClusteringVars.view());

	  alpaka::wait(queue);
	}

  void launch_prologue_test( Queue& queue,
                             ::reco::PFMultiDepthClusteringEdgeVarsHostCollection &hostClusteringEdgeVars,
                             const ::reco::PFMultiDepthClusteringVarsHostCollection &hostClusteringVars) {
    PrologueTest prologue_test{};                      

    auto hClusteringVars = hostClusteringVars.view();

    const int nClusters = hClusteringVars.size();

    reco::PFMultiDepthClusteringVarsDeviceCollection devClusteringVars{nClusters, queue};
    reco::PFMultiDepthClusteringEdgeVarsDeviceCollection devClusteringEdgeVars{2*nClusters, queue};

    alpaka::memcpy(queue, devClusteringVars.buffer(), hostClusteringVars.buffer());

    prologue_test.apply(queue, devClusteringEdgeVars, devClusteringVars);

    alpaka::wait(queue);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

using namespace ALPAKA_ACCELERATOR_NAMESPACE;


  void create(::reco::PFMultiDepthClusteringVarsHostCollection &hostClusteringVars, const std::vector<int> &roots, const int nClusters) {
    auto hClusteringVars = hostClusteringVars.view();

    std::mt19937 rng(12345);

    std::uniform_int_distribution<int> topo_distr(0, nClusters-1);

    for (int i = 0; i < nClusters; ++i) {
      hClusteringVars[i].depth() = 1.f;
      hClusteringVars[i].energy() = 0.f;

      hClusteringVars[i].eta() = 0.;
      hClusteringVars[i].phi() = 0.;

      hClusteringVars[i].etaRMS2() = 0.;
      hClusteringVars[i].phiRMS2() = 0.;

      bool is_root = std::binary_search(roots.begin(), roots.end(), i);

      hClusteringVars[i].mdpf_topoId() = is_root ? i : topo_distr(rng);
    }
  }


using namespace edm;
using namespace std;

int main() {
  // get the list of devices on the current platform
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cerr << "No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
      "the test will be skipped.\n";
    exit(EXIT_FAILURE);
  }

  const int nClusters = 100;

  std::vector<int> roots = {0,3,7,11,19,29,37,41,71,83,97};

  // run the test on each device
  for (auto const& device : devices) {
    auto queue = Queue(device);

    ::reco::PFMultiDepthClusteringVarsHostCollection hostClusteringVars{nClusters, queue};    
    ::reco::PFMultiDepthClusteringEdgeVarsHostCollection hostClusteringEdgeVars{2*nClusters, queue};

    auto hClusteringVars = hostClusteringVars.view();
    hClusteringVars.size() = nClusters;

    create(hostClusteringVars, roots, nClusters);

    launch_prologue_test(queue, hostClusteringEdgeVars, hostClusteringVars);
  }

  return 0;
}








