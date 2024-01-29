#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoVertex/PrimaryVertexProducer_Alpaka/plugins/alpaka/BlockAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace cms::alpakatools; 

  class createBlocksKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,  const portablevertex::TrackDeviceCollection::ConstView inputTracks,  portablevertex::TrackDeviceCollection::View trackInBlocks, double* blockOverlap, int32_t* blockSize) const{
      int32_t bsize = *blockSize;
      double boverlap = *blockOverlap;
      int32_t nTOld = inputTracks.nT();
      int32_t nBlocks = int32_t ((nTOld+bsize)/(boverlap*bsize));
      int32_t overlapStart = boverlap*bsize; // First block starts at 0, second block starts at overlapStart, third at 2*overlapStart and so on
      for (auto iNewTrack : elements_with_stride(acc, bsize) ) { // The accelerator has as much threads as blockSize, so i will enter once on each block
        for (int32_t iblock = 0; iblock < nBlocks+1; iblock++){
      	  int32_t oldIndex = ((int32_t) iblock*overlapStart) + iNewTrack; // I.e. first track in the block in which we are + thread in which we are
          int32_t newIndex = iNewTrack+iblock*bsize;
	  // And just copy
	  trackInBlocks[newIndex].x() = inputTracks[oldIndex].x();
          trackInBlocks[newIndex].y() = inputTracks[oldIndex].y();
          trackInBlocks[newIndex].z() = inputTracks[oldIndex].z();
          trackInBlocks[newIndex].px() = inputTracks[oldIndex].px();
          trackInBlocks[newIndex].py() = inputTracks[oldIndex].py();
          trackInBlocks[newIndex].pz() = inputTracks[oldIndex].pz();
          trackInBlocks[newIndex].weight() = inputTracks[oldIndex].weight();
          trackInBlocks[newIndex].tt_index() = inputTracks[oldIndex].tt_index(); // Relevant, as this let us merge tracks later
          trackInBlocks[newIndex].dz2() = inputTracks[oldIndex].dz2();
          trackInBlocks[newIndex].oneoverdz2() = inputTracks[oldIndex].oneoverdz2();
          trackInBlocks[newIndex].dxy2AtIP() = inputTracks[oldIndex].dxy2AtIP();
          trackInBlocks[newIndex].dxy2() = inputTracks[oldIndex].dxy2();
          trackInBlocks[newIndex].sum_Z() = inputTracks[oldIndex].order();
          trackInBlocks[newIndex].kmin() = inputTracks[oldIndex].kmin();
          trackInBlocks[newIndex].kmax() = inputTracks[oldIndex].kmax();
          trackInBlocks[newIndex].aux1() = inputTracks[oldIndex].aux1();
          trackInBlocks[newIndex].aux2() = inputTracks[oldIndex].aux2();
          trackInBlocks[newIndex].isGood() = inputTracks[oldIndex].isGood();
	} // iblock for
      } // iNewTrack for
    } // createBlocksKernel::operator()
  }; // class createBlocksKernel

  BlockAlgo::BlockAlgo(Queue& queue, const uint32_t nT, int32_t bSize, double bOverlap) : nT(nT),blockSize(cms::alpakatools::make_device_buffer<int32_t>(queue)),blockOverlap(cms::alpakatools::make_device_buffer<double>(queue))  {
    alpaka::memset(queue,  blockSize, bSize);
    alpaka::memset(queue,  blockOverlap, bOverlap);
  } // BlockAlgo::BlockAlgo
  
  void BlockAlgo::createBlocks(Queue& queue, const portablevertex::TrackDeviceCollection& inputTracks, portablevertex::TrackDeviceCollection& trackInBlocks, int32_t bSize, int32_t nBlocks){
    const int threadsPerBlock = bSize;
    const int blocks = divide_up_by(bSize, bSize); //1 thread will write nBlocks tracks
    alpaka::exec<Acc1D>(queue,
		        make_workdiv<Acc1D>(blocks, threadsPerBlock),
			createBlocksKernel{},
			inputTracks.view(), // TODO:: Maybe we can optimize the compiler by not making this const? Tracks would not be modified
			trackInBlocks.view(),
			blockOverlap.data(),
			blockSize.data()
			); 
  } // BlockAlgo::createBlocks
} // namespace ALPAKA_ACCELERATOR_NAMESPACE
