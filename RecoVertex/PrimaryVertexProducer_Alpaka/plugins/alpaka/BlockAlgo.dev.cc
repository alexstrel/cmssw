#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoVertex/PrimaryVertexProducer_Alpaka/plugins/alpaka/BlockAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace cms::alpakatools; 

  class createBlocksKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,  const portablevertex::TrackDeviceCollection::ConstView inputTracks,  portablevertex::TrackDeviceCollection::View trackInBlocks, double blockOverlap, int32_t blockSize) const{
      printf("[BlockAlgo::operator()] Start\n");
      printf("[BlockAlgo::operator()] blockOverlap: %1.3f, blockSize %i\n",blockOverlap, blockSize);
      int32_t bsize = blockSize;
      double boverlap = blockOverlap;
      int32_t nTOld = inputTracks.nT();
      printf("[BlockAlgo::operator()] bsize: %i, boverlap %1.3f, nTOld %i\n", bsize, boverlap, nTOld);
      int32_t nBlocks = nTOld > bsize ? int32_t ((nTOld-1)/(boverlap*bsize)) : 1; // If all fit within a block, no need to split
      printf("[BlockAlgo::operator()] nBlocks: %i\n", nBlocks);
      int32_t overlapStart = boverlap*bsize; // First block starts at 0, second block starts at overlapStart, third at 2*overlapStart and so on
      printf("[BlockAlgo::operator()] bsize: %i\n", bsize);
      for (auto iNewTrack : elements_with_stride(acc, bsize) ) { // The accelerator has as much threads as blockSize, so i will enter once on each block
	//P// printf("[BlockAlgo::operator()] iNewTrack %i, nBlocks %i\n", iNewTrack, nBlocks);
        for (int32_t iblock = 0; iblock < nBlocks; iblock++){
      	  int32_t oldIndex = ((int32_t) iblock*overlapStart) + iNewTrack; // I.e. first track in the block in which we are + thread in which we are
	  if (oldIndex >= nTOld) break; // I.e. we reached the end of the input block
          int32_t newIndex = iNewTrack+iblock*bsize;
	  printf("[BlockAlgo::operator()] iblock %i, oldIndex %i => newIndex %i, x: %1.3f, y: %1.3f, z:%1.3f\n", iblock, oldIndex, newIndex, inputTracks[oldIndex].x(),inputTracks[oldIndex].y(), inputTracks[oldIndex].z());
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
      printf("[BlockAlgo::operator()] End\n");
    } // createBlocksKernel::operator()
  }; // class createBlocksKernel

  BlockAlgo::BlockAlgo(Queue& queue) {
  } // BlockAlgo::BlockAlgo
  
  void BlockAlgo::createBlocks(Queue& queue, const portablevertex::TrackDeviceCollection& inputTracks, portablevertex::TrackDeviceCollection& trackInBlocks, int32_t bSize, double bOverlap){
    const int threadsPerBlock = bSize;// each thread will write nBlocks tracks
    const int blocks = 1;             // 1 block with all threads
    alpaka::exec<Acc1D>(queue,
		        make_workdiv<Acc1D>(blocks, threadsPerBlock),
			createBlocksKernel{},
			inputTracks.view(), // TODO:: Maybe we can optimize the compiler by not making this const? Tracks would not be modified
			trackInBlocks.view(),
			bOverlap,
			bSize
			); 
  } // BlockAlgo::createBlocks
} // namespace ALPAKA_ACCELERATOR_NAMESPACE
