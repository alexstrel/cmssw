#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoVertex/PrimaryVertexProducer_Alpaka/plugins/alpaka/ClusterizerAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace cms::alpakatools;
  ////////////////////// 
  // Device functions //
  //////////////////////
  template <bool debug = false, typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>> ALPAKA_FN_ACC static void initialize(const TAcc& acc, portablevertex::TrackDeviceCollection::View tracks, portablevertex::VertexDeviceCollection::View vertices, const portablevertex::ClusterParamsHostCollection::ConstView cParams){
    // Initialize all vertices as empty, a single vertex in each block will be initialized with all tracks associated to it
    int blockSize = alpaka::getWorkDiv<alpaka::Grid, alpaka:Blocks>(acc)[0u];
    int threadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; // Thread number inside block
    int blockIdx  = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]; // Block number inside grid
    int maxVerticesPerBlock = (int) 512/blockSize; // Max vertices size is 512 over number of blocks in grid
    vertices[blockIdx].nV() = 1; // We start with one vertex per block
    for (int ivertex = threadIdx+maxVerticesPerBlock*blockIdx; ivertex < maxVerticesPerBlock*(blockIdx+1); ivertex+=blockSize){ // Initialize vertices in parallel in the block
      vertices[ivertex].sw() = 0.;
      vertices[ivertex].se() = 0.;
      vertices[ivertex].swz() = 0.;
      vertices[ivertex].swE() = 0.;
      vertices[ivertex].exp() = 0.;
      vertices[ivertex].exparg() = 0.;
      vertices[ivertex].z() = 0.;
      vertices[ivertex].rho() = 0.;
      vertices[ivertex].isGood() = false;
      vertices[ivertex].order() = -1;
      if (ivertex == maxVerticesPerBlock*blockIdx){ // Now set up the initial single vetex containing everything
        // TODO:: Probably there is a cleaner way of doing this in Alpaka
	vertices[ivertex].rho() = 1.;
	vertices[ivertex].order() = maxVerticesPerBlock*blockIdx;
	vertices[ivertex].isGood() = true;
      } 
    } // end for
    alpaka::syncBlockThreads(acc);
    // Now assign all tracks in the block to the single vertex
    for (int itrack = threadIdx+blockIdx*blockSize; itrack < threadIdx+(blockIdx+1)*blockSize ; itrack += blockSize){ // Technically not a loop as each thread will have one track in the per block approach, but in the more general case this can be extended to BlockSize in Alpaka != BlockSize in algorithm
      tracks.kmin(itrack) = maxVerticesPerBlock*blockIdx; // Tracks are associated to vertex in list kmin, kmin+1,... kmax-1, so this just assign all tracks to the vertex we just created!
      tracks.kmax(itrack) = maxVerticesPerBlock*blockIdx + 1;
    }
    alpaka::syncBlockThreads(acc);
  }
  
  template <bool debug = false, typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>> ALPAKA_FN_ACC static void getBeta0(const TAcc& acc, portablevertex::TrackDeviceCollection::View tracks, portablevertex::VertexDeviceCollection::View vertices, const portablevertex::ClusterParamsHostCollection::ConstView cParams, double& _beta){
    // Computes first critical temperature
    int blockSize = alpaka::getWorkDiv<alpaka::Grid, alpaka:Blocks>(acc)[0u];
    int threadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; // Thread number inside block
    int blockIdx  = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]; // Block number inside grid
    int maxVerticesPerBlock = (int) 512/blockSize; // Max vertices size is 512 over number of blocks in grid
    for (int itrack = threadIdx+blockIdx*blockSize; itrack < threadIdx+(blockIdx+1)*blockSize ; itrack += blockSize){
      tracks[itrack].aux1() = tracks[itrack].weight()*tracks[itrack].oneoverdz2();  // Weighted weight
      tracks[itrack].aux2() = tracks[itrack].weight()*tracks[itrack].oneoverdz2()*tracks[itrack].z(); // Weighted position
    }
    // Initial vertex position
    alpaka::syncBlockThreads(acc);
    double& wnew = alpaka::declareSharedVar<double, __COUNTER__>(acc);
    double& znew = alpaka::declareSharedVar<double, __COUNTER__>(acc);
    if (once_per_block(acc)){
      wnew = 0.;
      znew = 0.;
    }
    alpaka::syncBlockThreads(acc);
    for (int itrack = threadIdx+blockIdx*blockSize; itrack < threadIdx+(blockIdx+1)*blockSize ; itrack += blockSize){ // TODO:Saving and reading in the tracks dataformat might be a bit too much?
      alpaka::atomicAdd(acc, &wnew, tracks[itrack].aux1(), alpaka::hierarchy::Threads{});
      alpaka::atomicAdd(acc, &znew, tracks[itrack].aux2(), alpaka::hierarchy::Threads{});
    }
    alpaka::syncBlockThreads(acc);
    if (once_per_block(acc)){
      vertices[maxVerticesPerBlock*blockIdx].z() = znew/wnew;
      znew = 0.;
    }
    alpaka::syncBlockThreads(acc);
    // Now do a chi-2 like of all tracks and save it again in znew
    for (int itrack = threadIdx+blockIdx*blockSize; itrack < threadIdx+(blockIdx+1)*blockSize ; itrack += blockSize){
      tracks[itrack].aux2() = tracks[itrack].aux1()*(vertices[maxVerticesPerBlock*blockIdx.x].z() - tracks[itrack].z() )*(vertices[maxVerticesPerBlock*blockIdx.x].z() - tracks[itrack].z())*tracks[itrack].oneoverdz2();
      alpaka::atomicAdd(acc, &znew, tracks[itrack].aux2(), alpaka::hierarchy::Threads{});
    }
    alpaka::syncBlockThreads(acc);
    if (once_per_block(acc)){
      (*_beta) = 2 * znew/wnew; // 1/beta_C, or T_C
      if (*_beta > cParams.TMin){ // If T_C > T_Min we have a game to play
        int coolingsteps = 1 - int(std::log((*_beta)/ cParams.TMin) / std::log(cParams.coolingFactor)); // A tricky conversion to round the number of cooling steps
        (*_beta) = std::pow(params.coolingFactor, coolingsteps)/cParams.TMin; // First cooling step
      }
      else *_beta = cParams.coolingFactor/cParams.TMin; // Otherwise, just one step
    }
    alpaka::syncBlockThreads(acc);
  }

  template <bool debug = false, typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>> ALPAKA_FN_ACC static void thermalize(const TAcc& acc, portablevertex::TrackDeviceCollection::View tracks, portablevertex::VertexDeviceCollection::View vertices, const portablevertex::ClusterParamsHostCollection::ConstView cParams, double& osumtkwt, double& _beta, double delta_highT, double rho0){
    // At a fixed temperature, iterate vertex position update until stable
    int blockSize = alpaka::getWorkDiv<alpaka::Grid, alpaka:Blocks>(acc)[0u];
    int threadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; // Thread number inside block
    int blockIdx  = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]; // Block number inside grid
    int maxVerticesPerBlock = (int) 512/blockSize; // Max vertices size is 512 over number of blocks in grid
    // Thermalizing iteration
    int niter = 0; 
    double zrange_min_ = 0.01; // Hard coded as in CPU
    double delta_max = cParams.delta_lowT;
    alpaka::syncBlockThreads(acc);
    // Stepping definition
    if (cParams.convergence_mode == 0){
      delta_max = delta_highT;
    }
    else if (cParams.convergence_mode == 1){
      delta_max = cParams.delta_lowT / sqrt(std::max(*_beta, 1.0));
    }
    int maxIterations = 1000;
    alpaka::syncBlockThreads(acc);
    // Always start by resetting track-vertex assignment
    set_vtx_range(acc, tracks, vertices, cParams, osumtkwt, _beta);
    alpaka::syncBlockThreads(acc);
    // Accumulator of variations
    double delta_sum_range = 0;
    while (niter++ < maxIterations){ // Loop until vertex position change is small
      // One iteration of new vertex positions
      update(acc, tracks, vertices, cParams, osumtkwt, _beta, rho0, false);
      alpaka::syncBlockThreads(acc);
      // One iteration of max variation
      double dmax = 0.;
      for (unsigned int ivertexO = maxVerticesPerBlock*blockIdx ; ivertexO < maxVerticesPerBlock*blockIdx + vertices[blockIdx].nV(); ivertexO++){ // TODO::Currently we are doing this in all threads in parallel, might be optimized in other way to multithread max finding?
        unsigned int ivertex = vertices[ivertexO].order();
        if (vertices[ivertex].aux1() >= dmax) dmax = vertices[ivertex].aux1();
      }
      delta_sum_range += dmax;
      alpaka::syncBlockThreads(acc);
      if (delta_sum_range > zrange_min_ && dmax > zrange_min_) {  // I.e., if a vertex moved too much we reassign
        set_vtx_range(acc, tracks, vertices, cParams, osumtkwt, _beta);
	delta_sum_range = 0.;
      }
      alpaka::syncBlockThreads(acc);
      if (dmax < delta_max){ // If it moved too little, we stop updating
        break;
      }
    } // end while
  } // thermalize

  template <bool debug = false, typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>> ALPAKA_FN_ACC static void coolingWhileSplitting(const TAcc& acc, portablevertex::TrackDeviceCollection::View tracks, portablevertex::VertexDeviceCollection::View vertices, const portablevertex::ClusterParamsHostCollection::ConstView cParams, double& osumtkwt, double& _beta){
    // Perform cooling of the deterministic annealing
    int blockIdx  = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]; // Block number inside grid
    double betafreeze = (1./cParams.Tmin) * sqrt(cParams.coolingFactor); // Last temperature
    while ((*_beta) < betafreeze){ // The cooling loop
      alpaka::syncBlockThreads(acc);
      int nprev = vertices[blockIdx].nV();
      alpaka::syncBlockThreads(acc);
      merge(acc, tracks, vertices, cParams, osumtkwt, _beta);
      alpaka::syncBlockThreads(acc);
      while (nprev !=  vertices[blockIdx].nV() ) { // If we are here, we merged before, keep merging until stable
        nprev = vertices[blockIdx].nV();
	alpaka::syncBlockThreads(acc);
	update(acc, tracks, vertices, cParams, osumtkwt, _beta, 0.0, false); // Update positions after merge
	alpaka::syncBlockThreads(acc);
	merge(acc, tracks, vertices, cParams, osumtkwt, _beta);
	alpaka::syncBlockThreads(acc);
      } // end while after merging
      split(acc, tracks, vertices, cParams, osumtkwt, _beta); // As we are close to a critical temperature, check if we need to split and if so, do it
      alpaka::syncBlockThreads(acc);
      if (once_per_block(acc)){ // Cool down
	(*_beta) = (*_beta)/cParams.coolingFactor;
      }
      alpaka::syncBlockThreads(acc);
      thermalize(acc, tracks, vertices, cParams, osumtkwt, _beta, cParams.delta_highT, 0.0); // Stabilize positions after cooling
      alpaka::syncBlockThreads(acc);
      set_vtx_range(acc, tracks, vertices, cParams, osumtkwt, _beta); // Reassign tracks to vertex
      alpaka::syncBlockThreads(acc);
      update(acc, tracks, vertices, cParams, osumtkwt, _beta, 0.0, false); // Last, update positions again
      alpaka::syncBlockThreads(acc);
    }
  } // end coolingWhileSplitting

  template <bool debug = false, typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>> ALPAKA_FN_ACC static void reMergeTracks(const TAcc& acc, portablevertex::TrackDeviceCollection::View tracks, portablevertex::VertexDeviceCollection::View vertices, const portablevertex::ClusterParamsHostCollection::ConstView cParams, double& osumtkwt, double& _beta){
    // After the cooling, we merge any closeby vertices
    int blockIdx  = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]; // Block number inside grid
    int nprev = vertices[blockIdx].nV();
    merge(acc, tracks, vertices, cParams, osumtkwt, _beta);
    while (nprev !=  vertices[blockIdx].nV() ) { // If we are here, we merged before, keep merging until stable
      set_vtx_range(acc, tracks, vertices, cParams, osumtkwt, _beta); // Reassign tracks to vertex
      alpaka::syncBlockThreads(acc);
      update(acc, tracks, vertices, cParams, osumtkwt, _beta, 0.0, false); // Update before any final merge
      alpaka::syncBlockThreads(acc);
      nprev = vertices[blockIdx].nV();
      merge(acc, tracks, vertices, cParams, osumtkwt, _beta);
      alpaka::syncBlockThreads(acc);
    } // end while
  } // end reMergeTracks
  
  template <bool debug = false, typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>> ALPAKA_FN_ACC static void reSplitTracks(const TAcc& acc, portablevertex::TrackDeviceCollection::View tracks, portablevertex::VertexDeviceCollection::View vertices, const portablevertex::ClusterParamsHostCollection::ConstView cParams, double& osumtkwt, double& _beta){
    // Last splitting at the minimal temperature which is a bit more permissive
    int blockIdx  = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]; // Block number inside grid
    unsigned int ntry = 0; 
    double threshold = 1.0;
    int nprev = vertices[blockIdx].nV();
    split(acc, tracks, vertices, cParams, osumtkwt, _beta);
    while (nprev !=  vertices[blockIdx].nV() ) {
      thermalize(acc, tracks, vertices, cParams, osumtkwt, _beta, cParams.delta_highT, 0.0);
      alpaka::syncBlockThreads(acc);
      nprev = vertices[blockIdx].nV();
      merge(acc, tracks, vertices, cParams, osumtkwt, _beta);
      alpaka::syncBlockThreads(acc);
      while (nprev !=  vertices[blockIdx].nV() ) {
	nprev = vertices[blockIdx].nV();
        update(acc, tracks, vertices, cParams, osumtkwt, _beta, 0.0, false);
	alpaka::syncBlockThreads(acc);
        merge(acc, tracks, vertices, cParams, osumtkwt, _beta);
	alpaka::syncBlockThreads(acc);
      }
      threshold *= 1.1; // Make it a bit easier to split
      split(ntracks, tracks, vertices, params, osumtkwt, beta, threshold);
      alpaka::syncBlockThreads(acc);
    }
  }

  template <bool debug = false, typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>> ALPAKA_FN_ACC static void rejectOutliers(const TAcc& acc, portablevertex::TrackDeviceCollection::View tracks, portablevertex::VertexDeviceCollection::View vertices, const portablevertex::ClusterParamsHostCollection::ConstView cParams, double& osumtkwt, double& _beta){
    // Treat outliers, either low quality vertex, or those with very far away tracks
    int blockIdx  = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]; // Block number inside grid
    double rho0 = 0.0; // Yes, here is where this thing is used
    if (cParams.dzCutOff > 0){
      rho0 = vertices[blockIdx].nV() > 1 ? 1./vertices[blockIdx.x].nV() : 1.;
      for (unsigned int rhoindex = 0; rhoindex < 5 ; rhoindex++){ //Can't be parallelized in any reasonable way
        update(acc, tracks, vertices, cParams, osumtkwt, _beta, rhoindex*rho0/5., false);
        alpaka::syncBlockThreads(acc);
      }
    } // end if
    thermalize(acc, tracks, vertices, cParams, osumtkwt, _beta, cParams.delta_lowT, rho0);
    int nprev = vertices[blockIdx].nV();
    alpaka::syncBlockThreads(acc);
    merge(acc, tracks, vertices, cParams, osumtkwt, _beta);
    alpaka::syncBlockThreads(acc);
    while (nprev !=  vertices[blockIdx].nV()) {
      set_vtx_range(acc, tracks, vertices, cParams, osumtkwt, _beta); // Reassign tracks to vertex
      alpaka::syncBlockThreads(acc);
      update(acc, tracks, vertices, cParams, osumtkwt, _beta, rho0, false); // At rho0 it changes the initial value of the partition function
      alpaka::syncBlockThreads(acc);
      nprev = vertices[blockIdx].nV();
      merge(acc, tracks, vertices, cParams, osumtkwt, _beta);
      alpaka::syncBlockThreads(acc);
    }
    while ((*_beta) < 1./cParams.Tpurge){ // Cool down to purge temperature
      alpaka::syncBlockThreads(acc);
      if (once_per_block(acc)){ // Cool down
        (*_beta) = std::min((*_beta)/cParams.coolingFactor, 1./cParams.Tpurge);
      }
      alpaka::syncBlockThreads(acc);
      thermalize(acc, tracks, vertices, cParams, osumtkwt, _beta, cParams.delta_lowT, rho0);
    }
    alpaka::syncBlockThreads(acc);
    // And now purge
    nprev = vertices[blockIdx].nV();
    purge(acc, tracks, vertices, cParams, osumtkwt, _beta, rho0);
    while (nprev !=  vertices[blockIdx].nV()) {
      thermalize(acc, tracks, vertices, cParams, osumtkwt, _beta, cParams.delta_lowT, rho0);
      nprev = vertices[blockIdx].nV();
      alpaka::syncBlockThreads(acc);
      purge(acc, tracks, vertices, cParams, osumtkwt, _beta, rho0);
      alpaka::syncBlockThreads(acc);
    }
    while ((*_beta) < 1./cParams.Tstop){ // Cool down to stop temperature
      alpaka::syncBlockThreads(acc);
      if (once_per_block(acc)){ // Cool down
        (*_beta) = std::min((*_beta)/cParams.coolingFactor, 1./cParams.Tstop);
      }
      alpaka::syncBlockThreads(acc);
      thermalize(acc, tracks, vertices, cParams, osumtkwt, _beta, cParams.delta_lowT, rho0);
    }
    alpaka::syncBlockThreads(acc);
    // The last track to vertex assignment of the clusterizer!
    set_vtx_range(acc, tracks, vertices, cParams, osumtkwt, _beta);
    alpaka::syncBlockThreads(acc);
  } // rejectOutliers

  template <bool debug = false, typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>> ALPAKA_FN_ACC static void resortVerticesAndAssign(const TAcc& acc, portablevertex::TrackDeviceCollection::View tracks, portablevertex::VertexDeviceCollection::View vertices, const portablevertex::ClusterParamsHostCollection::ConstView cParams){

  }

  template <bool debug = false, typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>> ALPAKA_FN_ACC static void finalizeVertices(const TAcc& acc, portablevertex::TrackDeviceCollection::View tracks, portablevertex::VertexDeviceCollection::View vertices, const portablevertex::ClusterParamsHostCollection::ConstView cParams){

  }

  class clusterizeKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,  portablevertex::TrackDeviceCollection::View tracks, portablevertex::VertexDeviceCollection::View vertices, const portablevertex::ClusterParamsHostCollection::ConstView cParams) const{ 
      // This has the core of the clusterization algorithm
      // First, declare beta=1/T
      double& _beta = alpaka::declareSharedVar<double, __COUNTER__>(acc);
      double& osumtkwt = alpaka::declareSharedVar<double, __COUNTER__>(acc);
      for (int itrack = threadIdx+blockIdx*blockSize; itrack < threadIdx+(blockIdx+1)*blockSize ; itrack += blockSize){ // TODO:Saving and reading in the tracks dataformat might be a bit too much?
        alpaka::atomicAdd(acc, &osumtkwt, tracks[itrack].weight(), alpaka::hierarchy::Threads{});
      }
      alpaka::syncBlockThreads(acc);
      // In each block, initialize to a single vertex with all tracks
      initialize(acc, tracks, vertices, cParams);
      alpaka::syncBlockThreads(acc);
      // First estimation of critical temperature
      getBeta0(acc, tracks, vertices, cParams, _beta);
      alpaka::syncBlockThreads(acc);
      // Cool down to beta0 with rho = 0.0 (no regularization term)
      thermalize(acc, tracks, vertices, cParams, osumtkwt, _beta, cParams.delta_highT(), 0.0);
      alpaka::syncBlockThreads(acc);
      // Now the cooling loop
      coolingWhileSplitting(acc, tracks, vertices, cParams, osumtkwt, _beta);
      alpaka::syncBlockThreads(acc);
      // After cooling, merge closeby vertices
      reMergeTracks(acc,tracks, vertices,cParams, osumtkwt, _beta);
      alpaka::syncBlockThreads(acc);
      // And split those with tension
      reSplitTracks(acc,tracks, vertices,cParams, osumtkwt, _beta);
      alpaka::syncBlockThreads(acc);
      // After splitting we might get some candidates that are very low quality/have very far away tracks
      rejectOutliers(acc,tracks, vertices,cParams, osumtkwt, _beta);
      alpaka::syncBlockThreads(acc);
    }
  }; // class kernel


  class arbitrateKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,  portablevertex::TrackDeviceCollection::View tracks, portablevertex::VertexDeviceCollection::View vertices, const portablevertex::ClusterParamsHostCollection::ConstView cParams) const{
      // This has the core of the clusterization algorithm
      resortVerticesAndAssign(acc, tracks, vertices,cParams);
      alpaka::syncBlockThreads(acc);
      finalizeVertices(acc, tracks, vertices, cParams); // In CUDA it used to be verticesAndClusterize
      alpaka::syncBlockThreads(acc);
    }       
  }; // class kernel


  ClusterizerAlgo::ClusterizerAlgo(Queue& queue) {
  } // ClusterizerAlgo::ClusterizerAlgo
  
  void ClusterizerAlgo::clusterize(Queue& queue, portablevertex::TrackDeviceCollection& deviceTrack, portablevertex::VertexDeviceCollection& deviceVertex, const std::shared_ptr<portablevertex::ClusterParamsHostCollection> cParams, int32_t nBlocks, int32_t blockSize){
    const int blocks = divide_up_by(nBlocks*blockSize, blockSize); //nBlocks of size blockSize
    alpaka::exec<Acc1D>(queue,
		        make_workdiv<Acc1D>(blocks, blockSize),
			clusterizeKernel{},
			deviceTrack.view(), // TODO:: Maybe we can optimize the compiler by not making this const? Tracks would not be modified
			deviceVertex.view(),
			cParams->view()); 
  } // ClusterizerAlgo::clusterize

  void ClusterizerAlgo::arbitrate(Queue& queue, portablevertex::TrackDeviceCollection& deviceTrack, portablevertex::VertexDeviceCollection& deviceVertex, const std::shared_ptr<portablevertex::ClusterParamsHostCollection> cParams, int32_t nBlocks, int32_t blockSize){
    const int blocks = divide_up_by(blockSize, blockSize); //Single block, as it has to converge to a single collection
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(blocks, blockSize),
                        arbitrateKernel{},
                        deviceTrack.view(), // TODO:: Maybe we can optimize the compiler by not making this const? Tracks would not be modified
                        deviceVertex.view(),
                        cParams->view());    
  } // arbitraterAlgo::arbitrate

} // namespace ALPAKA_ACCELERATOR_NAMESPACE
