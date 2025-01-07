#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/VecArray.h"
#include "HeterogeneousCore/AlpakaInterface/interface/blas/alpakaBlasCore.h"
#include "HeterogeneousCore/AlpakaInterface/interface/blas/alpakaBlockReduction.h"
#include "HeterogeneousCore/AlpakaInterface/interface/blas/alpakaReduceResource.h"
#include "HeterogeneousCore/AlpakaInterface/interface/blas/alpakaReducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/blas/alpakaTransformer.h"

using namespace cms::alpakatools;
using namespace ALPAKA_ACCELERATOR_NAMESPACE;

template <typename reduce_t, 
          typename Transformer, 
          typename Reducer, 
          typename DataType,
          uint64_t nSrc = 1,
          bool site_unroll_ = false 
          >
struct AxpyNorm : public blas::TransformReduceFunctor<reduce_t, Transformer, Reducer, site_unroll_> {

  using TR = blas::TransformReduceFunctor<reduce_t, Transformer, Reducer>;
  
  using transformer_t = typename TR::transformer_t;
  using reducer_t     = typename TR::reducer_t;  

  const DataType a;  
  
  AxpyNorm(const DataType &a, const DataType &, transformer_t t, reducer_t r) : blas::TransformReduceFunctor<reduce_t, Transformer, Reducer, site_unroll_>(t, r), a(a) {} 
  AxpyNorm(const DataType &a, const DataType &) 
        : blas::TransformReduceFunctor<reduce_t, Transformer, Reducer, site_unroll_>(Transformer(), Reducer()) 
        , a(a) 
        {}      
  
  template< typename TAcc, typename TDataXZ, typename TDataYW > 
  ALPAKA_FN_ACC inline reduce_t transform(TAcc const& acc, const cms::alpakatools::VecArray<TDataXZ*, nSrc> &x, 
                                                                 cms::alpakatools::VecArray<TDataYW*, nSrc> &y,
                                                                 cms::alpakatools::VecArray<TDataYW*, nSrc> &,
					                   const cms::alpakatools::VecArray<TDataXZ*, nSrc> &,                                                                 
                                                                 int i, 
                                                                 int j, 
                                                                 int k) {
    reduce_t res = cms::alpakatools::reduce::zero<reduce_t>();    
    //
    auto const srcIdx = k;

    y[srcIdx][i] = TR::transformer(a, x[srcIdx][i], y[srcIdx][i]);
    //
    auto const t = y[srcIdx][i] * y[srcIdx][i];     
    //      
    res = TR::reducer(res, t);       

    return res;
  }
  // not needed...
  template< typename TAcc >  
  ALPAKA_FN_ACC inline reduce_t reduce(TAcc const& acc, const reduce_t &x, const reduce_t &y ) const { return TR::reducer(x, y);  }  
 
  //constexpr int flops() const { return 4; }   //! flops per element
}; 

template<typename TAcc, typename TDevAcc, typename TransfromReducer>
class AxpyNormProductKernel
{
  public:
  
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TransfromReducer tr) const 
        -> void
    {
        // Thread/Block idx 
	auto const batch_idx = static_cast<uint32_t>( alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0] );//the slowest index
        tr.apply(acc, batch_idx);
    }

};


int main() {
  // get the list of devices on the current platform
  auto const& devices = cms::alpakatools::devices<Platform>();

  if (devices.empty()) {
    std::cerr << "No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
      "the test will be skipped.\n";
    exit(EXIT_FAILURE);
  }
  
  static const Idx N     = 1 << 8;
  //
  static constexpr Idx nSrc  = 4;    
  //
  std::cout << "N : " << N << std::endl;
  //
  using DataType = double;
    
  using reduce_t = DataType;  

  for (auto const& device : devices) {
    std::cout << "Test nsrc kernels on " << alpaka::getName(device) << '\n';
    
    // Select specific devices
    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost      = alpaka::getDevByIdx(platformHost, 0);
    auto const platformAcc  = alpaka::Platform<Acc3D>{};    
    
    auto computeQueue = Queue(device); // TODO: cannot be const due to unsupported combination for memcpy (specialization does not exist in Alpaka?)   
    
    if ( alpaka::getAccName<Acc3D>() == "AccCpuSerial<3,unsigned int>" ) {
      std::cout << "... skipped" << std::endl;	    
      return EXIT_SUCCESS;
    }    
    
    // get suitable device for this Acc
    auto const devAcc   = alpaka::getDevByIdx(platformAcc, 0);

    // Select queue
    using QueueProperty = alpaka::NonBlocking;
    using QueueAcc      = std::remove_cvref_t<decltype(computeQueue)>;//alpaka::Queue<Acc3D, QueueProperty>;
    //
    using reducer_t     = cms::alpakatools::reduce::plus<reduce_t>;
    using transformer_t = cms::alpakatools::transform::axpy<DataType>;     

    // Define the 1D extent (dimensions : e.g., phys volume)
    Vec1D const extent(static_cast<Idx>(N));    
    //
    using HostViewType = decltype(alpaka::createView(
      std::declval<decltype(devHost)>(),
      std::declval<DataType*>(),
      std::declval<Vec1D const&>()
    ));
    // Input vector allocation and copy to device buffer
    //
    const DataType a = 2.f;
    
    std::vector<DataType> xv(extent.prod(), 0.f);
    //
    std::vector<DataType> yv(extent.prod(), 0.f);   
        
    // Use increasing values as input
    //
    HostViewType xView = alpaka::createView(devHost, xv.data(), extent);//create 2D view 
    //
    HostViewType yView = alpaka::createView(devHost, yv.data(), extent);//create 2D view       

    // Input buffer at device
    using Buf_t = alpaka::Buf<Acc3D, DataType, Dim1D, Idx>;
    
    std::vector<Buf_t> xAcc;  xAcc.reserve(nSrc);
    std::vector<Buf_t> yAcc;  yAcc.reserve(nSrc);    
    
    // Initialize random number generator
    std::random_device rd; 
    std::mt19937 gen(rd()); 

    // Define a normal (Gaussian) distribution with mean 0 and standard deviation 1
    std::normal_distribution<double> distr(0.0, 1.0);   
    
    constexpr bool use_random_nums = true;

    for(Idx i = 0; i < nSrc; i++) {
       // populate the vector with random numbers
      if constexpr (use_random_nums) {
        std::generate(xv.begin(), xv.end(), [&]() { return distr(gen); });
        //
        std::generate(yv.begin(), yv.end(), [&]() { return distr(gen); });
      } else {	// use int sequence (and shuffle)
        //
        std::iota(xv.begin(), xv.end(), 1.0f); 
        std::iota(yv.begin(), yv.end(), 2.0f); 
        //	
        std::shuffle(xv.begin(), xv.end(), gen);
        std::shuffle(yv.begin(), yv.end(), gen);
      }       
      //
      xAcc.push_back(alpaka::allocBuf<DataType, Idx>(devAcc, extent));
      //
      yAcc.push_back(alpaka::allocBuf<DataType, Idx>(devAcc, extent));
      //
      alpaka::memcpy(computeQueue, xAcc[i], xView);
      //
      alpaka::memcpy(computeQueue, yAcc[i], yView);
      //
      double gres = 0.0;
    
      auto axpy = transformer_t();
      auto r    = reducer_t();

      for(size_t j{0}; j < N; ++j) {
        yView[j] = axpy(a, xView[j], yView[j]);
        auto const t = yView[j]*yView[j];           
        gres = r(gres, t);  
      }
   
      std::cout << "Result on the host :: (nSrc " << i << " ) " << std::setprecision(16)
                      << gres << std::endl;       
    }
       
    alpaka::wait(computeQueue);  
      
    // Simulate
    constexpr Idx block_x_dim = 256 / nSrc;//tuning needed! 
    
    Vec3D const grid_size{1, 1, (N / block_x_dim)};
    Vec3D const block_size{nSrc, 1, block_x_dim};

    std::cout << "GRID size : " << grid_size << std::endl; 
    std::cout << "BLCK size : " << block_size << std::endl;   
    //
    //
    const Idx n_blocks  = block_size[0]*grid_size[2];//nSrc * blockDimX
    
    auto max_reduce_blocks = 2 * alpaka::getAccDevProps<Acc3D>(devAcc).m_multiProcessorCount;//only 2 blocks per MP are active    
    
    std::cout << "NBlocks :: " << n_blocks << " : " << max_reduce_blocks << std::endl;
    //
    using func_t        = AxpyNorm<reduce_t, transformer_t, reducer_t, DataType, nSrc, false>;

    std::cout << "Running improved test..." << std::endl;

    auto reduce_bufs = cms::alpakatools::reduce::create_reduction_resources<DataType, Acc3D, decltype(devAcc), QueueAcc>(devAcc, computeQueue, nSrc);    

    auto msrc_axpyNorm_functor = blas::instantiateTransformReducer<Acc3D,
                                                                   decltype(devAcc),
                                                                   QueueAcc,
                                                                   Buf_t,
                                                                   Buf_t,
                                                                   reduce_t,
                                                                   DataType,
                                                                   func_t,
                                                                   nSrc, false > (devAcc, computeQueue, a, a, xAcc, yAcc, yAcc, xAcc, reduce_bufs);


    AxpyNormProductKernel<Acc3D, decltype(devAcc), decltype(msrc_axpyNorm_functor)> axpyNormProduct;

    alpaka::WorkDivMembers<Dim3D, Idx> workDiv{grid_size, block_size, Vec3D::ones()};

    alpaka::exec<Acc3D>(
        computeQueue,
        workDiv,
        axpyNormProduct,
        msrc_axpyNorm_functor
        );

    // Copy device -> host
    alpaka::wait(computeQueue);
    //
    std::vector<DataType> check_yv(extent.prod(), 0.f);
    
    auto check_yView = alpaka::createView(devHost, check_yv.data(), extent);    
     
    alpaka::memcpy(computeQueue, check_yView, yAcc[nSrc-1], extent);    

    msrc_axpyNorm_functor.template fetch<QueueAcc>(computeQueue);    
    
    alpaka::wait(computeQueue);
    
    auto host_values = msrc_axpyNorm_functor.host_reduced_values();
    
    double gres = 0.0;
    
    auto r  = reducer_t();

    for(size_t j{0}; j < N; ++j) { //dof 
      auto const t = yView[j] * yView[j]; 
      gres = r(t, gres);  
    }
   
    std::cout << "Result on the host :: " << std::setprecision(16)
                      << gres << std::endl;       
    //
    for(size_t i{0}; i < nSrc; ++i) {
      std::cout << "CHECK output : " << std::setprecision(16) << host_values[i] << std::endl;
      //
      alpaka::memcpy(computeQueue, yView, yAcc[i]);
      alpaka::wait(computeQueue);
      //
      double gnrm = 0.0;

      for(size_t j{0}; j < N; ++j) { //dof
        gnrm += (yView[j] * yView[j]);
      }
   
      std::cout << "NORM on the host :: " << std::setprecision(16)
                      << gnrm << std::endl;      
      
    }
    //  Print results
    std::cout << "Multisrc reduction kernel.\n";
    std::cout << "Vector Size:" << N << "x" << ", src number:" << nSrc << "\n";

    std::cout << "Sampled result checks are correct!\n";
  }

  return EXIT_SUCCESS;
}
