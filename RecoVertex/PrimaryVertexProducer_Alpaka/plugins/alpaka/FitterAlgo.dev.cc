#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoVertex/PrimaryVertexProducer_Alpaka/plugins/alpaka/FitterAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace cms::alpakatools; 

  class fitVertices {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,  const portablevertex::TrackDeviceCollection::ConstView tracks, portablevertex::VertexDeviceCollection::View vertices, const portablevertex::BeamSpotDeviceCollection::ConstView beamSpot, bool* useBeamSpotConstraint) const{
      // These are the kernel operations themselves
      const unsigned int nTrueVertex = vertices[0].nV(); // Set max true vertex
      // Magic numbers from https://github.com/cms-sw/cmssw/blob/master/RecoVertex/PrimaryVertexProducer/interface/WeightedMeanFitter.h#L12
      const float precision = 1e-24;
      float corr_x = 1.2;
      const float corr_z = 1.4;
      const int maxIterations = 50;
      const float muSquare = 9.;
      // BeamSpot coordinates are initialized to 0, if we use beamSpot, we change them
      double bserrx = 0;
      double bserry = 0;
      double bsx = 0;
      double bsy = 0;
      if (*useBeamSpotConstraint){
        bserrx = beamSpot.sx()*beamSpot.sx();
        bserry = beamSpot.sy()*beamSpot.sy();
        bsx    = beamSpot.x();
        bsy    = beamSpot.y();
	corr_x = 1.0;
      }

      for (auto i : elements_with_stride(acc, nTrueVertex) ) { // By construction nTrueVertex <= 512, so this is just a 1 thread to 1 vertex assignment
        if (not(vertices[i].isGood())) continue; // If vertex was killed before, just skip
        // Initialize positions and errors to 0
        double x = 0;
        double y = 0;
        double z = 0;
        double errx = 0;
        double errz = 0;

	for (unsigned int itrackInVertex = 0; itrackInVertex < vertices[i].ntracks(); itrackInVertex++){
	  unsigned int itrack = vertices[i].track_id()[itrackInVertex];
	  double wxy = tracks[itrack].dxy2();
	  double wz  = tracks[itrack].dz2();
	  x += tracks[itrack].x()*wxy;
	  y += tracks[itrack].y()*wxy;
	  z += tracks[itrack].z()*wz;
	  errx += wxy; // x and y have the same error due to symmetry
	  errz += wz;
	}
        double erry = errx;

        // Now add the BeamSpot and get first estimation, if no beamspot, this changes nothing
	x = (x + bsx*bserrx)/(bserrx + errx);
	y = (y + bsy*bserry)/(bserry + erry);
	z /= errz;

        // Weights and square weights for iteration	
        double s_wx, s_wz;
	int ndof;
	// Run iterative weighted mean fitter
	unsigned int niter = 0;
	while ((niter++) < maxIterations){
          double old_x = x;
	  double old_y = y;
	  double old_z = z;
	  s_wx = 0.;
	  s_wz = 0.;
	  x = 0.;
	  y = 0.;
	  z = 0.;
	  ndof = 0;
	  for (unsigned int itrackInVertex = 0; itrackInVertex < vertices[i].ntracks(); itrackInVertex++){
            unsigned int itrack = vertices[i].track_id()[itrackInVertex];
            // Position (ref point) of the track
	    double tx = tracks[itrack].x();
	    double ty = tracks[itrack].y();
	    double tz = tracks[itrack].z();
	    // Momentum of the track
            double px = tracks[itrack].px();
            double py = tracks[itrack].py();
            double pz = tracks[itrack].pz();
	    // Compute the PCA of the track to the current vertex
	    double pnorm2 = px*px+py*py+pz*pz;
	    // This is the 'time' needed to move from the ref point to the PCA scalar product of (x_v-x_t), p_t over magnitude squared of p_t
	    double t = (px*(old_x-tx)+py*(old_y-ty)+pz*(old_z-tz))/pnorm2;
            // Advance the track until the PCA
	    tx += px*t;
	    ty += py*t;
	    tz += pz*t;
            double wx = tracks[itrack].dxy2();
	    double wz = tracks[itrack].dz();
	    if (((tx-old_x)*(tx-old_x)/(wx+errx) < muSquare) && ((ty-old_y)*(ty-old_y)/(wx+erry) < muSquare) && ((ty-old_y)*(tx-old_y)/(wz+errz) < muSquare)){ // I.e., old coordinates of PCA are within 3 sigma of current vertex position, keep the track
	      ndof += 1;
	      vertices[i].track_weight()[itrackInVertex] = 1;
	      wx = 1./wx;
	      wz = 1./wz;
	      s_wx += wx;
	      s_wz += wz;
	    }
	    else{ // Otherwise, discard track
              vertices[i].track_weight()[itrackInVertex] = 0;
	      wx = 0.;
	      wz = 0.;
	    }
	    // Here, will only change if track is within 3 sigma
            x += tx*wx;
	    y += ty*wx;
	    z += tz*wz;
	  } // end for
	  // After all tracks, add BS uncertainties, will do nothing if not used
          x += bsx*bserrx;
          y += bsy*bserry;
	  s_wx += errx;
        
	  x /= s_wx;
	  y /= s_wx;
	  z /= s_wz;
    	  errx = 1/s_wx;
	  errz = 1/s_wz;
	  erry = errx;
	  if ((abs(old_x-x) < precision) && (abs(old_y-y) < precision) && (abs(old_z-z) < precision)) break; // If good enough, stop the iterations
        } // end while 
        // Assign everything back in global memory to get the fitted vertex!
        errx *= corr_x*corr_x;
        errz *= corr_z*corr_z;
        vertices[i].x() = x;
        vertices[i].y() = y;
        vertices[i].z() = z;
        vertices[i].errx() = errx;
        vertices[i].erry() = errx;
        vertices[i].errz() = errz;
        vertices[i].ndof() = ndof;
        // Last get the degrees of freedom of the final vertex fit 
        double chi2 = 0.;
        for (unsigned int itrackInVertex = 0; itrackInVertex < vertices[i].ntracks(); itrackInVertex++){
          unsigned int itrack = vertices[i].track_id()[itrackInVertex];
          // Position (ref point) of the track
          double tx = tracks[itrack].x();
          double ty = tracks[itrack].y();
          double tz = tracks[itrack].z();
          double wx = tracks[itrack].dxy2();
          double wz = tracks[itrack].dz();
          chi2 += ((tx-x)*(tx-x)+ (ty-y)*(ty-y))/(errx+wx) + (tz-z)*(tz-z)/(errz+wz); // chi2 doesn't use the PCA distance, but the ref point coordinates as in https://github.com/cms-sw/cmssw/blob/master/RecoVertex/PrimaryVertexProducer/interface/WeightedMeanFitter.h#L316
        } // end for
      vertices[i].chi2() = chi2;
      } // end for (stride) loop
    } // operator()
  }; // class fitVertices

  FitterAlgo::FitterAlgo(Queue& queue, const uint32_t nV, fitterParameters fPar) : useBeamSpotConstraint(cms::alpakatools::make_device_buffer<bool>(queue)) {
    // Set fitter parameters
    alpaka::memset(queue,  useBeamSpotConstraint, fPar.useBeamSpotConstraint);
  } // FitterAlgo::FitterAlgo
  
  void FitterAlgo::fit(Queue& queue, const portablevertex::TrackDeviceCollection& deviceTrack, portablevertex::VertexDeviceCollection& deviceVertex, const portablevertex::BeamSpotDeviceCollection& deviceBeamSpot){
    const int nVertexToFit = 512; // TODO:: Right now it executes for all 512 vertex, even if vertex collection is empty (in which case the kernel passes). Can we make this dynamic to vertex size?
    const int threadsPerBlock = 512;
    const int blocks = divide_up_by(nVertexToFit, threadsPerBlock);
    alpaka::exec<Acc1D>(queue,
		        make_workdiv<Acc1D>(blocks, threadsPerBlock),
			fitVertices{},
			deviceTrack.view(), // TODO:: Maybe we can optimize the compiler by not making this const? Tracks would not be modified
			deviceVertex.view(),
			deviceBeamSpot.view(), // TODO:: Same as for tracks
			useBeamSpotConstraint.data()); 
  } // FitterAlgo::fit
} // namespace ALPAKA_ACCELERATOR_NAMESPACE
