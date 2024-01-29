#ifndef DataFormats_PortableVertex_interface_VertexSoA_h
#define DataFormats_PortableVertex_interface_VertexSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

#include <Eigen/Core>
#include <Eigen/Dense>

namespace portablevertex {

  using VertexToTrack = Eigen::Vector<double, 1024>; // 512 is the max vertex allowed
  using VertexToTrackInt = Eigen::Vector<unsigned int, 1024>;
  // SoA layout with x, y, z, id fields
  GENERATE_SOA_LAYOUT(VertexSoALayout,
                      // columns: one value per element
                      SOA_COLUMN(double, x),
                      SOA_COLUMN(double, y),
                      SOA_COLUMN(double, z),
                      SOA_COLUMN(double, t),

                      SOA_COLUMN(double, errx),
                      SOA_COLUMN(double, erry),
                      SOA_COLUMN(double, errz),
                      SOA_COLUMN(double, errt),

                      SOA_COLUMN(double, chi2),
                      SOA_COLUMN(double, ndof),
                      SOA_COLUMN(unsigned int, ntracks),
                      SOA_COLUMN(double, rho),

                      SOA_COLUMN(double, aux1),
                      SOA_COLUMN(double, aux2),

                      SOA_EIGEN_COLUMN(VertexToTrackInt, track_id),
                      SOA_EIGEN_COLUMN(VertexToTrack, track_weight),

                      SOA_COLUMN(bool, isGood),
                      SOA_COLUMN(unsigned int, order),


                      SOA_COLUMN(double, sw),
                      SOA_COLUMN(double, se),
                      SOA_COLUMN(double, swz),
                      SOA_COLUMN(double, swE),
                      SOA_COLUMN(double, exp),
                      SOA_COLUMN(double, exparg),


                      // scalars: one value for the whole structure
                      SOA_COLUMN(uint32_t, nV))

  using VertexSoA = VertexSoALayout<>;


  using TrackToVertex = Eigen::Vector<double, 512>; // 512 is the max vertex allowed
  GENERATE_SOA_LAYOUT(TrackSoALayout,
                      // columns: one value per element
                      SOA_COLUMN(double, dxy2),
                      SOA_COLUMN(double, dxy2AtIP),
                      SOA_COLUMN(double, dz2),
                      SOA_COLUMN(double, oneoverdz2),
                      SOA_COLUMN(double, weight),
                      SOA_COLUMN(double, sum_Z),
                      SOA_COLUMN(unsigned int, kmin),
                      SOA_COLUMN(unsigned int, kmax),
                      SOA_COLUMN(bool, isGood),
                      SOA_COLUMN(unsigned int, order),
                      SOA_COLUMN(unsigned int, tt_index),

                      SOA_COLUMN(double, x),
                      SOA_COLUMN(double, y),
                      SOA_COLUMN(double, z),

                      SOA_COLUMN(double, xAtIP),
                      SOA_COLUMN(double, yAtIP),

                      SOA_COLUMN(double, dx),
                      SOA_COLUMN(double, dy),
                      SOA_COLUMN(double, dz),

                      SOA_COLUMN(double, dxError),
                      SOA_COLUMN(double, dyError),
                      SOA_COLUMN(double, dzError),

                      SOA_COLUMN(double, px),
                      SOA_COLUMN(double, py),
                      SOA_COLUMN(double, pz),

                      SOA_COLUMN(double, aux1),
                      SOA_COLUMN(double, aux2),

                      // Track-vertex association
                      SOA_EIGEN_COLUMN(TrackToVertex, vert_sw),
                      SOA_EIGEN_COLUMN(TrackToVertex, vert_se),
                      SOA_EIGEN_COLUMN(TrackToVertex, vert_swz),
                      SOA_EIGEN_COLUMN(TrackToVertex, vert_swE),
                      SOA_EIGEN_COLUMN(TrackToVertex, vert_exp),
                      SOA_EIGEN_COLUMN(TrackToVertex, vert_exparg),

                      // scalars: one value for the whole structure
                      SOA_SCALAR(uint32_t, nT),
                      SOA_SCALAR(double, totweight))

  using TrackSoA = TrackSoALayout<>;

  GENERATE_SOA_LAYOUT(BeamSpotSoALayout,
		      SOA_SCALAR(double, x),
		      SOA_SCALAR(double, y),
		      SOA_SCALAR(double, sx),
		      SOA_SCALAR(double, sy))

  using BeamSpotSoA = BeamSpotSoALayout<>;

  GENERATE_SOA_LAYOUT(ClusterParams,
		      SOA_SCALAR(double, d0CutOff),
		      SOA_SCALAR(double, TMin),
		      SOA_SCALAR(double, delta_lowT),
		      SOA_SCALAR(double, zmerge),
                      SOA_SCALAR(double, dzCutOff),
                      SOA_SCALAR(double, Tpurge),
                      SOA_SCALAR(int, convergence_mode),
                      SOA_SCALAR(double, delta_highT),
                      SOA_SCALAR(double, Tstop),
                      SOA_SCALAR(double, coolingFactor),
                      SOA_SCALAR(double, vertexSize),
                      SOA_SCALAR(double, uniquetrkweight),
                      SOA_SCALAR(double, uniquetrkminp),
                      SOA_SCALAR(double, zrange))

  using ClusterParamsSoA = ClusterParams<>;
		      
}  // namespace portablevertex

#endif  // DataFormats_PortableVertex_interface_VertexSoA_h
