#ifndef PFClusterProducer_plugins_alpaka_PFMultiDepthECLCCEpilogue_h
#define PFClusterProducer_plugins_alpaka_PFMultiDepthECLCCEpilogue_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/VecArray.h"

#include "HeterogeneousCore/AlpakaMath/interface/deltaPhi.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringEdgeVarsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringVarsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusterWarpIntrinsics.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusterizerHelper.h"

/**
 * @file PFMultiDepthECLCCEpilogue.h
 * @brief Warp-based postprocessing kernel for multi-depth particle flow clustering (ECL-CC Epilogue).
 *
 * This header defines and implements an Alpaka GPU kernel that finalizes the clustering of
 * particle flow clusters after connected components (ECL-CC) detection.
 *
 * The kernel performs:
 * - Consolidation of connected component membership for each cluster.
 * - Assignment of component energy sums based on rechit fractions.
 * - Remapping of cluster indices to component indices.
 * - Masked warp-scope reductions to ensure efficient and divergence-free operations.
 *
 * Key outputs:
 * - mdpf_component()       : representative vertex index per cluster.
 * - mdpf_componentEnergy() : total rechit energy for each component.
 * - mdpf_componentIndex()  : compressed component index for final sorting.
 *
 * - Warp-masked ballot, shuffle, and scan operations are used throughout.
 * - Shared memory usage depends on max_w_items ensure adequate resource sizing.
 * - Only a single block (group == 0) is active during execution.
 * 
 * Ensure consistency between Prologue (adjacency construction) and Epilogue (component labeling) stages.
 *
 */

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

/**
 * @class ECLCCEpilogueKernel
 * @brief Finalizes cluster component information after ECL-CC labeling.
 *
 * The ECLCCEpilogueKernel aggregates information about particle flow clusters
 * after they have been linked into connected components by the ECL-CC algorithm.
 * 
 * Responsibilities:
 * - Calculate total rechit energy per component.
 * - Map cluster vertices to their connected component representatives.
 * - Assign compressed component indices for further processing.
 * 
 * - Warp-masked operations are used throughout to eliminate divergence.
 * - Component aggregation and rechit assignment use warp-level masked scans.
 * 
 * @tparam max_w_items Maximum number of warp tiles processed per block () controls shared memory footprint ).
 *
 */


  template<unsigned int max_w_items = 32>
  class ECLCCEpilogueKernel {
    public:

      template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>> 
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    reco::PFMultiDepthClusteringVarsDeviceCollection::View pfClusteringVars,
                                    const reco::PFRecHitDeviceCollection::ConstView pfRecHits
                                    ) const {
        static_assert(max_w_items <= 32, 
                      "ECLCCEpilogueKernel: Maximum number of supported warps per block is 32, "
                      "assuming one warp per 32 threads." );
        constexpr unsigned int max_w_extent = 32;                   
	//
        const unsigned int nVertices = pfClusteringVars.size();
        //
        const unsigned int nBlocks = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u]; //
        //
        const unsigned int w_extent = alpaka::warp::getSize(acc);  
        const unsigned int w_items  = alpaka::math::min(acc, nBlocks / w_extent, max_w_items ); 
        //
        auto& intern_connected_comp_masks(alpaka::declareSharedVar<::cms::alpakatools::VecArray<int, max_w_items * max_w_extent>, __COUNTER__>(acc));
        auto& extern_connected_comp_masks(alpaka::declareSharedVar<::cms::alpakatools::VecArray<int, max_w_items * max_w_extent>, __COUNTER__>(acc));
        //
        auto& component_roots(alpaka::declareSharedVar<::cms::alpakatools::VecArray<int, max_w_items * max_w_extent + 1>, __COUNTER__>(acc));
        auto& connected_comp_buffer(alpaka::declareSharedVar<::cms::alpakatools::VecArray<int, max_w_items * max_w_extent+1>, __COUNTER__>(acc));
        // 
        auto& connected_comp_offsets = connected_comp_buffer; 
        auto& connected_comp_sizes   = connected_comp_buffer;
        //
        auto& comp_offsets = component_roots; 
        
        auto& component_cluster_seeds(alpaka::declareSharedVar<::cms::alpakatools::VecArray<float, max_w_items * max_w_extent>, __COUNTER__>(acc));
        auto& component_cluster_energies(alpaka::declareSharedVar<::cms::alpakatools::VecArray<float, max_w_items * max_w_extent>, __COUNTER__>(acc)); 
        
        auto& component_vertex_rhf_offsets(alpaka::declareSharedVar<::cms::alpakatools::VecArray<int, max_w_items * max_w_extent>, __COUNTER__>(acc));
        auto& component_vertex_rhf_sizes(alpaka::declareSharedVar<::cms::alpakatools::VecArray<int, max_w_items * max_w_extent>, __COUNTER__>(acc));  
        //
        auto& connected_comp_vertices = component_vertex_rhf_offsets;
        auto& connected_comp_pos      = component_vertex_rhf_sizes;
        //
        auto& subdomain_offsets(alpaka::declareSharedVar<::cms::alpakatools::VecArray<unsigned int, max_w_items>, __COUNTER__>(acc));
        //
        // Setup all shared mem buffers:
        //
        for ( auto group : ::cms::alpakatools::uniform_groups(acc) ) {//loop over thread blocks
          // Skip inactive groups:
          if(group != 0) continue;
          // Init shared_buffer   
          for( auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent)) ) { 
            const auto warp_idx   = idx.local / w_extent;
	    // Reset shared memory buffers to zero:
            component_roots[ idx.local ]       = -1;
            connected_comp_buffer[ idx.local ] = 0;
            //
            intern_connected_comp_masks[ idx.local ]  = 0x0;
            extern_connected_comp_masks[ idx.local ]  = 0x0;
            //
            component_cluster_seeds[ idx.local ]      = 0.0f;
            component_cluster_energies[ idx.local ]   = 0.0f;
            //
            component_vertex_rhf_offsets[ idx.local ] = 0;
            component_vertex_rhf_sizes[ idx.local ]   = 0;
            //
            if (warp_idx == 0) subdomain_offsets[ idx.local ] = 0;       
          } 
          //
          alpaka::syncBlockThreads(acc);
          // Identify all neigbors:
          for( auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent)) ) { 
            //
            const unsigned int active_lanes_mask = alpaka::warp::ballot(acc, idx.local < nVertices);
            // Skip inactive lanes:
            if(idx.local >= nVertices) continue;
            //
            const auto vertex_idx = idx.local; 
            //
            //const auto warp_idx   = idx.local / w_extent;
            const auto lane_idx   = idx.local % w_extent;
            //
            const unsigned int rep_idx = pfClusteringVars[vertex_idx].mdpf_topoId();  
            //
            component_roots[vertex_idx] = rep_idx;
            //
            component_cluster_seeds[ idx.local ]      = pfClusteringVars[vertex_idx].seedRHIdx();
            //
            component_vertex_rhf_offsets[ idx.local ] = pfClusteringVars[vertex_idx].rhfracOffset();
            component_vertex_rhf_sizes[ idx.local ]   = pfClusteringVars[vertex_idx].rhfracSize();

            // Find out as to whether the current lane holds the representative vertex:
            const bool is_warp_local_representative = vertex_idx == rep_idx; 
            // Find out how many vertices in the warp connected to a given representative:
            warp::syncWarpThreads_mask(acc, active_lanes_mask);

            unsigned int component_mask = warp::match_any_mask(acc, active_lanes_mask, rep_idx);
            // Compute number of such vertices. Note that intern_component_size is always > 0 
            // since each vertex locally at least self-connected, that is, it can be locally isolated. 
            const unsigned int component_size = alpaka::popcount(acc, component_mask);
            // Define a master lane for each sub-component, note that the lane which holds the root is 
            // always selected as master, otherwise choose the lane with the lowest index.
            // Note that, by construction, if a vertex happened to be the local reprentaitve, it has always the lowest
            // lane index.   
            const unsigned int master_lane_idx = get_ls1b_idx(acc, component_mask); 
            //
            warp::syncWarpThreads_mask(acc, active_lanes_mask);
            // Store internal/external component masks in the shared memory
            if ( master_lane_idx == lane_idx ) {
              if (is_warp_local_representative) { 
                // no race condition here since the operation warp-local
                connected_comp_sizes[ rep_idx ] = component_size;
                intern_connected_comp_masks[ rep_idx ] = component_mask;
              } else { 
                // no race condition                  
                extern_connected_comp_masks[ vertex_idx ] = component_mask;
              }
            }   
          }
          //
          alpaka::syncBlockThreads(acc);
          //

          for( auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent)) ) { 
            // Skip inactive lanes:
            if(idx.local >= nVertices) continue;
            //
            const auto vertex_idx = idx.local; 
            //
            //const auto warp_idx   = idx.local / w_extent;
            //const auto lane_idx   = idx.local % w_extent;
            //
            const unsigned int component_mask = extern_connected_comp_masks[vertex_idx];
            //
            const int component_size = alpaka::popcount(acc, component_mask);
            //
            if(component_size == 0) continue;
            //
            const unsigned int rep_idx = component_roots[ vertex_idx ];
            alpaka::atomicAdd(acc, &connected_comp_sizes[rep_idx], component_size, alpaka::hierarchy::Threads{});   
          }
          //
          alpaka::syncBlockThreads(acc);
          //
          for( auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent)) ) {         
            //
            const auto active_lanes_mask = alpaka::warp::ballot(acc, true);
            // 
            const auto warp_idx   = idx.local / w_extent;
            const auto lane_idx   = idx.local % w_extent;
            //
            const unsigned int component_size = connected_comp_sizes[idx.local];
            // Note that component_size = 0 for idx.local >= nVertices and connected child vertices,
            // only represntatives hold a non-zero value.
            const unsigned int valid_lanes_mask = warp::ballot_mask(acc, active_lanes_mask, component_size > 0);
            //
            warp::syncWarpThreads_mask(acc, active_lanes_mask);
            // Warp-uniform operation. Note that we exclude a trivial case 
            // (when valid_lanes_mask = 0x0, that is, when all vertices processed by a warp are connected to external representatives)
            //const auto  local_warp_offset = valid_lanes_mask != 0x0 ? warp_exclusive_sum(acc, active_lanes_mask, component_size, lane_idx) : 0;
            const auto  local_warp_offset = warp_exclusive_sum(acc, valid_lanes_mask, component_size, lane_idx);
            // Store warp offsets in a separate buffer:
            if(lane_idx == 0 and valid_lanes_mask != 0x0) subdomain_offsets[warp_idx] = local_warp_offset;
            // Store local offsets (only for valid lanes, otherwise set to -1) :
            //connected_comp_offsets[idx.local] = lane_idx > 0 ? local_warp_offset : 0;
            if (component_size > 0) {
              const auto low_lane_idx = get_ls1b_idx(acc, valid_lanes_mask);
              connected_comp_offsets[idx.local] = lane_idx != low_lane_idx ? local_warp_offset : 0;
            } else {
              connected_comp_offsets[idx.local] = -1;
            }
          }
          //
          alpaka::syncBlockThreads(acc);
          //
          for( auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent)) ) {
            //
            const auto active_lanes_mask = alpaka::warp::ballot(acc, true); 
            //
            const auto warp_idx   = idx.local / w_extent;
            // Skip inactive warps:
            if (warp_idx != 0) continue; 
            //
            const auto warp_content_size = subdomain_offsets[idx.local];
            //
            warp::syncWarpThreads_mask(acc, active_lanes_mask);
            const auto warp_offset = warp_exclusive_sum(acc, active_lanes_mask, warp_content_size, idx.local);

            subdomain_offsets[idx.local] = warp_offset;//NOTE: lane 0 get total nnz
          }
          //
          alpaka::syncBlockThreads(acc);
          //
          for( auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent)) ) { 
            //
            const unsigned int active_lanes_mask = alpaka::warp::ballot(acc, idx.local < nVertices);
            // Skip inactive lanes:
            if(idx.local >= nVertices) continue;
            // 
            const auto warp_idx   = idx.local / w_extent;
            const auto lane_idx   = idx.local % w_extent; 
            //
            const unsigned warp_offset = lane_idx == 0 and warp_idx > 0 ? subdomain_offsets[warp_idx] : 0;
            const int lane_offset = connected_comp_offsets[idx.local]; // -1 for child vertices
            //  
            warp::syncWarpThreads_mask(acc, active_lanes_mask);
            // We need to exclude void lanes (all ones that correponds to offset -1):
            const auto valid_lanes_mask = warp::ballot_mask(acc, active_lanes_mask, lane_offset != -1);
            //
            if (lane_offset == -1) continue;

            // we need to broadcast the global warp offset (for all warps except warp 0):
            const unsigned shift = warp_idx > 0 ? warp::shfl_mask(acc, valid_lanes_mask, warp_offset, 0, w_extent) : 0;
            //
            const unsigned global_offset = lane_offset + shift;
            // We just need to sync threads in the warp, 
            warp::syncWarpThreads_mask(acc, valid_lanes_mask);
            // Store final offsets in shared memory:
            connected_comp_offsets[idx.local] = global_offset;
            // We need this extra step for future cycles:
            const auto low_idx = get_ls1b_idx(acc, valid_lanes_mask);
            if ( low_idx != 0 ) connected_comp_offsets[idx.local - low_idx] = global_offset;
            // Last entry for the total number of components:
            // Note: warp_idx = 0 lane_idx = 0 has always valid offset (vertex id 0 is always the root!)
            // so this lane always valid:
            if (warp_idx == 0 && lane_idx == 0) connected_comp_offsets[nVertices] = subdomain_offsets[0];
          }
          //
          alpaka::syncBlockThreads(acc);
          //
          for( auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent)) ) { 
            //
            unsigned int active_lanes_mask = alpaka::warp::ballot(acc, idx.local <= nVertices);
            // Skip inactive lanes:
            if(idx.local > nVertices) continue;
            //
            auto is_valid_lane  = [](const unsigned int mask, const unsigned int lid) -> bool { 
              return ((mask >> lid) & 1); 
            };
            //
            auto get_high_neighbor_logical_lane_idx = [&](const unsigned int active_mask, const unsigned int custom_mask, const int lid) {
              // Zero out all bits <= lid
              const auto zeroed_lowbit_mask = custom_mask & (active_mask << (lid+1));
              // Just in case if the mask is exactly zero (may happen!):
              return zeroed_lowbit_mask == 0x0 ? lid : get_ls1b_idx(acc, zeroed_lowbit_mask);  // Count 1s below next to current lane
            };           
            // Determin actual warp-level work dimension: it coincides with w_extent for all warps 
            // except (potentially!) the last one:
            const auto warp_work_extent = warp::ballot_mask(acc, active_lanes_mask, true);
            //
            const auto warp_idx   = idx.local / w_extent;
            const auto lane_idx   = idx.local % w_extent;
            //const auto lane_mask  = (1 << lane_idx);
            //
            // Get local coordinate:
            const int begin = connected_comp_offsets[idx.local];
            //
            const auto valid_offsets_mask = warp::ballot_mask(acc, active_lanes_mask, begin != -1);
            //
            warp::syncWarpThreads_mask(acc, active_lanes_mask);
            //
            const auto neigh_lane_idx = is_valid_lane(valid_offsets_mask, lane_idx) ? get_high_neighbor_logical_lane_idx(active_lanes_mask, valid_offsets_mask, lane_idx) : -1;
            //
            warp::syncWarpThreads_mask(acc, active_lanes_mask);
            //
            const unsigned int warp_neigh_begin = is_valid_lane(valid_offsets_mask, lane_idx) ? warp::shfl_mask(acc, valid_offsets_mask, begin, neigh_lane_idx, w_extent) : begin;
            const unsigned int end  = lane_idx == neigh_lane_idx ? connected_comp_offsets[idx.local+(warp_work_extent - lane_idx)] : warp_neigh_begin;
            //
            // We need to exclude all vertices that are globally isolated 
            // that is, those which belong to connected components with component size equal to one
            const unsigned component_size = end - begin;
            // Determin a custom mask for such vertices:
            warp::syncWarpThreads_mask(acc, active_lanes_mask);
            unsigned int valid_vertex_mask = warp::ballot_mask(acc, active_lanes_mask, (component_size != 1));
            // If the warp does not contain valid vertices for processing at all, then skip to the next warp:
            if( valid_vertex_mask == 0x0 ) continue;
            //
            const unsigned int first_valid_lane_idx = get_ls1b_idx(acc, valid_vertex_mask);
            const unsigned int last_valid_lane_idx  = get_ms1b_idx(acc, valid_vertex_mask);
            //
            const int cluster_rhf_size   = is_valid_lane(valid_vertex_mask, lane_idx) ? component_vertex_rhf_sizes[idx.local] : 0;
            const int cluster_rhf_offset = is_valid_lane(valid_vertex_mask, lane_idx) ? component_vertex_rhf_sizes[idx.local] : 0; 
        
            unsigned int src_rhf_global_offset = 0;
        
            // Initialize valid iterative lane index:
            unsigned int iter_lane_idx         = first_valid_lane_idx;
            // Broadcast first valid cluster rhfrac size among the rest of the lanes:
            warp::syncWarpThreads_mask(acc, active_lanes_mask);
            
            unsigned int src_rhf_leftover_size = warp::shfl_mask( acc, active_lanes_mask, cluster_rhf_size, first_valid_lane_idx, w_extent );       
            unsigned int src_rhf_consumed_size = 0;

            // Start iterations untill valid vertex mask will be empty:
            while (iter_lane_idx <= last_valid_lane_idx) {

              unsigned int src_rhf_local_offset  = lane_idx + src_rhf_consumed_size; 

              unsigned int src_lane_idx = iter_lane_idx;

              bool are_all_active_lanes_got_job = false; 
              // The initial (iterative) work extent is set to the total warp work extent.
              // Later, we check whether all lanes in the active lanes mask have been assigned work.
              // For example, this is true if the current rechit fraction size is greater than or equal 
              // to the iterative work extent.
              // If the rh fraction size is smaller than the iterative work extent, some lanes will remain idle,
              // and we must continue with the next cluster's rh fraction size to assign work to those idle lanes.
              // Note that work is distributed to lanes in order of increasing lane index (i.e., idle lanes will have 
              // higher lane index)
              unsigned int iter_work_extent = warp_work_extent;
              // Accumulated  work indicates actual work extent 
              unsigned int accum_work_extent = 0;

              // We need to decide whether to broadcast the current cluster RH fraction offset.
              // The criterion for broadcasting is that the leftover size is zero
              // meaning the offset was not already broadcasted in a previous iteration.
              bool do_broadcast = (src_rhf_leftover_size == 0);

              while ( are_all_active_lanes_got_job == false ) {
                // 
                if (src_rhf_leftover_size >= iter_work_extent ) {
                  // Okay , all lanes got job:
                  are_all_active_lanes_got_job = true; 
                  //
                  if( src_rhf_leftover_size == iter_work_extent ) { 
                    // Update iteration lane index for the next iteration cycle               
                    // 1. Erase ls1b in the current iterative mask:
                    valid_vertex_mask = erase_ls1b(acc, valid_vertex_mask); 
                    // 2. Compute lowest index of the new ls1b:
                    iter_lane_idx = get_ls1b_idx(acc, valid_vertex_mask);
                    // In the (rare) case when current cluster rh frac size exactly matches work extent
                    // there is nothing left for the next iteration, and we can safely update iteration
                    // lane index for the next cluster. 
                    // We need to reset src_rhf_consumed_size in case if this is not the first iteration
                    // for the current cluster, namely, if we process leftover work:
                    src_rhf_consumed_size = 0; 
                    // 
                    warp::syncWarpThreads_mask(acc, active_lanes_mask);
                    //
                    src_rhf_leftover_size = warp::shfl_mask( acc, active_lanes_mask, cluster_rhf_size, iter_lane_idx, w_extent );
                    //              
                  } else {
                    // otherwise subtracted consumed portion from total rh fraction size:
                    // and keep same iter lane idx but update cluster's rhf size to be processed
                    // in the next iteration (leftover size)
                    src_rhf_leftover_size -= iter_work_extent;
                    src_rhf_consumed_size += iter_work_extent; 
                  }
                  // 
                } else if (iter_lane_idx < last_valid_lane_idx) { 
                  // If iterative rh fraction size is smaller then work extent, 
                  // then it makes sense to load new rh fraction size from the next valid
                  // cluster (that is why the condition iter_lane_idx < last_valid_lane_idx).
                  // Determin the next valid vertex:
                  // 1. Erase ls1b in the current iterative mask:
                  valid_vertex_mask = erase_ls1b(acc, valid_vertex_mask); 
                  // 2. Compute lowest index of the new ls1b:
                  iter_lane_idx = get_ls1b_idx(acc, valid_vertex_mask);
                  // update accumulated work dimension (accum work size):
                  accum_work_extent += src_rhf_leftover_size;
                  // update leftover work dimension:
                  iter_work_extent  -= src_rhf_leftover_size;
                  // update source lane idx for appropriate (leftover) lanes, also set broadcast flag for new cluster:
                  if (lane_idx >= accum_work_extent) {
                    do_broadcast  = true;
                    // unpdate source lane index 
                    src_lane_idx  = iter_lane_idx;
                    //we also need to define (new) shifted local offset:
                    src_rhf_local_offset = lane_idx - accum_work_extent;
                  }
                  //
                  warp::syncWarpThreads_mask(acc, active_lanes_mask);
                  //
                  src_rhf_leftover_size = warp::shfl_mask( acc, active_lanes_mask, cluster_rhf_size, iter_lane_idx, w_extent );
                  //              
                  are_all_active_lanes_got_job = ( accum_work_extent == warp_work_extent ); 
                } else { //iter_lane_idx == last_valid_lane_idx
                  //
                  //active_lanes_mask = active_lanes_mask | ??;
                  //
                  are_all_active_lanes_got_job = true;
                }// all active lanes got job 
              } // warp extent is filled with work

              warp::syncWarpThreads_mask(acc, active_lanes_mask);

              const unsigned int broadcast_mask = warp::ballot_mask(acc, active_lanes_mask, do_broadcast);
              //
              warp::syncWarpThreads_mask(acc, active_lanes_mask);
              //
              const unsigned int new_src_rhf_global_offset = warp::shfl_mask( acc, broadcast_mask, cluster_rhf_offset, src_lane_idx, w_extent);
              //
              warp::syncWarpThreads_mask(acc, active_lanes_mask);
              // Update global offset if necessary:
              src_rhf_global_offset = do_broadcast ? new_src_rhf_global_offset : src_rhf_global_offset;

              const bool is_master_lane = lane_idx == src_lane_idx;

              const auto detIdx  =  pfRecHits[src_rhf_global_offset+src_rhf_local_offset].detId();
              const auto seedIdx =  is_master_lane ? component_cluster_seeds[idx.local] : 0;
              warp::syncWarpThreads_mask(acc, active_lanes_mask);
              const unsigned int seedIdx_ = warp::shfl_mask( acc, active_lanes_mask, seedIdx, src_lane_idx, w_extent );

              warp::syncWarpThreads_mask(acc, active_lanes_mask);
              const auto result_mask = warp::ballot_mask(acc, active_lanes_mask, detIdx == seedIdx_); 

              // Determin ALL winning lanes : 
              // note that by construction they correspond to different source clusters.
              const bool is_selected = result_mask & (1 << lane_idx);
              // 
              if (is_selected) { 
                component_cluster_energies[src_lane_idx+warp_idx*w_extent] = pfRecHits[src_rhf_global_offset+src_rhf_local_offset].energy();
              }
            } // end while over valid vertices.
            //
          }
          //
          alpaka::syncBlockThreads(acc);
          //
          for( auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent)) ) { 
            // Reset shared memory buffers to zero:
            connected_comp_vertices[ idx.local ] = 0;
            connected_comp_pos[ idx.local ] = 0;
            //
          } 
          //
          alpaka::syncBlockThreads(acc);          
          //
          for( auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent)) ) { 
            //
            unsigned int active_lanes_mask = alpaka::warp::ballot(acc, idx.local < nVertices);
            // Skip inactive lanes:
            if(idx.local >= nVertices) continue;
            //
            auto is_root_lane  = [](const unsigned int mask, const unsigned int lid) -> bool { 
              return ((mask >> lid) & 1); 
            };
            //
            // Determin actual warp-level work dimension: it coincides with w_extent for all warps 
            // except (potentially!) the last one:
            const auto warp_work_extent = warp::ballot_mask(acc, active_lanes_mask, true);
            //        
            const auto vertex_idx = idx.local; 
            //
            const auto warp_idx   = idx.local / w_extent;
            const auto lane_idx   = idx.local % w_extent;
            //
            const unsigned int component_root_idx = component_roots[vertex_idx];
            //
            const bool is_component_root = component_root_idx == vertex_idx;
            //
            warp::syncWarpThreads_mask(acc, active_lanes_mask);
            //
            const auto component_root_mask = warp::ballot_mask(acc, active_lanes_mask, is_component_root);            

            if (is_root_lane(component_root_mask, lane_idx) == false) continue;

            const unsigned begin = connected_comp_offsets[idx.local];
            //
            const unsigned int intern_connected_comp_mask = intern_connected_comp_masks[idx.local];
            //
            const auto root_lanes_mask = active_lanes_mask & component_root_mask;
            //
            unsigned int connected_vertex_pos = begin;
            //
            connected_comp_vertices[connected_vertex_pos++] = component_root_idx;

            for (unsigned lid = 0; lid < warp_work_extent; ++lid) {
              const auto target_lane_idx = (intern_connected_comp_mask >> lid) & 1;
              if (target_lane_idx != 0) {
                const unsigned int connected_vertex_idx = lid + warp_idx * w_extent;
                connected_comp_vertices[connected_vertex_pos++]  = connected_vertex_idx;
              }
            }
            //
            warp::syncWarpThreads_mask(acc, root_lanes_mask);
            //
            connected_comp_pos[component_root_idx] = connected_vertex_pos;
          } 

          //
          alpaka::syncBlockThreads(acc);
          //
          for( auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent)) ) { 
            //
            unsigned int active_lanes_mask = alpaka::warp::ballot(acc, idx.local < nVertices);
            // Skip inactive lanes:
            if(idx.local >= nVertices) continue;
            //
            // Determin actual warp-level work dimension: it coincides with w_extent for all warps 
            // except (potentially!) the last one:
            const auto warp_work_extent = warp::ballot_mask(acc, active_lanes_mask, true);
            //                 
            const auto vertex_idx = idx.local; 

            const auto warp_idx   = idx.local / w_extent;
            // Get local coordinate:
            const unsigned int extern_connected_comp_mask = extern_connected_comp_masks[idx.local];
 
            const auto nnz = alpaka::popcount(acc, extern_connected_comp_mask);
 
            warp::syncWarpThreads_mask(acc, active_lanes_mask);
            //const auto outer_mask = warp::ballot_mask(acc, active_lanes_mask, nnz > 0);
 
            if (nnz == 0) continue; // skip inactive lanes
            //
            const unsigned int component_root_idx = component_roots[vertex_idx];
 
            unsigned int connected_vertex_pos = alpaka::atomicAdd(acc, &connected_comp_pos[component_root_idx], nnz, alpaka::hierarchy::Threads{});
 
            for (unsigned lid = 0; lid < warp_work_extent; ++lid) {
              const auto target_lane_idx = (extern_connected_comp_mask >> lid) & 1;// number of target lanes is equal to nnz
              if (target_lane_idx != 0) {
                const unsigned int connected_vertex_idx = lid + warp_idx * w_extent;
                connected_comp_vertices[connected_vertex_pos++] = connected_vertex_idx;
              }
            }
          } 
      
          alpaka::syncBlockThreads(acc);

          for( auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent)) ) { 
            //
            const auto warp_idx   = idx.local / w_extent;
            //
            if (warp_idx > 0) continue;
            //
            const auto full_mask = alpaka::warp::ballot(acc, true);
            //
            auto is_valid_lane  = [](const unsigned int mask, const unsigned int lid) -> bool { 
              return ((mask >> lid) & 1); 
            };
            //
            auto get_logical_lane_idx = [&](const unsigned int mask, const unsigned int lane_idx) {
              // Zero out all bits lane_id
              const auto lane_mask = mask & ((1 << lane_idx) - 1);
              return alpaka::popcount(acc, lane_mask);  // Count 1s below current lane
            };

            const auto lane_idx = idx.local % w_extent;
            //
            unsigned int inc = 0;

            for (unsigned int i = 0; i < w_items; i++){
              const auto j = lane_idx+i * w_extent;
              const auto offset = j < nVertices ? connected_comp_offsets[j] : -1;
              //
              warp::syncWarpThreads_mask(acc, full_mask);
              const auto valid_offset_mask = alpaka::warp::ballot(acc, offset != -1);

              if (valid_offset_mask == 0x0) continue;
              //
              if(is_valid_lane(valid_offset_mask, lane_idx)) {
                const auto logical_lane_idx = get_logical_lane_idx(valid_offset_mask, lane_idx);
                //
                comp_offsets[logical_lane_idx + inc] = offset;
              }
              inc += alpaka::popcount(acc, valid_offset_mask);
            }
            if(lane_idx == 0) comp_offsets[nVertices] = connected_comp_offsets[nVertices];
          } 
      
          alpaka::syncBlockThreads(acc);          

          for( auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent)) ) { 
            // Skip inactive lanes:
            if(idx.local >= nVertices) continue;
            //
            pfClusteringVars[idx.local].mdpf_componentEnergy() = component_cluster_energies[idx.local];
            pfClusteringVars[idx.local].mdpf_component()       = connected_comp_vertices[idx.local];     
                        
            const auto offset = comp_offsets[idx.local];

            if (offset != -1) pfClusteringVars[idx.local].mdpf_componentIndex() = offset; 

            if(idx.local == 0) pfClusteringVars.mdpf_nTopos() = comp_offsets[nVertices];
          }             
        }                                        
      }
  };
  
} // ALPAKA_ACCELERATOR_NAMESPACE

#endif


