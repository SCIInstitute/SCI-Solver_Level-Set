#ifndef __MESHFIM2D_H__
#define __MESHFIM2D_H__

#include <TriMesh.h>
#include <redistance.h>
//#include "TriMesh_algo.h"
#include <typeinfo>
#include <functional>
#include <queue>
#include <list>
#include <types.h>
#include <time.h>

#ifndef _EPS
#define _EPS 1e-12
#endif

class meshFIM2d
{
  public:

    void updateT_single_stage(double, int, int, vector<int>&);
    void updateT_single_stage_d(double, int, IdxVector_d&, int);
    void getPartIndicesNegStart(IdxVector_d& sortedPartition,
      IdxVector_d& partIndices);
    void mapAdjacencyToBlock(IdxVector_d &adjIndexes, 
      IdxVector_d &adjacency, IdxVector_d
      &adjacencyBlockLabel, IdxVector_d &blockMappedAdjacency,
      IdxVector_d &fineAggregate);
    void updateT_two_stage(double, int, int);

    void SetMesh(TriMesh* mesh, int nNoiseIter)
    {
      m_meshPtr = mesh;
    }
    std::vector< std::vector <float> > GenerateData(
      const char* filename, int nsteps, double timestep,
        int inside_niter, int nside, int block_size, double bandwidth,
        int part_type, int metis_size, bool verbose = false);
    void Partition_METIS(int metissize, bool verbose = false);
    void GraphPartition_Square(int squareLength, int squareWidth, 
      int blockLength, int blockWidth, bool verbose = false);
    void InitPatches(bool verbose = false);
    void InitPatches2();
    void compute_deltaT(int num_narrowband, bool verbose = false);
    void GenerateBlockNeighbors();
    void writeVTK(std::vector< std::vector <float> > time_values);
    void writeFLD();

    meshFIM2d(TriMesh* mesh)
    {
      m_meshPtr = mesh;
      vertT_out = Vector_d(mesh->vertices.size());
      tmp_vertT_before_permute_d = Vector_d(mesh->vertices.size());
    };

    ~meshFIM2d()
    {
    };

    TriMesh* m_meshPtr;
    redistance* m_redist;
    int NumComputation;
    vector<int> narrowband;
    IdxVector_h npart_h;
    IdxVector_d m_npart_d;
    IdxVector_d m_part_label_d;
    IdxVector_h epart_h;
    IdxVector_d epart_d;
    IdxVector_d m_ele_offsets_d;
    IdxVector_d m_vert_offsets_d;
    IdxVector_d m_narrowband_d;
    Vector_d m_vert_after_permute_d;
    IdxVector_d m_ele_after_permute_d;
    Vector_d m_ele_local_coords_d;
    Vector_d m_cadv_local_d;
    Vector_d timestep_per_block;
    Vector_d Rin_per_block;
    Vector_d m_cadv_global_d;
    Vector_d m_ceik_global_d;
    Vector_d m_ccurv_global_d;
    double m_timestep;
    double m_maxRin;
    int nparts;
    int largest_vert_part;
    int largest_ele_part;
    int m_largest_num_inside_mem;
    int full_num_ele;
    int largest_Rin;
    Vector_d m_vertT_d;
    Vector_d m_vertT_after_permute_d;
    IdxVector_d m_mem_locations;
    IdxVector_d m_mem_location_offsets;
    IdxVector_d m_xadj_d;
    IdxVector_d m_adjncy_d;
    IdxVector_d m_neighbor_sizes_d;
    IdxVector_d m_block_xadj_d;
    IdxVector_d m_block_adjncy_d;
    IdxVector_d m_vert_permute_d;
    IdxVector_d ele_d;
    IdxVector_h ele_h;
    IdxVector_d ele_permute;
    Vector_d m_Rinscribe_d;
    Vector_d m_Rinscribe_before_permute_d;
    Vector_d vert_d;
    Vector_d vertT_out;
    Vector_d tmp_vertT_before_permute_d;
};
#endif
