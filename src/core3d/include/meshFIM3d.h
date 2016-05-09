#ifndef MESHFIM3d_H
#define MESHFIM3d_H


#include <tetmesh.h>
#include <redistance3d.h>
#include <typeinfo>
#include <functional>
#include <queue>
#include <list>
#include <types.h>
#include <time.h>

#ifndef _EPS
#define _EPS 1e-5
#endif
class meshFIM3d
{
  public:

    void updateT_single_stage(float, int, int, std::vector<int>&);
    void updateT_single_stage_d(float, int, IdxVector_d&, int);
    void getPartIndicesNegStart(IdxVector_d& sortedPartition, IdxVector_d& partIndices);
    void mapAdjacencyToBlock(IdxVector_d &adjIndexes, IdxVector_d &adjacency,
        IdxVector_d &adjacencyBlockLabel, IdxVector_d &blockMappedAdjacency, IdxVector_d &fineAggregate);
    void updateT_two_stage(float, int, int);

    void SetMesh(TetMesh* mesh, int nNoiseIter)
    {
      m_meshPtr = mesh;
    }
    std::vector < std::vector < float> >  GenerateData(
        char* filename, int nsteps, float timestep,
        int inside_niter, int nside, int block_size,
        float bandwidth, int part_type, int metis_size,
        bool verbose = false);
    void Partition_METIS(int metissize, bool verbose = false);
    void GraphPartition_Square(int squareLength, int squareWidth, int squareHeight,
        int blockLength, int blockWidth, int blockHeight, bool verbose = false);
    void InitPatches(bool verbose = false);
    void InitPatches2();
    void GenerateBlockNeighbors();
    void writeVTK(std::vector < std::vector <float> > values);
    void writeFLD();

    meshFIM3d(TetMesh* mesh)
    {
      size_t nn = mesh->vertices.size();
      size_t ne = mesh->tets.size();
      m_meshPtr = mesh;
      vertT_out = Vector_d(nn, 0.0);
      tmp_vertT_before_permute_d = Vector_d(nn, 0.0);
      ele_label_d = IdxVector_d(ne);
      ele_offsets_d = IdxVector_d(ne + 1);
      m_mem_location_offsets = IdxVector_d(nn + 1);
      m_largest_num_inside_mem = 0;
      NumComputation = 0;
      full_num_ele = 0;
      nparts = 0;
      largest_vert_part = 0;
      largest_ele_part = 0;
      m_redist = 0;
    };

    ~meshFIM3d()
    {
    };

    TetMesh* m_meshPtr;
    redistance3d* m_redist;
    int NumComputation;
    std::vector<int> narrowband;
    IdxVector_h npart_h;
    IdxVector_d m_npart_d;
    IdxVector_d m_part_label_d;
    IdxVector_h epart_h;
    IdxVector_d epart_d;
    IdxVector_d m_ele_offsets_d;
    IdxVector_d m_vert_offsets_d;
    Vector_d m_vert_after_permute_d;
    IdxVector_d m_ele_after_permute_d;
    Vector_d m_ele_local_coords_d;
    int nparts;
    int largest_vert_part;
    int largest_ele_part;
    int m_largest_num_inside_mem;
    int full_num_ele;
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
    Vector_d vert_d;
    Vector_d vertT_out;
    Vector_d tmp_vertT_before_permute_d;
    Vector_d m_cadv_global_d;
    Vector_d m_cadv_local_d;
  private:
    IdxVector_d ones;
    IdxVector_d tmp;
    IdxVector_d reduce_output;
    IdxVector_d ele_full_label;
    IdxVector_d ele_label_d;
    IdxVector_d ele_offsets_d;
    IdxVector_d adjacencyBlockLabel;
    IdxVector_d blockMappedAdjacency;
};
#endif
