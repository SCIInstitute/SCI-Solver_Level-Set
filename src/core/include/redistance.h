#ifndef REDIST_H
#define REDIST_H

#include <tetmesh.h>
#include <typeinfo>
#include <functional>
#include <queue>
#include <list>
#include <time.h>
#include <stdio.h>
#include <types.h>
using namespace std;

#ifndef _EPS
#define _EPS 1e-9
#endif

class redistance
{
public:
  typedef int index;

  enum LabelType
  {
    SeedPoint, ActivePoint, AlivePoint, FarPoint
  };

  void FindSeedPoint(const IdxVector_d& old_narrowband, const int num_old_narrowband, TetMesh* mesh, Vector_d& vertT_after_permute_d, int nparts, int largest_vert_part, int largest_ele_part, int largest_num_inside_mem, int full_num_ele,
                     Vector_d& vert_after_permute_d, IdxVector_d& vert_offsets_d,
                     IdxVector_d& ele_after_permute_d, IdxVector_d& ele_offsets_d, Vector_d& ele_local_coords_d, IdxVector_d& mem_location_offsets, IdxVector_d& mem_locations,
                     IdxVector_d& part_label_d, IdxVector_d& block_xadj, IdxVector_d& block_adjncy);

  void GenerateData(IdxVector_d& new_narrowband, int& num_new_narrowband, LevelsetValueType bandwidth, int stepcount, TetMesh* mesh, Vector_d& vertT_after_permute_d, int nparts, int largest_vert_part, int largest_ele_part, int largest_num_inside_mem, int full_num_ele,
                    Vector_d& vert_after_permute_d, IdxVector_d& vert_offsets_d,
                    IdxVector_d& ele_after_permute_d, IdxVector_d& ele_offsets_d, Vector_d& ele_local_coords_d, IdxVector_d& mem_location_offsets, IdxVector_d& mem_locations,
                    IdxVector_d& part_label_d, IdxVector_d& block_xadj, IdxVector_d& block_adjncy, bool verbose = false);
  void ReInitTsign(TetMesh* mesh, Vector_d& vertT_after_permute_d, int nparts, int largest_vert_part, int largest_ele_part, int largest_num_inside_mem, int full_num_ele,
                   Vector_d& vert_after_permute_d, IdxVector_d& vert_offsets_d,
                   IdxVector_d& ele_after_permute_d, IdxVector_d& ele_offsets_d, Vector_d& ele_local_coords_d, IdxVector_d& mem_location_offsets, IdxVector_d& mem_locations,
                   IdxVector_d& part_label_d, IdxVector_d& block_xadj, IdxVector_d& block_adjncy);

  redistance(TetMesh* mesh, int nparts, IdxVector_d& block_xadj, IdxVector_d& block_adjncy)
  {
    int nn = mesh->vertices.size();
    m_DT_d = Vector_d(nn);
    m_active_block_list_d = IdxVector_d(nparts + 1);
    m_Tsign_d = CharVector_d(nn);
    m_Label_d = IdxVector_d(nn);
    h_ActiveListNew.reserve(nparts);
    tmp_new_narrowband = IdxVector_d(nparts + 1);
    DT_d_out = Vector_d(nn, 0.0);
    d_vert_con = IdxVector_d(nn, 0);
    d_block_con = BoolVector_d(nparts, 0);
    h_block_con = BoolVector_h(nparts);
    block_xadj_h = IdxVector_h(block_xadj.begin(), block_xadj.end());
    block_adjncy_h = IdxVector_h(block_adjncy.begin(), block_adjncy.end());
    h_ActiveList = IdxVector_h(nparts);
    h_BlockLabel = vector<int>(nparts, FarPoint);
    d_block_vertT_min = Vector_d(nparts);
    h_block_vertT_min = Vector_h(nparts);
  };

  ~redistance()
  {
  };

  int NumComputation;
  list<index> m_ActivePoints;
  vector<index> m_SeedPoints;
  vector<LabelType> m_Label;
  IdxVector_d m_Label_d;
  IdxVector_d m_active_block_list_d;
  Vector_d m_DT_d;
  CharVector_d m_Tsign_d;
  vector<int> h_ActiveListNew;
  IdxVector_d tmp_new_narrowband;
  Vector_d DT_d_out;
  IdxVector_d d_vert_con;
  IdxVector_d d_block_con;
  IdxVector_h h_block_con;
  IdxVector_h block_xadj_h;
  IdxVector_h block_adjncy_h;

  IdxVector_h h_ActiveList;
  vector<int> h_BlockLabel;
  Vector_d d_block_vertT_min;
  Vector_h h_block_vertT_min;
};

#endif
