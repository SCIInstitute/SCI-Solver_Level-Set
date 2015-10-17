#include <redistance.h>
#include <redistance_kernels.h>
#include <Vec.h>
#include <math.h>
#include <stdio.h>
#include <mycutil.h>

#include "cusp/print.h"

void redistance::ReInitTsign(TriMesh* mesh, Vector_d& vertT_after_permute_d, int nparts, int largest_vert_part, int largest_ele_part, int largest_num_inside_mem, int full_num_ele,
                             Vector_d& vert_after_permute_d, IdxVector_d& vert_offsets_d,
                             IdxVector_d& ele_after_permute_d, IdxVector_d& ele_offsets_d, Vector_d& ele_local_coords_d, IdxVector_d& mem_location_offsets, IdxVector_d& mem_locations,
                             IdxVector_d& part_label_d, IdxVector_d& block_xadj, IdxVector_d& block_adjncy)
{
  int nn = mesh->vertices.size();
  int nthreads = 256;
  int nblocks = min((int)ceil((LevelsetValueType)nn / nthreads), 65535);
  cudaSafeCall((kernel_reinit_Tsign << <nblocks, nthreads >> >(nn, CAST(vertT_after_permute_d), CAST(m_Tsign_d))));
}

void redistance::FindSeedPoint(const IdxVector_d& old_narrowband, const int num_old_narrowband, TriMesh* mesh, Vector_d& vertT_after_permute_d, int nparts,
                               int largest_vert_part, int largest_ele_part, int largest_num_inside_mem, int full_num_ele,
                               Vector_d& vert_after_permute_d, IdxVector_d& vert_offsets_d,
                               IdxVector_d& ele_after_permute_d, IdxVector_d& ele_offsets_d, Vector_d& ele_local_coords_d,
                               IdxVector_d& mem_location_offsets, IdxVector_d& mem_locations,
                               IdxVector_d& part_label_d, IdxVector_d& block_xadj, IdxVector_d& block_adjncy)
{
  int ne = mesh->faces.size();
  int nn = mesh->vertices.size();
  int nnb = num_old_narrowband;
  //  printf("nnb = %d\n", nnb);
  thrust::fill(m_DT_d.begin(), m_DT_d.end(), LARGENUM);
  //  m_active_block_list_d.resize(nparts + 1);
  m_active_block_list_d[0] = 0;
  if(nnb == 0)
  {
    //    m_Label_d.resize(nn, FarPoint);
    thrust::fill(m_Label_d.begin(), m_Label_d.end(), FarPoint);
    int nthreads = largest_ele_part;
    int nblocks = nparts;

    cudaSafeCall((kernel_seedlabel << <nblocks, nthreads >> >(nn, full_num_ele,
                                                              CAST(vert_after_permute_d),
                                                              CAST(vert_offsets_d),
                                                              CAST(ele_after_permute_d),
                                                              CAST(ele_offsets_d),
                                                              CAST(m_Label_d),
                                                              CAST(vertT_after_permute_d),
                                                              CAST(m_DT_d),
                                                              CAST(m_active_block_list_d))));
    //    Vector_h tmp = m_DT_d;
    //    for(int i=0; i<tmp.size(); i++)
    //      printf("m_DT_d[%d] = %f\n", i, tmp[i]);
    //
    //    IdxVector_h idxtmp = m_active_block_list_d;
    //    for(int i=0; i<idxtmp.size(); i++)
    //      printf("m_active_block_list_d[%d] = %d\n", i, idxtmp[i]);
  }
  else
  {
    //    m_Label_d.resize(nn, FarPoint);
    thrust::fill(m_Label_d.begin(), m_Label_d.end(), FarPoint);
    int nthreads = largest_ele_part;
    //    Vector_h dtmp_h = m_vertT_after_permute_d;
    //		for(int i =0; i<dtmp_h.size(); i++)
    //		{
    //			printf("m_vertT_after_permute_d[%d] = %f\n", i, dtmp_h[i]);
    //		}
    int nblocks = nnb;
    cudaSafeCall((kernel_seedlabel_narrowband << <nblocks, nthreads >> >(nn, full_num_ele, CAST(old_narrowband),
                                                                         CAST(vert_after_permute_d),
                                                                         CAST(vert_offsets_d),
                                                                         CAST(ele_after_permute_d),
                                                                         CAST(ele_offsets_d),
                                                                         CAST(m_Label_d),
                                                                         CAST(vertT_after_permute_d),
                                                                         CAST(m_DT_d),
                                                                         CAST(m_active_block_list_d))));

  }
  //  Vector_h tmp_h = m_DT_d;
  //  for(int i = 0; i < tmp_h.size(); i++)
  //  {
  //    printf("m_DT_d[%d] = %f\n", i, tmp_h[i]);
  //  }
  //
  //  tmp_h = m_active_block_list_d;
  //  printf("Num active list = %d\n", tmp_h[0]);
  //  for(int i = 0; i < tmp_h[0]; i++)
  //  {
  //    printf("m_active_block_list_d[%d]=%d\n", i, tmp_h[i + 1]);
  //  }
}

void redistance::GenerateData(IdxVector_d& new_narrowband, int& new_num_narrowband, LevelsetValueType bandwidth, int stepcount, TriMesh* mesh, Vector_d& vertT_after_permute_d,
                              int nparts, int largest_vert_part, int largest_ele_part, int largest_num_inside_mem, int full_num_ele,
                              Vector_d& vert_after_permute_d, IdxVector_d& vert_offsets_d,
                              IdxVector_d& ele_after_permute_d, IdxVector_d& ele_offsets_d, Vector_d& ele_local_coords_d,
                              IdxVector_d& mem_location_offsets, IdxVector_d& mem_locations,
                              IdxVector_d& part_label_d, IdxVector_d& block_xadj, IdxVector_d& block_adjncy, bool verbose)
{
  int nn = mesh->vertices.size();
  int totalIterationNumber = 0;
  int nblocks, nthreads, shared_size;
  int NUM_ITER = 10;
  int nTotalIter = 0;
  int numActive = m_active_block_list_d[0];
  //	printf("numActive=%d\n", numActive);
  thrust::copy(m_DT_d.begin(), m_DT_d.end(), DT_d_out.begin());
  thrust::fill(d_vert_con.begin(), d_vert_con.end(), 0);
  thrust::fill(d_block_con.begin(), d_block_con.end(), 0);
  //  h_block_con = BoolVector_h(nparts);
  //  block_xadj_h = IdxVector_h(block_xadj.begin(), block_xadj.end());
  //  block_adjncy_h = IdxVector_h(block_adjncy.begin(), block_adjncy.end());

  thrust::copy(m_active_block_list_d.begin() + 1, m_active_block_list_d.begin() + 1 + numActive, h_ActiveList.begin());
  h_BlockLabel.assign(nparts, FarPoint);
  //  d_block_vertT_min = Vector_d(nparts);
  //  h_block_vertT_min = Vector_h(nparts);
  while(numActive > 0)
  {
    if (verbose) {
      size_t act = numActive / 3;
      for (size_t ab = 0; ab < 60; ab++) {
        if (ab < act)
          printf("=");
        else
          printf(" ");
      }
      printf(" %d Active blocks.\n", numActive);
    }
//    printf("nTotalIter = %d, numActive=%d\n", nTotalIter, numActive);
    ///////////////////////////step 1: run solver //////////////////////////////////////////////////////////////////
    nTotalIter++;
    totalIterationNumber += numActive;

    nblocks = numActive;
    nthreads = largest_ele_part;
    m_active_block_list_d = h_ActiveList;
    shared_size = sizeof(LevelsetValueType)* 3 * largest_ele_part + sizeof(short)*largest_vert_part*largest_num_inside_mem;
    //    if(stepcount == 1 && nTotalIter == 1)
    //    Vector_h tmp_h = m_DT_d;
    //    for(int i = 0; i < tmp_h.size(); i++)
    //    {
    //      printf("m_DT_d[%d] = %f\n", i, tmp_h[i]);
    //    }
    //
    //    IdxVector_h itmp_h = m_active_block_list_d;
    //    for(int i = 0; i < numActive; i++)
    //    {
    //      printf("m_active_block_list_d[%d] = %d\n", i, itmp_h[i]);
    //    }

    //
    //    tmp_h = m_mem_location_offsets;
    //    for (int i = 0; i < tmp_h.size(); i++)
    //    {
    //      printf("m_mem_location_offsets[%d] = %d\n", i, tmp_h[i]);
    //    }
    //    Vector_h oldDT_h = m_DT_d;
    cudaSafeCall((kernel_update_values << <nblocks, nthreads, shared_size >> >(CAST(m_active_block_list_d), CAST(m_Label_d), largest_ele_part, largest_vert_part, full_num_ele,
                                                                               CAST(ele_after_permute_d), CAST(ele_offsets_d),
                                                                               CAST(vert_offsets_d), CAST(m_DT_d),
                                                                               CAST(ele_local_coords_d), largest_num_inside_mem, CAST(mem_locations), CAST(mem_location_offsets),
                                                                               NUM_ITER, CAST(DT_d_out), CAST(d_vert_con))));
    //    tmp_h = DT_d_out;
    //    for(int i = 0; i < tmp_h.size(); i++)
    //    {
    //      printf("m_DT_d[%d] = %f\n", i, tmp_h[i]);
    //    }

    nthreads = largest_vert_part;
    cudaSafeCall((CopyOutBack << <nblocks, nthreads >> >(CAST(m_active_block_list_d),
                                                         CAST(vert_offsets_d), CAST(m_DT_d), CAST(DT_d_out))));

    //    IdxVector_h itmp_h = d_vert_con;

    //		if(stepcount == 1 && nTotalIter ==1)
    //		{
    //		    Vector_h dtmp_h = m_DT_d;
    //				IdxVector_h tmp_h = m_Label_d;
    //        for (int i = 0; i < dtmp_h.size(); i++)
    //        {
    //          printf("oldDT_h[%d] = %f, m_DT_d[%d] = %f, m_Label_d[%d] = %d\n", i, oldDT_h[i], i, dtmp_h[i], i, tmp_h[i]);
    //        }
    //		}
    //    Vector_h tmp_h = m_DT_d;
    //////////////////////step 2: reduction////////////////////////////////////////////////
    if(nthreads <= 32)
    {
      cudaSafeCall((run_reduction_bandwidth < 32 > << <nblocks, 32 >> > (CAST(d_vert_con), CAST(d_block_con), CAST(m_active_block_list_d),
                                                                         CAST(DT_d_out), CAST(d_block_vertT_min), CAST(vert_offsets_d))));
    }
    else if(nthreads <= 64)
    {
      cudaSafeCall((run_reduction_bandwidth < 64 > << <nblocks, 64 >> > (CAST(d_vert_con), CAST(d_block_con), CAST(m_active_block_list_d),
                                                                         CAST(DT_d_out), CAST(d_block_vertT_min), CAST(vert_offsets_d))));
    }
    else if(nthreads <= 128)
    {
      cudaSafeCall((run_reduction_bandwidth < 128 > << <nblocks, 128 >> > (CAST(d_vert_con), CAST(d_block_con), CAST(m_active_block_list_d),
                                                                           CAST(DT_d_out), CAST(d_block_vertT_min), CAST(vert_offsets_d))));
    }
    else if(nthreads <= 256)
    {
      cudaSafeCall((run_reduction_bandwidth < 256 > << <nblocks, 256 >> > (CAST(d_vert_con), CAST(d_block_con), CAST(m_active_block_list_d),
                                                                           CAST(DT_d_out), CAST(d_block_vertT_min), CAST(vert_offsets_d))));
    }
    else if(nthreads <= 512)
    {
      cudaSafeCall((run_reduction_bandwidth < 512 > << <nblocks, 512 >> > (CAST(d_vert_con), CAST(d_block_con), CAST(m_active_block_list_d),
                                                                           CAST(DT_d_out), CAST(d_block_vertT_min), CAST(vert_offsets_d))));
    }
    else
    {
      printf("Error: nthreads greater than 256!!!\n");
    }

    //    Vector_h dtmp_h = d_block_vertT_min;
    //    for(int i = 0; i < dtmp_h.size(); i++)
    //    {
    //      printf("d_block_con[%d]=%f\n", i, dtmp_h[i]);
    //    }
    // 3. check neighbor tiles of converged tile 
    thrust::copy(d_block_con.begin(), d_block_con.end(), h_block_con.begin());
    //    if(stepcount == 2 && nTotalIter == 3)
    //    {
    //      Vector_h tmp_h = d_block_vertT_min;
    //      for(int i = 0; i < tmp_h.size(); i++)
    //      {
    //        printf("d_block_vertT_min[%d] = %f\n", i, tmp_h[i]);
    //      }
    //    }
    h_block_vertT_min = d_block_vertT_min;
    int nOldActiveBlock = numActive;
    numActive = 0;
    h_ActiveListNew.clear();
    for(int i = 0; i < nOldActiveBlock; i++)
    {
      int currBlkIdx = h_ActiveList[i];
      h_BlockLabel[currBlkIdx] = FarPoint;
      if(!h_block_con[currBlkIdx]) // if not converged
      {
        h_BlockLabel[currBlkIdx] = ActivePoint;
      }
    }
    for(int i = 0; i < nOldActiveBlock; i++)
    {
      int currBlkIdx = h_ActiveList[i];

      if(h_block_con[currBlkIdx] && h_block_vertT_min[currBlkIdx] < bandwidth) //converged
      {
        int start = block_xadj_h[currBlkIdx];
        int end = block_xadj_h[currBlkIdx + 1];

        for(int iter = 0; iter < end - start; iter++)
        {
          int currIdx = block_adjncy_h[iter + start];
          if(h_BlockLabel[currIdx] == FarPoint)
          {
            h_BlockLabel[currIdx] = ActivePoint;
            h_ActiveListNew.push_back(currIdx);
          }
        }
      }
    }

    for(int i = 0; i < nOldActiveBlock; i++)
    {
      int currBlkIdx = h_ActiveList[i];
      if(!h_block_con[currBlkIdx]) // if not converged
      {
        h_ActiveList[numActive++] = currBlkIdx;
      }
    }

    //		for(int i=0; i<h_ActiveListNew.size(); i++)
    //		{
    //			printf("h_ActiveListNew[%d]=%d\n", i, h_ActiveListNew[i]);
    //		}

    //////////////////////////////////////////////////////////////////
    // 4. run solver only once for neighbor blocks of converged block
    // current active list contains active blocks and neighbor blocks of
    // any converged blocks

    if(h_ActiveListNew.size() > 0)
    {
      int numActiveNew = h_ActiveListNew.size();
      m_active_block_list_d = h_ActiveListNew;
      nblocks = numActiveNew;
      nthreads = largest_ele_part;

      int sharedSize = sizeof(LevelsetValueType)* 3 * largest_ele_part + sizeof(short)*largest_vert_part*largest_num_inside_mem;
      cudaSafeCall((kernel_run_check_neghbor << <nblocks, nthreads, shared_size >> >(CAST(m_active_block_list_d), CAST(m_Label_d), largest_ele_part, largest_vert_part,
                                                                                     full_num_ele,
                                                                                     CAST(ele_after_permute_d), CAST(ele_offsets_d),
                                                                                     CAST(vert_offsets_d), CAST(m_DT_d),
                                                                                     CAST(ele_local_coords_d), largest_num_inside_mem, CAST(mem_locations), CAST(mem_location_offsets), 1,
                                                                                     CAST(DT_d_out), CAST(d_vert_con))));

      //      nthreads = largest_vert_part;
      //      cudaSafeCall((CopyOutBack << <nblocks, nthreads >> >(CAST(m_active_block_list_d),
      //                                                           CAST(vert_offsets_d), CAST(m_DT_d), CAST(DT_d_out))));

      ////////////////////////////////////////////////////////////////
      // 5. reduction 
      ////////////////////////////////////////////////////////////////
      nthreads = largest_vert_part;
      run_reduction << <nblocks, nthreads >> >(CAST(d_vert_con), CAST(d_block_con), CAST(m_active_block_list_d), CAST(vert_offsets_d));

      //////////////////////////////////////////////////////////////////
      // 6. update active list
      // read back active volume from the device and add 
      // active block to active list on the host memory
      h_block_con = d_block_con;
      for(int i = 0; i < h_ActiveListNew.size(); i++)
      {
        int currBlkIdx = h_ActiveListNew[i];
        if(!h_block_con[currBlkIdx]) // false : activate block (not converged)
        {
          h_ActiveList[numActive++] = currBlkIdx;
        }
        else h_BlockLabel[currBlkIdx] = FarPoint;
      }
    }
  }
  //compute new narrow band list
  //  if(stepcount == 2)
  //  {
  //    Vector_h tmp_h = m_DT_d;
  //    for(int i = 0; i < tmp_h.size(); i++)
  //    {
  //      printf("m_DT_d[%d] = %f\n", i, tmp_h[i]);
  //    }
  //  }
  nblocks = nparts;
  nthreads = largest_vert_part;
  tmp_new_narrowband[0] = 0;

  if(nthreads <= 32)
  {
    cudaSafeCall((kernel_compute_new_narrowband < 32 > << <nblocks, 32 >> > (CAST(tmp_new_narrowband), CAST(m_DT_d), CAST(vert_offsets_d), bandwidth)));
  }
  else if(nthreads <= 64)
  {
    cudaSafeCall((kernel_compute_new_narrowband < 64 > << <nblocks, 64 >> >(CAST(tmp_new_narrowband), CAST(m_DT_d), CAST(vert_offsets_d), bandwidth)));
  }
  else if(nthreads <= 128)
  {
    cudaSafeCall((kernel_compute_new_narrowband < 128 > << <nblocks, 128 >> >(CAST(tmp_new_narrowband), CAST(m_DT_d), CAST(vert_offsets_d), bandwidth)));
  }
  else if(nthreads <= 256)
  {
    cudaSafeCall((kernel_compute_new_narrowband < 256 > << <nblocks, 256 >> >(CAST(tmp_new_narrowband), CAST(m_DT_d), CAST(vert_offsets_d), bandwidth)));
  }
  else if(nthreads <= 512)
  {
    cudaSafeCall((kernel_compute_new_narrowband < 512 > << <nblocks, 512 >> >(CAST(tmp_new_narrowband), CAST(m_DT_d), CAST(vert_offsets_d), bandwidth)));
  }
  else
  {
    printf("Error: nthreads greater then 256!!!\n");
  }
  int numb = tmp_new_narrowband[0];
  new_num_narrowband = numb;
//  printf("Narrowband size is %d\n", new_num_narrowband);
  //  new_narrowband.resize(numb);

  //	IdxVector_h tmp_h = new_narrowband;
  //	for(int i=0; i<tmp_h.size(); i++)
  //	{
  //	  printf("new_narrowband[%d]=%d\n", i, tmp_h[i]);	
  //	}
  nblocks = numb;
  //  Vector_h tmp_h = m_DT_d;
  //  for(int i = 0; i < tmp_h.size(); i++)
  //  {
  //    printf("m_DT_d[%d] = %f\n", i, tmp_h[i]);
  //  }
  thrust::copy(m_DT_d.begin(), m_DT_d.end(), vertT_after_permute_d.begin());

  thrust::copy(tmp_new_narrowband.begin() + 1, tmp_new_narrowband.begin() + numb + 1, new_narrowband.begin());

  //  if(stepcount == 1)
  //  {
  //    IdxVector_h tmp_h = new_narrowband;
  //    for(int i = 0; i < tmp_h.size(); i++)
  //    {
  //      printf("new_narrowband[%d] = %d\n", i, tmp_h[i]);
  //    }
  //		tmp_h = m_vert_offsets_d;
  //    for(int i = 0; i < tmp_h.size(); i++)
  //    {
  //      printf("m_vert_offsets_d[%d] = %d\n", i, tmp_h[i]);
  //    }
  //
  //    CharVector_h ctmp_h = m_Tsign_d;
  //    for(int i = 0; i < ctmp_h.size(); i++)
  //    {
  //      printf("m_Tsign_d[%d] = %d\n", i, ctmp_h[i]);
  //    }
  //		Vector_h tmpT = m_vertT_after_permute_d;
  //		for(int i=0; i<tmpT.size(); i++)
  //		{
  //			tmpT[i] *= 2*ctmp_h[i]-1;
  //		}
  //		m_vertT_after_permute_d = tmpT;
  //  }

  //  printf("nblocks=%d, nthreads=%d\n", nblocks, nthreads);
  //  cudaSafeCall((kernel_recover_Tsign << <nblocks, nthreads >> >(CAST(new_narrowband), CAST(vert_offsets_d), CAST(vertT_after_permute_d), CAST(m_Tsign_d))));
  nthreads = 256;
  nblocks = min((int)ceil((LevelsetValueType)nn / nthreads), 65535);
  cudaSafeCall((kernel_recover_Tsign_whole << <nblocks, nthreads >> >(nn, CAST(vertT_after_permute_d), CAST(m_Tsign_d))));

  //	tmp_h = vertT_after_permute_d;
  //  for(int i = 0; i < tmp_h.size(); i++)
  //  {
  //    printf("vertT_after_permute_d[%d] = %f\n", i, tmp_h[i]);
  //  }
  //  cudaThreadSynchronize();
  //  int tmp = 0;
  //  if(stepcount == 1)
  //  {
  //    Vector_h tmpT = m_vertT_after_permute_d;
  //    for(int i = 0; i < tmpT.size(); i++)
  //    {
  //			printf("m_vertT_after_permute_d[%d] = %f\n", i, tmpT[i]);
  //    }
  //  }
}