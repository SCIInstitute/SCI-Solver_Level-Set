/*
 * File:   redistance_kernels.h
 * Author: zhisong
 *
 * Created on October 28, 2012, 4:14 PM
 */

#ifndef REDISTANCE_KERNELS_H
#define  REDISTANCE_KERNELS_H

#include <redistance.h>

struct or_op
{

  __host__ __device__ bool operator () (const bool& x, const bool& y) const
  {
    return x || y;
  }
};

template<int SZ>
__global__ void kernel_compute_new_narrowband(int* new_narrowband, float* vertT, int* vert_offsets, float bandwidth)
{
  int block_idx = blockIdx.x;
  int tx = threadIdx.x;
  int start = vert_offsets[block_idx];
  int end = vert_offsets[block_idx + 1];
  int blocksize = end - start;
  __shared__ float sdata[SZ];
  if (tx < blocksize)
    sdata[tx] = vertT[start + tx];
  else
    sdata[tx] = LARGENUM;

  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if (tx < s)
    {
      sdata[tx] = fmin(sdata[tx], sdata[tx + s]);
    }
    __syncthreads();
  }

  if (tx == 0)
  {
    if (sdata[0] <= bandwidth)
    {
      int index = atomicInc((unsigned int*)&new_narrowband[0], LARGENUM);
      new_narrowband[index + 1] = blockIdx.x;
    }
  }
}

template<int SZ>
__global__ void run_reduction_bandwidth(int *con, int *blockCon, int* ActiveList, float* vertT, float* block_vertT_min, int* vert_offsets)
{
  int list_idx = blockIdx.x;
  int tx = threadIdx.x;
  int block_idx = ActiveList[list_idx];
  int start = vert_offsets[block_idx];
  int end = vert_offsets[block_idx + 1];
  int blocksize = end - start;
  __shared__ int s_block_conv;
  __shared__ float sdata[SZ];

  if (tx < blocksize)
    sdata[tx] = vertT[start + tx];
  else
    sdata[tx] = LARGENUM;

  s_block_conv = 1;
  __syncthreads();

  if (tx < blocksize)
  {
    if (!con[start + tx])
      s_block_conv = 0;
  }

  for (unsigned int s = SZ / 2; s > 0; s >>= 1)
  {
    if (tx < s)
    {
      sdata[tx] = fmin(sdata[tx], sdata[tx + s]);
    }
    __syncthreads();
  }

  if (tx == 0)
  {
    blockCon[block_idx] = s_block_conv; // active list is negation of tile convergence (active = not converged)
    block_vertT_min[block_idx] = sdata[0];
  }
}

__global__ void run_reduction(int *con, int *blockCon, int* ActiveList, int* vert_offsets)
{
  int list_idx = blockIdx.x;
  int tx = threadIdx.x;
  int block_idx = ActiveList[list_idx];
  int start = vert_offsets[block_idx];
  int end = vert_offsets[block_idx + 1];
  int blocksize = end - start;
  __shared__ int s_block_conv;
  s_block_conv = 1;
  __syncthreads();

  if (tx < blocksize)
  {
    if (!con[start + tx])
      s_block_conv = 0;
  }
  __syncthreads();

  if (tx == 0)
  {
    blockCon[block_idx] = s_block_conv; // active list is negation of tile convergence (active = not converged)
  }
}

__device__ float localSolverTet1(float TA, float TB, float TC, float ACAC, float ACBC, float ACCD, float BCBC, float BCCD, float CDCD)
{
  if (TA >= LARGENUM && TB >= LARGENUM && TC >= LARGENUM)
    return LARGENUM;
  float p, q, r;
  float lambda1, lambda2, lambda3;
  float FaceTAC = LARGENUM, FaceTAB = LARGENUM, FaceTBC = LARGENUM;
  float delta, TE;
  float TD = LARGENUM;
  float TAC = TC - TA;
  float TBC = TC - TB;
  float TAB = TB - TA;

  //calculate FaceTBC, let lambda1 = 0
  p = BCBC * TBC * TBC - BCBC*BCBC;
  q = (BCCD + BCCD) * TBC * TBC - 2 * BCBC*BCCD;
  r = TBC * TBC * CDCD - BCCD*BCCD;

  delta = q * q - 4 * p*r;

  if (delta >= 0.0)
  {
    lambda1 = 0.0;
    lambda2 = (-q + sqrt(delta)) / (2.0 * p);
    lambda3 = 1 - lambda1 - lambda2;

    if (lambda2 >= 0 && lambda2 <= 1)
    {
      TE = TB * lambda2 + TC*lambda3;
      FaceTBC = fmin(FaceTBC, TE + sqrt((lambda2 * BCBC + BCCD) * lambda2 + (lambda2 * BCCD + CDCD)));
    }

    lambda1 = 0.0;
    lambda2 = (-q - sqrt(delta)) / (2.0 * p);
    lambda3 = 1 - lambda1 - lambda2;

    if (lambda2 >= 0 && lambda2 <= 1)
    {
      TE = TB * lambda2 + TC*lambda3;
      FaceTBC = fmin(FaceTBC, TE + sqrt((lambda2 * BCBC + BCCD) * lambda2 + (lambda2 * BCCD + CDCD)));
    }
  }

  FaceTBC = fmin(FaceTBC, fmin(TB + sqrt((BCBC + BCCD) + (BCCD + CDCD)), TC + sqrt(CDCD)));

  //calculate FaceTAB, let lambda3 = 0
  float gammax = ACAC - ACBC, gammay = ACBC - BCBC, gammaz = ACCD - BCCD;
  p = (TAB * TAB * ACAC - gammax * gammax) + (BCBC * TAB * TAB - gammay * gammay) - ((ACBC + ACBC) * TAB * TAB - 2 * gammax * gammay);

  q = -(BCBC * TAB * TAB - gammay * gammay)*2 +
    ((ACBC + ACBC) * TAB * TAB - 2 * gammax * gammay) +
    ((ACCD + ACCD) * TAB * TAB - 2 * gammax * gammaz) -
    ((BCCD + BCCD) * TAB * TAB - 2 * gammay * gammaz);

  r = (TAB * TAB * BCBC - gammay * gammay) + ((BCCD + BCCD) * TAB * TAB - 2 * gammay * gammaz) + (TAB * TAB * CDCD - gammaz * gammaz);

  delta = q * q - 4 * p*r;

  if (delta >= 0.0)
  {
    lambda1 = (-q + sqrt(delta)) / (2.0 * p);
    lambda2 = 1 - lambda1;
    lambda3 = 0.0;

    if (lambda1 >= 0 && lambda1 <= 1)
    {
      TE = TA * lambda1 + TB*lambda2;
      FaceTAB = fmin(FaceTAB, TE + sqrt((lambda1 * ACAC + lambda2 * ACBC + ACCD) * lambda1 + (lambda1 * ACBC + lambda2 * BCBC + BCCD) * lambda2 + (lambda1 * ACCD + lambda2 * BCCD + CDCD)));
    }

    lambda1 = (-q - sqrt(delta)) / (2.0 * p);
    lambda2 = 1 - lambda1;
    lambda3 = 0.0;

    if (lambda1 >= 0 && lambda1 <= 1)
    {
      TE = TA * lambda1 + TB*lambda2;
      FaceTAB = fmin(FaceTAB, TE + sqrt((lambda1 * ACAC + lambda2 * ACBC + ACCD) * lambda1 + (lambda1 * ACBC + lambda2 * BCBC + BCCD) * lambda2 + (lambda1 * ACCD + lambda2 * BCCD + CDCD)));
    }
  }
  FaceTAB = fmin(FaceTAB, fmin(TB + sqrt((BCBC + BCCD) + (BCCD + CDCD)), TA + sqrt((ACAC + ACCD) + (ACCD + CDCD))));

  //calculate FaceTAC, let lambda2 = 0
  p = ACAC * TAC * TAC - ACAC*ACAC;
  q = (ACCD + ACCD) * TAC * TAC - 2 * ACAC*ACCD;
  r = TAC * TAC * CDCD - ACCD*ACCD;

  delta = q * q - 4 * p*r;

  if (delta >= 0.0)
  {
    lambda1 = (-q + sqrt(delta)) / (2.0 * p);
    lambda2 = 0.0;
    lambda3 = 1 - lambda1 - lambda2;

    if (lambda1 >= 0 && lambda1 <= 1)
    {
      TE = TA * lambda1 + TC*lambda3;
      FaceTAC = fmin(FaceTAC, TE + sqrt((lambda1 * ACAC + ACCD) * lambda1 + (lambda1 * ACCD + CDCD)));
    }

    lambda1 = (-q - sqrt(delta)) / (2.0 * p);
    lambda2 = 0.0;
    lambda3 = 1 - lambda1 - lambda2;

    if (lambda1 >= 0 && lambda1 <= 1)
    {
      TE = TA * lambda1 + TB * lambda2 + TC*lambda3;
      FaceTAC = fmin(FaceTAC, TE + sqrt((lambda1 * ACAC + ACCD) * lambda1 + (lambda1 * ACCD + CDCD)));
    }
  }

  FaceTAC = fmin(FaceTAC, fmin(TA + sqrt((ACAC + ACCD) + (ACCD + CDCD)), TC + sqrt(CDCD)));

  ////////Done calculating FaceTAC/////////////////////////

  float s = TAC * ACBC - TBC*ACAC;
  float t = TAC * BCCD - TBC*ACCD;
  float h = -(TAC * BCBC - TBC * ACBC);

  p = (TAC * TAC * ACAC - ACAC * ACAC) * h * h + (BCBC * TAC * TAC - ACBC * ACBC) * s * s + ((ACBC + ACBC) * TAC * TAC - 2 * ACAC * ACBC) * s*h;

  q = (BCBC * TAC * TAC - ACBC * ACBC)*2 * s * t +
    ((ACBC + ACBC) * TAC * TAC - 2 * ACAC * ACBC) * t * h +
    ((ACCD + ACCD) * TAC * TAC - 2 * ACAC * ACCD) * h * h +
    ((BCCD + BCCD) * TAC * TAC - 2 * ACBC * ACCD) * s*h;

  r = (TAC * TAC * BCBC - ACBC * ACBC) * t * t + ((BCCD + BCCD) * TAC * TAC - 2 * ACBC * ACCD) * t * h + (TAC * TAC * CDCD - ACCD * ACCD) * h*h;

  delta = q * q - 4 * p*r;

  if (delta >= 0.0)
  {
    lambda1 = (-q + sqrt(delta)) / (2.0 * p);
    lambda2 = (s * lambda1 + t) / (h + SMALLNUM);
    lambda3 = 1 - lambda1 - lambda2;

    if (lambda1 >= 0 && lambda1 <= 1 && lambda2 >= 0 && lambda2 <= 1 && lambda3 >= 0 && lambda3 <= 1)
    {
      TE = TA * lambda1 + TB * lambda2 + TC*lambda3;
      TD = fmin(TD, TE + sqrt((lambda1 * ACAC + lambda2 * ACBC + ACCD) * lambda1 + (lambda1 * ACBC + lambda2 * BCBC + BCCD) * lambda2 + (lambda1 * ACCD + lambda2 * BCCD + CDCD)));
    }

    lambda1 = (-q - sqrt(delta)) / (2.0 * p);
    lambda2 = (s * lambda1 + t) / (h + SMALLNUM);
    lambda3 = 1 - lambda1 - lambda2;

    if (lambda1 >= 0 && lambda1 <= 1 && lambda2 >= 0 && lambda2 <= 1 && lambda3 >= 0 && lambda3 <= 1)
    {
      TE = TA * lambda1 + TB * lambda2 + TC*lambda3;
      TD = fmin(TD, TE + sqrt((lambda1 * ACAC + lambda2 * ACBC + ACCD) * lambda1 + (lambda1 * ACBC + lambda2 * BCBC + BCCD) * lambda2 + (lambda1 * ACCD + lambda2 * BCCD + CDCD)));
    }

    TD = fmin(TD, fmin(FaceTBC, fmin(FaceTAB, FaceTAC)));
  }
  else
  {
    TD = fmin(TD, fmin(FaceTBC, fmin(FaceTAB, FaceTAC)));
  }

  return TD;
}

__global__ void kernel_update_values(int* active_block_list, int* seed_label, int largest_ele_part, int largest_vert_part, int full_ele_num, int* ele, int* ele_offsets,
    int* vert_offsets, float* vertT, float* ele_local_coords,
    int largest_num_inside_mem, int* mem_locations, int* mem_location_offsets, const int NITER, float* vertT_out, int* con)
{
  int bidx = active_block_list[blockIdx.x];
  int tidx = threadIdx.x;
  int ele_start = ele_offsets[bidx];
  int ele_end = ele_offsets[bidx + 1];
  int vert_start = vert_offsets[bidx];
  int vert_end = vert_offsets[bidx + 1];

  int nv = vert_end - vert_start;
  int ne = ele_end - ele_start;
  float oldT, newT;

  extern __shared__ char s_array[];
  float* s_vertT = (float*)s_array;
  float* s_eleT = (float*)s_array;
  short* s_mem = (short*)&s_vertT[largest_vert_part];
  short l_mem[32] = {-1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1};
  int count = 0;
  if (tidx < nv)
  {
    int mem_start = mem_location_offsets[vert_start + tidx];
    int mem_end = mem_location_offsets[vert_start + tidx + 1];
    s_vertT[tidx] = vertT[vert_start + tidx];
    oldT = s_vertT[tidx];
    newT = oldT;
    for (int i = mem_start; i < mem_end; i++)
    {
      int lmem = mem_locations[i];
      if ((lmem % full_ele_num) >= ele_start && (lmem % full_ele_num) < ele_end)
      {
        int local_ele_index = (lmem % full_ele_num) - ele_start;
        int ele_off = lmem / full_ele_num;
        s_mem[tidx * largest_num_inside_mem + count] = (short)(local_ele_index * 4 + ele_off);
        count++;
      }
    }
  }
  __syncthreads();

  if (tidx < nv)
  {
#pragma unroll
    for (int i = 0; i < 32; i++)
    {
      if (i < count)
        l_mem[i] = s_mem[tidx * largest_num_inside_mem + i];
    }
  }

  float eleT[4];
  if (tidx < ne)
  {
    for (int i = 0; i < 4; i++)
    {
      int global_vidx = ele[i * full_ele_num + ele_start + tidx];
      if (global_vidx >= vert_start && global_vidx < vert_end)
      {
        short local_vidx = (short)(global_vidx - vert_start);
        eleT[i] = s_vertT[local_vidx];
      }
      else
      {
        eleT[i] = vertT[global_vidx];
      }
    }
  }

  __syncthreads();

  if (tidx < ne)
  {
    for (int i = 0; i < 4; i++)
    {
      s_eleT[4 * tidx + i] = eleT[i];
    }
  }
  __syncthreads();

  float TD, TA, TB, TC;

  float L[6];
  if (tidx < ne)
  {
#pragma unroll
    for (int i = 0; i < 6; i++)
    {
      L[i] = ele_local_coords[i * full_ele_num + (tidx + ele_start) ];
    }
    float ACBC = L[1]*(L[1] - L[0]) + L[2] * L[2];
    float BCCD = (L[1] - L[0])*(L[3] - L[1]) + L[2] * (L[4] - L[2]);
    float ACCD = L[1]*(L[3] - L[1]) + L[2] * (L[4] - L[2]);
    float ADBD = L[3]*(L[3] - L[0]) + L[4] * L[4] + L[5] * L[5];
    float ACAD = L[1] * L[3] + L[2] * L[4];
    float BCBD = (L[1] - L[0])*(L[3] - L[0]) + L[2] * L[4];

    float ADBC = ACBC + BCCD;
    float ACBD = ACBC + ACCD;
    float CDBD = ADBD - ACBD;
    float ADCD = ADBD - ADBC;
    float CDCD = ADCD - ACCD;
    float ADAD = ADCD + ACAD;
    float ACAC = ACAD - ACCD;
    float BDBD = BCBD + CDBD;
    float BCBC = BCBD - BCCD;

    for (int iter = 0; iter < NITER; iter++)
    {
      if (tidx < nv)
      {
        oldT = s_eleT[l_mem[0]];
      }
      __syncthreads();

      TA = s_eleT[tidx * 4 + 0];
      TB = s_eleT[tidx * 4 + 1];
      TC = s_eleT[tidx * 4 + 2];
      TD = s_eleT[tidx * 4 + 3];

      s_eleT[tidx * 4 + 3] = fmin(TD, localSolverTet1(TA, TB, TC, ACAC, ACBC, ACCD, BCBC, BCCD, CDCD));
      s_eleT[tidx * 4 + 0] = fmin(TA, localSolverTet1(TB, TD, TC, BCBC, -BCCD, -ACBC, CDCD, ACCD, ACAC));
      s_eleT[tidx * 4 + 1] = fmin(TB, localSolverTet1(TA, TD, TC, ACAC, -ACCD, -ACBC, CDCD, BCCD, BCBC));
      s_eleT[tidx * 4 + 2] = fmin(TC, localSolverTet1(TA, TB, TD, ADAD, ADBD, -ADCD, BDBD, -CDBD, CDCD));
      __syncthreads();


      if (tidx < nv)
      {
        newT = s_eleT[l_mem[0]];
#pragma unroll
        for (int i = 1; i < 32; i++)
        {
          short lmem = l_mem[i];
          if (lmem > -1)
            newT = fmin(s_eleT[lmem], newT);
        }

#pragma unroll
        for (int i = 0; i < 32; i++)
        {
          short lmem = l_mem[i];
          if (lmem > -1)
            s_eleT[lmem] = newT;
        }
      }
      __syncthreads();
    }

    if (tidx < nv)
    {
      if (seed_label[vert_start + tidx] != redistance::SeedPoint)
      {
        vertT_out[vert_start + tidx] = newT;
        if (oldT - newT < SMALLNUM) con[vert_start + tidx] = true;
        else con[vert_start + tidx] = false;
      }
      else
        con[vert_start + tidx] = true;
    }
  }
}

__global__ void kernel_run_check_neghbor(int* active_block_list, int* seed_label, int largest_ele_part, int largest_vert_part, int full_ele_num, int* ele, int* ele_offsets,
    int* vert_offsets, float* vertT, float* ele_local_coords,
    int largest_num_inside_mem, int* mem_locations, int* mem_location_offsets, const int NITER, float* vertT_out, int* con)
{
  int bidx = active_block_list[blockIdx.x];
  int tidx = threadIdx.x;
  int ele_start = ele_offsets[bidx];
  int ele_end = ele_offsets[bidx + 1];
  int vert_start = vert_offsets[bidx];
  int vert_end = vert_offsets[bidx + 1];

  int nv = vert_end - vert_start;
  int ne = ele_end - ele_start;
  float oldT, newT;

  extern __shared__ char s_array[];
  float* s_vertT = (float*)s_array;
  float* s_eleT = (float*)s_array;
  short* s_mem = (short*)&s_vertT[largest_vert_part];
  short l_mem[32] = {-1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1};
  int count = 0;
  if (tidx < nv)
  {
    int mem_start = mem_location_offsets[vert_start + tidx];
    int mem_end = mem_location_offsets[vert_start + tidx + 1];
    s_vertT[tidx] = vertT[vert_start + tidx];
    oldT = s_vertT[tidx];
    newT = oldT;
    for (int i = mem_start; i < mem_end; i++)
    {
      int lmem = mem_locations[i];
      if ((lmem % full_ele_num) >= ele_start && (lmem % full_ele_num) < ele_end)
      {
        int local_ele_index = (lmem % full_ele_num) - ele_start;
        int ele_off = lmem / full_ele_num;
        s_mem[tidx * largest_num_inside_mem + count] = (short)(local_ele_index * 4 + ele_off);
        count++;
      }
    }
  }

  __syncthreads();

  if (tidx < nv)
  {
#pragma unroll
    for (int i = 0; i < 32; i++)
    {
      if (i < count)
        l_mem[i] = s_mem[tidx * largest_num_inside_mem + i];
    }
  }

  float eleT[4];
  if (tidx < ne)
  {
    for (int i = 0; i < 4; i++)
    {
      int global_vidx = ele[i * full_ele_num + ele_start + tidx];
      if (global_vidx >= vert_start && global_vidx < vert_end)
      {
        short local_vidx = (short)(global_vidx - vert_start);
        eleT[i] = s_vertT[local_vidx];
      }
      else
      {
        eleT[i] = vertT[global_vidx];
      }
    }
  }

  __syncthreads();

  if (tidx < ne)
  {
    for (int i = 0; i < 4; i++)
    {
      s_eleT[4 * tidx + i] = eleT[i];
    }
  }
  __syncthreads();

  float TD, TA, TB, TC;

  float L[6];
  if (tidx < ne)
  {
#pragma unroll
    for (int i = 0; i < 6; i++)
    {
      L[i] = ele_local_coords[i * full_ele_num + (tidx + ele_start) ];
    }
    float ACBC = L[1]*(L[1] - L[0]) + L[2] * L[2];
    float BCCD = (L[1] - L[0])*(L[3] - L[1]) + L[2] * (L[4] - L[2]);
    float ACCD = L[1]*(L[3] - L[1]) + L[2] * (L[4] - L[2]);
    float ADBD = L[3]*(L[3] - L[0]) + L[4] * L[4] + L[5] * L[5];
    float ACAD = L[1] * L[3] + L[2] * L[4];
    float BCBD = (L[1] - L[0])*(L[3] - L[0]) + L[2] * L[4];

    float ADBC = ACBC + BCCD;
    float ACBD = ACBC + ACCD;
    float CDBD = ADBD - ACBD;
    float ADCD = ADBD - ADBC;
    float CDCD = ADCD - ACCD;
    float ADAD = ADCD + ACAD;
    float ACAC = ACAD - ACCD;
    float BDBD = BCBD + CDBD;
    float BCBC = BCBD - BCCD;

    for (int iter = 0; iter < NITER; iter++)
    {
      if (tidx < nv)
      {
        oldT = s_eleT[l_mem[0]];
      }
      __syncthreads();

      TA = s_eleT[tidx * 4 + 0];
      TB = s_eleT[tidx * 4 + 1];
      TC = s_eleT[tidx * 4 + 2];
      TD = s_eleT[tidx * 4 + 3];

      s_eleT[tidx * 4 + 3] = fmin(TD, localSolverTet1(TA, TB, TC, ACAC, ACBC, ACCD, BCBC, BCCD, CDCD));
      s_eleT[tidx * 4 + 0] = fmin(TA, localSolverTet1(TB, TD, TC, BCBC, -BCCD, -ACBC, CDCD, ACCD, ACAC));
      s_eleT[tidx * 4 + 1] = fmin(TB, localSolverTet1(TA, TD, TC, ACAC, -ACCD, -ACBC, CDCD, BCCD, BCBC));
      s_eleT[tidx * 4 + 2] = fmin(TC, localSolverTet1(TA, TB, TD, ADAD, ADBD, -ADCD, BDBD, -CDBD, CDCD));
      __syncthreads();


      if (tidx < nv)
      {
        newT = s_eleT[l_mem[0]];
#pragma unroll
        for (int i = 1; i < 32; i++)
        {
          short lmem = l_mem[i];
          if (lmem > -1)
            newT = fmin(s_eleT[lmem], newT);
        }

#pragma unroll
        for (int i = 0; i < 32; i++)
        {
          short lmem = l_mem[i];
          if (lmem > -1)
            s_eleT[lmem] = newT;
        }


      }
      __syncthreads();
    }

    if (tidx < nv)
    {
      if (seed_label[vert_start + tidx] != redistance::SeedPoint)
      {
        if (oldT - newT < SMALLNUM) con[vert_start + tidx] = true;
        else con[vert_start + tidx] = false;
      }
      else
        con[vert_start + tidx] = true;
    }
  }
}

__global__ void kernel_seedlabel(int nn, int full_ele_num, float* vert_after_permute, int* vert_offsets, int* ele_after_permute, int* ele_offsets, int* label, float* vertT_after_permute, float* DT, int* active_block_list)
{
  int bidx = blockIdx.x;
  int tidx = threadIdx.x;
  int ele_start = ele_offsets[bidx];
  int ele_end = ele_offsets[bidx + 1];
  int ne = ele_end - ele_start;
  __shared__ bool isSeed[1];
  if (tidx == 0) isSeed[0] = false;
  __syncthreads();


  if (tidx < ne)
  {
    int e0 = ele_after_permute[0 * full_ele_num + ele_start + tidx];
    int e1 = ele_after_permute[1 * full_ele_num + ele_start + tidx];
    int e2 = ele_after_permute[2 * full_ele_num + ele_start + tidx];
    int e3 = ele_after_permute[3 * full_ele_num + ele_start + tidx];

    float v0 = vertT_after_permute[e0];
    float v1 = vertT_after_permute[e1];
    float v2 = vertT_after_permute[e2];
    float v3 = vertT_after_permute[e3];

    if (v0 == 0.0)
    {
      label[e0] = redistance::SeedPoint;
      isSeed[0] = true;
      DT[e0] = fabs(v0);
    }
    if (v1 == 0.0)
    {
      label[e1] = redistance::SeedPoint;
      isSeed[0] = true;
      DT[e1] = fabs(v1);
    }
    if (v2 == 0.0)
    {
      label[e2] = redistance::SeedPoint;
      isSeed[0] = true;
      DT[e2] = fabs(v2);
    }
    if (v3 == 0.0)
    {
      label[e3] = redistance::SeedPoint;
      isSeed[0] = true;
      DT[e3] = fabs(v3);
    }

    if (!((v0 >= 0 && v1 >= 0 && v2 >= 0 && v3 >= 0) || (v0 <= 0 && v1 <= 0 && v2 <= 0 && v3 <= 0)))
    {
      label[e0] = redistance::SeedPoint;
      label[e1] = redistance::SeedPoint;
      label[e2] = redistance::SeedPoint;
      label[e3] = redistance::SeedPoint;
      isSeed[0] = true;
      DT[e0] = fabs(v0);
      DT[e1] = fabs(v1);
      DT[e2] = fabs(v2);
      DT[e3] = fabs(v3);
    }
  }

  __syncthreads();

  if (tidx == 0)
  {
    if (isSeed[0])
    {
      unsigned int location = atomicInc((unsigned int*)&active_block_list[0], LARGENUM);
      active_block_list[location + 1] = bidx;
    }
  }
}

__global__ void kernel_seedlabel_narrowband(int nn, int full_ele_num, const int* narrowband, float* vert_after_permute, int* vert_offsets, int* ele_after_permute, int* ele_offsets, int* label, float* vertT_after_permute, float* DT, int* active_block_list)
{
  int bidx = narrowband[blockIdx.x];
  int tidx = threadIdx.x;
  int ele_start = ele_offsets[bidx];
  int ele_end = ele_offsets[bidx + 1];
  int ne = ele_end - ele_start;
  __shared__ bool isSeed[1];
  if (tidx == 0) isSeed[0] = false;
  __syncthreads();

  if (tidx < ne)
  {
    int e0 = ele_after_permute[0 * full_ele_num + ele_start + tidx];
    int e1 = ele_after_permute[1 * full_ele_num + ele_start + tidx];
    int e2 = ele_after_permute[2 * full_ele_num + ele_start + tidx];
    int e3 = ele_after_permute[3 * full_ele_num + ele_start + tidx];

    float v0 = vertT_after_permute[e0];
    float v1 = vertT_after_permute[e1];
    float v2 = vertT_after_permute[e2];
    float v3 = vertT_after_permute[e3];

    if (v0 == 0.0)
    {
      label[e0] = redistance::SeedPoint;
      isSeed[0] = true;
      DT[e0] = fabs(v0);
    }
    if (v1 == 0.0)
    {
      label[e1] = redistance::SeedPoint;
      isSeed[0] = true;
      DT[e1] = fabs(v1);
    }
    if (v2 == 0.0)
    {
      label[e2] = redistance::SeedPoint;
      isSeed[0] = true;
      DT[e2] = fabs(v2);
    }
    if (v3 == 0.0)
    {
      label[e3] = redistance::SeedPoint;
      isSeed[0] = true;
      DT[e3] = fabs(v3);
    }

    if (!((v0 >= 0 && v1 >= 0 && v2 >= 0 && v3 >= 0) || (v0 <= 0 && v1 <= 0 && v2 <= 0 && v3 <= 0)))
    {
      label[e0] = redistance::SeedPoint;
      label[e1] = redistance::SeedPoint;
      label[e2] = redistance::SeedPoint;
      label[e3] = redistance::SeedPoint;
      isSeed[0] = true;
      DT[e0] = fabs(v0);
      DT[e1] = fabs(v1);
      DT[e2] = fabs(v2);
      DT[e3] = fabs(v3);
    }
  }

  __syncthreads();

  if (tidx == 0)
  {
    if (isSeed[0])
    {
      unsigned int location = atomicInc((unsigned int*)&active_block_list[0], LARGENUM);
      active_block_list[location + 1] = bidx;
    }
  }
}

__global__ void CopyOutBack(int* active_block_list, int* vert_offsets, float* vertT, float* vertT_out)
{
  int bidx = active_block_list[blockIdx.x];
  int tidx = threadIdx.x;

  int start = vert_offsets[bidx];
  int end = vert_offsets[bidx + 1];
  if (tidx < end - start)
  {
    vertT[tidx + start] = vertT_out[tidx + start];
  }
}

__global__ void kernel_reinit_Tsign(int nn, float* vertT, char* Tsign)
{
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int vidx = tidx; vidx < nn; vidx += blockDim.x*gridDim.x)
  {
    Tsign[vidx] = (vertT[vidx] < 0) ? 0 : 1;
  }
}

__global__ void kernel_recover_Tsign(int* block_list, int* vert_offsets, float* vertT, char* Tsign)
{
  int tidx = threadIdx.x;
  int bidx = block_list[blockIdx.x];
  int vstart = vert_offsets[bidx];
  int vend = vert_offsets[bidx + 1];
  int nv = vend - vstart;
  if (tidx < nv)
  {
    vertT[vstart + tidx] *= (2 * (int)Tsign[vstart + tidx] - 1);
  }
}

__global__ void kernel_recover_Tsign_whole(int nn, float* vertT, char* Tsign)
{
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int vidx = tidx; vidx < nn; vidx += blockDim.x*gridDim.x)
  {
    vertT[vidx] *= (2 * (int)Tsign[vidx] - 1);
  }
}



#endif  /* REDISTANCE_KERNELS_H */

