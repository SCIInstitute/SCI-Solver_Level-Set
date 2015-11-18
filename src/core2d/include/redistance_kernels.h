/* 
 * File:   redistance_kernels.h
 * Author: zhisong
 *
 * Created on October 28, 2012, 4:14 PM
 */

#ifndef REDISTANCE_KERNELS_H
#define	REDISTANCE_KERNELS_H

#include <redistance.h>

struct or_op
{

  __host__ __device__ bool operator () (const bool& x, const bool& y) const
  {
    return x || y;
  }
};

template<int SZ>
__global__ void kernel_compute_new_narrowband(int* new_narrowband, double* vertT, int* vert_offsets, double bandwidth)
{
  int block_idx = blockIdx.x;
  int tx = threadIdx.x;
  int start = vert_offsets[block_idx];
  int end = vert_offsets[block_idx + 1];
  int blocksize = end - start;
  __shared__ double sdata[SZ];
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
__global__ void run_reduction_bandwidth(int *con, int *blockCon, int* ActiveList, double* vertT, double* block_vertT_min, int* vert_offsets)
{
  int list_idx = blockIdx.x;
  int tx = threadIdx.x;
  int block_idx = ActiveList[list_idx];
  int start = vert_offsets[block_idx];
  int end = vert_offsets[block_idx + 1];
  int blocksize = end - start;
  __shared__ int s_block_conv;
  __shared__ double sdata[SZ];

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

__device__ double localSolverTet1(double TA, double TB, double TC, double ACAC, double ACBC, double ACCD, double BCBC, double BCCD, double CDCD)
{
  if (TA >= LARGENUM && TB >= LARGENUM && TC >= LARGENUM)
    return LARGENUM;
  double p, q, r;
  double lambda1, lambda2, lambda3;
  double FaceTAC = LARGENUM, FaceTAB = LARGENUM, FaceTBC = LARGENUM;
  double delta, TE;
  double TD = LARGENUM;
  double TAC = TC - TA;
  double TBC = TC - TB;
  double TAB = TB - TA;

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
  double gammax = ACAC - ACBC, gammay = ACBC - BCBC, gammaz = ACCD - BCCD;
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

  double s = TAC * ACBC - TBC*ACAC;
  double t = TAC * BCCD - TBC*ACCD;
  double h = -(TAC * BCBC - TBC * ACBC);

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

__device__ double localSolverTri(double TA, double TB, double LenAB, double LenBC, double LenAC)
{
  if (TA >= LARGENUM && TB >= LARGENUM)
    return LARGENUM;
  double TAB, a, b, delta, lambda1, lambda2, EdgeTA, EdgeTB, cosA, LenAD, LenCD;
  double TC1 = LARGENUM;
  double TC2 = LARGENUM;
  double TC = LARGENUM;
  double speedI = 1.0;
  TAB = TB - TA;

  a = (speedI * speedI * LenAB * LenAB - TAB * TAB) * LenAB*LenAB;

  EdgeTA = TA + LenAC * speedI;
  EdgeTB = TB + LenBC * speedI;

  if (a > 0.0f)
  {
    cosA = (LenAC * LenAC + LenAB * LenAB - LenBC * LenBC) / (2.0f * LenAC * LenAB);

    b = 2.0f * LenAB * LenAC * cosA * (TAB * TAB - speedI * speedI * LenAB * LenAB);
    delta = 4.0f * LenAC * LenAC * a * TAB * TAB * (1.0f - cosA * cosA);

    lambda1 = (-b + sqrtf(delta)) / (2.0f * a);
    lambda2 = (-b - sqrtf(delta)) / (2.0f * a);

    if (lambda1 >= 0.0f && lambda1 <= 1.0f)
    {
      LenAD = lambda1*LenAB;
      LenCD = sqrtf(LenAC * LenAC + LenAD * LenAD - 2 * LenAC * LenAD * cosA);
      TC1 = lambda1 * TAB + TA + LenCD*speedI;
    }
    if (lambda2 >= 0.0f && lambda2 <= 1.0f)
    {
      LenAD = lambda2*LenAB;
      LenCD = sqrtf(LenAC * LenAC + LenAD * LenAD - 2 * LenAC * LenAD * cosA);
      TC2 = lambda2 * TAB + TA + LenCD*speedI;
    }

    TC = fmin(TC2, fmin(TC1, fmin(EdgeTA, EdgeTB)));
  }
  else
  {
    TC = fmin(EdgeTA, EdgeTB);
  }

  return TC;
}

__global__ void kernel_update_values(int* active_block_list, int* seed_label, int largest_ele_part, int largest_vert_part, int full_ele_num, int* ele, int* ele_offsets,
                                     int* vert_offsets, double* vertT, double* ele_local_coords,
                                     int largest_num_inside_mem, int* mem_locations, int* mem_location_offsets, const int NITER, double* vertT_out, int* con)
{
  int bidx = active_block_list[blockIdx.x];
  int tidx = threadIdx.x;
  int ele_start = ele_offsets[bidx];
  int ele_end = ele_offsets[bidx + 1];
  int vert_start = vert_offsets[bidx];
  int vert_end = vert_offsets[bidx + 1];

  int nv = vert_end - vert_start;
  int ne = ele_end - ele_start;
  double oldT, newT;

  extern __shared__ char s_array[];
  double* s_vertT = (double*)s_array;
  double* s_eleT = (double*)s_array;
  short* s_mem = (short*)&s_vertT[largest_vert_part];
  short l_mem[16] = {-1, -1, -1, -1, -1, -1, -1, -1,
                     -1, -1, -1, -1, -1, -1, -1, -1};
  //  short* l_mem = (short*)&s_eleT[4 * largest_ele_part];
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
        s_mem[tidx * largest_num_inside_mem + count] = (short)(local_ele_index * 3 + ele_off);
        //        l_mem[tidx * largest_num_inside_mem + count] = (short)(local_ele_index * 4 + ele_off);
        count++;
      }
    }
  }
  __syncthreads();

  if (tidx < nv)
  {
#pragma unroll
    for (int i = 0; i < 16; i++)
    {
      if (i < count)
        l_mem[i] = s_mem[tidx * largest_num_inside_mem + i];
    }
  }

  double eleT[3];
  if (tidx < ne)
  {
    for (int i = 0; i < 3; i++)
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
    for (int i = 0; i < 3; i++)
    {
      s_eleT[3 * tidx + i] = eleT[i];
    }
  }
  __syncthreads();

  double TA, TB, TC;

  double LenAB, LenBC, LenCA;
  if (tidx < ne)
  {
    double c0 = ele_local_coords[0 * full_ele_num + (tidx + ele_start)];
    double c1 = ele_local_coords[1 * full_ele_num + (tidx + ele_start)];
    double c2 = ele_local_coords[2 * full_ele_num + (tidx + ele_start)];

    LenAB = fabs(c0);
    LenBC = sqrt(c2 * c2 + (c1 - c0)*(c1 - c0));
    LenCA = sqrt(c1 * c1 + c2 * c2);

    for (int iter = 0; iter < NITER; iter++)
    {
      if (tidx < nv)
      {
        //        oldT = s_eleT[l_mem[tidx * largest_num_inside_mem + 0]];
        oldT = s_eleT[l_mem[0]];
      }
      __syncthreads();

      TA = s_eleT[tidx * 3 + 0];
      TB = s_eleT[tidx * 3 + 1];
      TC = s_eleT[tidx * 3 + 2];
      //      if (bidx == 4 && tidx == 1 && iter == 0) printf("bidx=%d, tidx=%d, iter=%d, TA=%f, TB=%f, TC=%f, TD=%f\n", bidx, tidx, iter, TA, TB, TC, TD);

      s_eleT[tidx * 3 + 2] = fmin(TC, localSolverTri(TA, TB, LenAB, LenBC, LenCA));
      s_eleT[tidx * 3 + 0] = fmin(TA, localSolverTri(TB, TC, LenBC, LenCA, LenAB));
      s_eleT[tidx * 3 + 1] = fmin(TB, localSolverTri(TC, TA, LenCA, LenAB, LenBC));

//			if(tidx == 493 && iter ==0) printf("bidx=%d, tidx=%d, s_eleT[%d]=%f, s_eleT[%d]=%f, s_eleT[%d]=%f, TA=%f, TB=%f, TC=%f, LenAB=%f, LenBC=%f, LenCA=%f, c0=%f, c1=%f, c2=%f\n", 
//																					bidx, tidx, tidx * 3 + 0, s_eleT[tidx * 3 + 0], tidx * 3 + 1, s_eleT[tidx * 3 + 1], tidx * 3 + 2, s_eleT[tidx * 3 + 2], TA, TB,TC, LenAB, LenBC, LenCA, c0, c1, c2);
      __syncthreads();
      //      if (bidx == 4 && tidx == 0) for (int i = 0; i < 11; i++) if (l_mem[i] > -1) printf("iter=%d, i=%d, s_eleT[%d]=%f\n", iter, i, l_mem[i], s_eleT[l_mem[i]]);

      if (tidx < nv)
      {
        newT = s_eleT[l_mem[0]];
        //        newT = s_eleT[l_mem[tidx * largest_num_inside_mem + 0]];
//				if (tidx == 278 && iter == 0) printf("bidx=%d, tidx=%d, oldT=%f, s_eleT[%d]=%f\n", bidx, tidx, oldT, l_mem[0], s_eleT[l_mem[0]]);
#pragma unroll
        for (int i = 1; i < 16; i++)
          //        for (int i = 1; i < count; i++)
        {
          short lmem = l_mem[i];
          //          short lmem = l_mem[tidx * largest_num_inside_mem + i];
          if (lmem > -1)
					{
            newT = fmin(s_eleT[lmem], newT);
//						if (tidx == 278 && iter == 0) printf("bidx=%d, tidx=%d, oldT=%f, s_eleT[%d]=%f\n", bidx, tidx, oldT, lmem, s_eleT[lmem]);
					}
        }

#pragma unroll
        for (int i = 0; i < 16; i++)
          //        for (int i = 0; i < count; i++)
        {
          short lmem = l_mem[i];
          //          short lmem = l_mem[tidx * largest_num_inside_mem + i];
          if (lmem > -1)
            s_eleT[lmem] = newT;
        }

        //        if (tidx == 3) printf("bidx=%d, tidx=%d, vert_start=%d, iter=%d, oldT=%f, newT=%f, vert_start = %d, vert_end = %d\n", bidx, tidx, vert_start, iter, oldT, newT, vert_start, vert_end);
        //        if (oldT - newT < SMALLNUM) converged = true;
        //        else converged = false;
        //        if (bidx == 60)
        //          printf("bidx=%d, tidx=%d, iter=%d, oldT=%f, newT=%f, converged=%d\n", bidx, tidx, iter, oldT, newT, converged);
      }
      __syncthreads();
    }

    if (tidx < nv)
    {
      //      if (vert_start + tidx == 3)
      //        printf("bidx=%d, tidx=%d, vert_start=%d,seed_label=%d, newT=%f\n", bidx, tidx, vert_start, seed_label[vert_start + tidx], newT);
      if (seed_label[vert_start + tidx] != redistance::SeedPoint)
      {
        vertT_out[vert_start + tidx] = newT;
        if (oldT - newT < SMALLNUM) con[vert_start + tidx] = true;
        else con[vert_start + tidx] = false;
        //        con[vert_start + tidx] = converged;
      }
      else
        con[vert_start + tidx] = true;
    }
  }
}

__global__ void kernel_run_check_neghbor(int* active_block_list, int* seed_label, int largest_ele_part, int largest_vert_part, int full_ele_num, int* ele, int* ele_offsets,
                                         int* vert_offsets, double* vertT, double* ele_local_coords,
                                         int largest_num_inside_mem, int* mem_locations, int* mem_location_offsets, const int NITER, double* vertT_out, int* con)
{
  int bidx = active_block_list[blockIdx.x];
  int tidx = threadIdx.x;
  int ele_start = ele_offsets[bidx];
  int ele_end = ele_offsets[bidx + 1];
  int vert_start = vert_offsets[bidx];
  int vert_end = vert_offsets[bidx + 1];

  int nv = vert_end - vert_start;
  int ne = ele_end - ele_start;
  double oldT, newT;

  extern __shared__ char s_array[];
  double* s_vertT = (double*)s_array;
  double* s_eleT = (double*)s_array;
  short* s_mem = (short*)&s_vertT[largest_vert_part];
  short l_mem[16] = {-1, -1, -1, -1, -1, -1, -1, -1,
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
        s_mem[tidx * largest_num_inside_mem + count] = (short)(local_ele_index * 3 + ele_off);
        count++;
      }
    }
  }

  __syncthreads();

  if (tidx < nv)
  {
#pragma unroll
    for (int i = 0; i < 16; i++)
    {
      if (i < count)
        l_mem[i] = s_mem[tidx * largest_num_inside_mem + i];
    }
  }
  //  if (bidx == 4 && tidx == 0) for (int i = 0; i < 32; i++) printf("%d %d %d l_mem[%d]=%d\n", bidx, tidx, count, i, l_mem[i]);

  double eleT[3];
  if (tidx < ne)
  {
    for (int i = 0; i < 3; i++)
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
    for (int i = 0; i < 3; i++)
    {
      s_eleT[3 * tidx + i] = eleT[i];
    }
  }
  __syncthreads();

  double TA, TB, TC;
  double LenAB, LenBC, LenCA;
  if (tidx < ne)
  {
    double c0 = ele_local_coords[0 * full_ele_num + (tidx + ele_start)];
    double c1 = ele_local_coords[1 * full_ele_num + (tidx + ele_start)];
    double c2 = ele_local_coords[2 * full_ele_num + (tidx + ele_start)];

    LenAB = fabs(c0);
    LenBC = sqrt(c2 * c2 + (c1 - c0)*(c1 - c0));
    LenCA = sqrt(c1 * c1 + c2 * c2);

    for (int iter = 0; iter < NITER; iter++)
    {
      if (tidx < nv)
      {
        //        oldT = s_eleT[l_mem[tidx * largest_num_inside_mem + 0]];
        oldT = s_eleT[l_mem[0]];
      }
      __syncthreads();

      TA = s_eleT[tidx * 3 + 0];
      TB = s_eleT[tidx * 3 + 1];
      TC = s_eleT[tidx * 3 + 2];
      //      if (bidx == 4 && tidx == 1 && iter == 0) printf("bidx=%d, tidx=%d, iter=%d, TA=%f, TB=%f, TC=%f, TD=%f\n", bidx, tidx, iter, TA, TB, TC, TD);

      s_eleT[tidx * 3 + 2] = fmin(TC, localSolverTri(TA, TB, LenAB, LenBC, LenCA));
      s_eleT[tidx * 3 + 0] = fmin(TA, localSolverTri(TB, TC, LenBC, LenCA, LenAB));
      s_eleT[tidx * 3 + 1] = fmin(TB, localSolverTri(TC, TA, LenCA, LenAB, LenBC));
      __syncthreads();
      //      if (bidx == 4 && tidx == 0) for (int i = 0; i < 11; i++) if (l_mem[i] > -1) printf("iter=%d, i=%d, s_eleT[%d]=%f\n", iter, i, l_mem[i], s_eleT[l_mem[i]]);

      if (tidx < nv)
      {
        newT = s_eleT[l_mem[0]];
        //        newT = s_eleT[l_mem[tidx * largest_num_inside_mem + 0]];
#pragma unroll
        for (int i = 1; i < 16; i++)
          //        for (int i = 1; i < count; i++)
        {
          short lmem = l_mem[i];
          //          short lmem = l_mem[tidx * largest_num_inside_mem + i];
          if (lmem > -1)
            newT = fmin(s_eleT[lmem], newT);
        }

#pragma unroll
        for (int i = 0; i < 16; i++)
          //        for (int i = 0; i < count; i++)
        {
          short lmem = l_mem[i];
          //          short lmem = l_mem[tidx * largest_num_inside_mem + i];
          if (lmem > -1)
            s_eleT[lmem] = newT;
        }

        //        if (tidx == 3) printf("bidx=%d, tidx=%d, vert_start=%d, iter=%d, oldT=%f, newT=%f, vert_start = %d, vert_end = %d\n", bidx, tidx, vert_start, iter, oldT, newT, vert_start, vert_end);
        //        if (oldT - newT < SMALLNUM) converged = true;
        //        else converged = false;
        //        if (bidx == 60)
        //          printf("bidx=%d, tidx=%d, iter=%d, oldT=%f, newT=%f, converged=%d\n", bidx, tidx, iter, oldT, newT, converged);
      }
      __syncthreads();
    }

    if (tidx < nv)
    {
      if (seed_label[vert_start + tidx] != redistance::SeedPoint)
      {
        //        vertT_out[vert_start + tidx] = newT;
        if (oldT - newT < SMALLNUM) con[vert_start + tidx] = true;
        else con[vert_start + tidx] = false;
      }
      else
        con[vert_start + tidx] = true;
    }
  }
}

__global__ void kernel_seedlabel(int nn, int full_ele_num, double* vert_after_permute, int* vert_offsets, int* ele_after_permute, int* ele_offsets, int* label, double* vertT_after_permute, double* DT, int* active_block_list)
{
  int bidx = blockIdx.x;
  int tidx = threadIdx.x;
  int ele_start = ele_offsets[bidx];
  int ele_end = ele_offsets[bidx + 1];
  int ne = ele_end - ele_start;
  __shared__ bool isSeed[1];
  if (tidx == 0) isSeed[0] = false;
  __syncthreads();

  //load values to shared memory
  //  extern __shared__ double s_vertT[];
  //  for (int vidx = vert_start; vidx < vert_end; vidx++)
  //  {
  //      s_vertT[vidx - vert_start] = vertT_after_permute[vidx];
  //  }
  //  __syncthreads();

  if (tidx < ne)
  {
    int e0 = ele_after_permute[0 * full_ele_num + ele_start + tidx];
    int e1 = ele_after_permute[1 * full_ele_num + ele_start + tidx];
    int e2 = ele_after_permute[2 * full_ele_num + ele_start + tidx];

    double v0 = vertT_after_permute[e0];
    double v1 = vertT_after_permute[e1];
    double v2 = vertT_after_permute[e2];
    //    if (e0 == 3 || e1 == 3 || e2 == 3 || e3 == 3)
    //    {
    //      printf("some is three: %d %d %f %f %f %f %d %d %d %d\n", bidx, tidx, v0, v1, v2, v3, e0, e1, e2, e3);
    //    }
    //    if (bidx == 0 && (tidx == 25 || tidx == 26))
    //    {
    //      printf("%d %d %f %f %f %f %d %d %d %d\n", bidx, tidx, v0, v1, v2, v3, e0, e1, e2, e3);
    //    }

    if (v0 == 0.0)
    {
      //      if (bidx == 0 && tidx == 25) printf("v0 == 0.0\n");
      label[e0] = redistance::SeedPoint;
      //			if (e0==3) printf("e0 is set: %d %d %f %f %f %f %d %d %d %d\n", bidx, tidx, v0, v1, v2, v3, e0, e1, e2, e3);
      isSeed[0] = true;
      DT[e0] = fabs(v0);
    }
    if (v1 == 0.0)
    {
      //      if (bidx == 0 && tidx == 25) printf("v1 == 0.0\n");
      label[e1] = redistance::SeedPoint;
      //			if (e1==3) printf("e1 is set: %d %d %f %f %f %f %d %d %d %d\n", bidx, tidx, v0, v1, v2, v3, e0, e1, e2, e3);
      isSeed[0] = true;
      DT[e1] = fabs(v1);
    }
    if (v2 == 0.0)
    {
      //      if (bidx == 0 && tidx == 25) printf("v2 == 0.0\n");
      label[e2] = redistance::SeedPoint;
      //			if (e2==3) printf("e2 is set: %d %d %f %f %f %f %d %d %d %d\n", bidx, tidx, v0, v1, v2, v3, e0, e1, e2, e3);
      isSeed[0] = true;
      DT[e2] = fabs(v2);
    }

    if (!((v0 >= 0 && v1 >= 0 && v2 >= 0) || (v0 <= 0 && v1 <= 0 && v2 <= 0)))
    {
      //      if (bidx == 0 && tidx == 25) printf("four v\n");
      label[e0] = redistance::SeedPoint;
      label[e1] = redistance::SeedPoint;
      label[e2] = redistance::SeedPoint;
      //			if (e0 == 3 || e1 == 3 || e2 == 3 || e3 == 3)
      //      {
      //        printf("3 is set: %d %d %f %f %f %f %d %d %d %d\n", bidx, tidx, v0, v1, v2, v3, e0, e1, e2, e3);
      //      }
      isSeed[0] = true;
      DT[e0] = fabs(v0);
      DT[e1] = fabs(v1);
      DT[e2] = fabs(v2);
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

__global__ void kernel_seedlabel_narrowband(int nn, int full_ele_num, const int* narrowband, double* vert_after_permute, int* vert_offsets, int* ele_after_permute, int* ele_offsets, int* label, double* vertT_after_permute, double* DT, int* active_block_list)
{
  int bidx = narrowband[blockIdx.x];
  int tidx = threadIdx.x;
  int ele_start = ele_offsets[bidx];
  int ele_end = ele_offsets[bidx + 1];
  int ne = ele_end - ele_start;
  __shared__ bool isSeed[1];
  if (tidx == 0) isSeed[0] = false;
  __syncthreads();

  //load values to shared memory
  //  extern __shared__ double s_vertT[];
  //  for (int vidx = vert_start; vidx < vert_end; vidx++)
  //  {
  //      s_vertT[vidx - vert_start] = vertT_after_permute[vidx];
  //  }
  //  __syncthreads();

  if (tidx < ne)
  {
    int e0 = ele_after_permute[0 * full_ele_num + ele_start + tidx];
    int e1 = ele_after_permute[1 * full_ele_num + ele_start + tidx];
    int e2 = ele_after_permute[2 * full_ele_num + ele_start + tidx];

    double v0 = vertT_after_permute[e0];
    double v1 = vertT_after_permute[e1];
    double v2 = vertT_after_permute[e2];
    //    if (e0 == 3 || e1 == 3 || e2 == 3 || e3 == 3)
    //    {
    //      printf("some is three: %d %d %f %f %f %f %d %d %d %d\n", bidx, tidx, v0, v1, v2, v3, e0, e1, e2, e3);
    //    }
    //    if (bidx == 0 && (tidx == 25 || tidx == 26))
    //    {
    //      printf("%d %d %f %f %f %f %d %d %d %d\n", bidx, tidx, v0, v1, v2, v3, e0, e1, e2, e3);
    //    }

    if (v0 == 0.0)
    {
      //      if (bidx == 0 && tidx == 25) printf("v0 == 0.0\n");
      label[e0] = redistance::SeedPoint;
      //			if (e0==3) printf("e0 is set: %d %d %f %f %f %f %d %d %d %d\n", bidx, tidx, v0, v1, v2, v3, e0, e1, e2, e3);
      isSeed[0] = true;
      DT[e0] = fabs(v0);
    }
    if (v1 == 0.0)
    {
      //      if (bidx == 0 && tidx == 25) printf("v1 == 0.0\n");
      label[e1] = redistance::SeedPoint;
      //			if (e1==3) printf("e1 is set: %d %d %f %f %f %f %d %d %d %d\n", bidx, tidx, v0, v1, v2, v3, e0, e1, e2, e3);
      isSeed[0] = true;
      DT[e1] = fabs(v1);
    }
    if (v2 == 0.0)
    {
      //      if (bidx == 0 && tidx == 25) printf("v2 == 0.0\n");
      label[e2] = redistance::SeedPoint;
      //			if (e2==3) printf("e2 is set: %d %d %f %f %f %f %d %d %d %d\n", bidx, tidx, v0, v1, v2, v3, e0, e1, e2, e3);
      isSeed[0] = true;
      DT[e2] = fabs(v2);
    }

    if (!((v0 >= 0 && v1 >= 0 && v2 >= 0) || (v0 <= 0 && v1 <= 0 && v2 <= 0)))
    {
      //      if (bidx == 0 && tidx == 25) printf("four v\n");
      label[e0] = redistance::SeedPoint;
      label[e1] = redistance::SeedPoint;
      label[e2] = redistance::SeedPoint;
      //			if (e0 == 3 || e1 == 3 || e2 == 3 || e3 == 3)
      //      {
      //        printf("3 is set: %d %d %f %f %f %f %d %d %d %d\n", bidx, tidx, v0, v1, v2, v3, e0, e1, e2, e3);
      //      }
      isSeed[0] = true;
      DT[e0] = fabs(v0);
      DT[e1] = fabs(v1);
      DT[e2] = fabs(v2);
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

__global__ void CopyOutBack(int* active_block_list, int* vert_offsets, double* vertT, double* vertT_out)
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

__global__ void kernel_reinit_Tsign(int nn, double* vertT, char* Tsign)
{
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int vidx = tidx; vidx < nn; vidx += blockDim.x * gridDim.x)
  {
    Tsign[vidx] = (vertT[vidx] < 0) ? 0 : 1;
  }
}

__global__ void kernel_recover_Tsign(int* block_list, int* vert_offsets, double* vertT, char* Tsign)
{
  int tidx = threadIdx.x;
  int bidx = block_list[blockIdx.x];
  int vstart = vert_offsets[bidx];
  int vend = vert_offsets[bidx + 1];
  int nv = vend - vstart;
  if (tidx < nv)
  {
    vertT[vstart + tidx] *= (2 * Tsign[vstart + tidx] - 1);
  }
}

__global__ void kernel_recover_Tsign_whole(int nn, double* vertT, char* Tsign)
{
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int vidx = tidx; vidx < nn; vidx += blockDim.x * gridDim.x)
  {
    vertT[vidx] *= (2 * (int)Tsign[vidx] - 1);
  }
}

#endif	/* REDISTANCE_KERNELS_H */

