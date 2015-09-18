/*
 * File:   meshFIM_kernels.h
 * Author: zhisong
 *
 * Created on October 24, 2012, 3:29 PM
 */

#ifndef MESHFIM_KERNELS_H
#define  MESHFIM_KERNELS_H

#include <mycutil.h>

struct count_op
{

  __host__ __device__ int operator () (const int& x, const int& y) const
  {
    return (int)(x > 0) + (int)(y > 0);
  }
};

__global__ void kernel_fill_sequence(int nn, int* sequence)
{
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = tidx; i < nn; i += blockDim.x * gridDim.x)
  {
    sequence[i] = i;
  }
}

__global__ void CopyOutBack_levelset(int* narrowband_list, int* vert_offsets, LevelsetValueType* vertT, LevelsetValueType* vertT_out)
{
  int bidx = narrowband_list[blockIdx.x];
  int tidx = threadIdx.x;

  int start = vert_offsets[bidx];
  int end = vert_offsets[bidx + 1];
  if (tidx < end - start)
  {
    vertT[tidx + start] = vertT_out[tidx + start];
  }
}

__global__ void kernel_updateT_single_stage(LevelsetValueType timestep, int* narrowband_list, int largest_ele_part, int largest_vert_part, int full_ele_num, int* ele, int* ele_offsets,
    LevelsetValueType* cadv_local,
    int nn, int* vert_offsets, LevelsetValueType* vert, LevelsetValueType* vertT, LevelsetValueType* ele_local_coords,
    int largest_num_inside_mem, int* mem_locations, int* mem_location_offsets, LevelsetValueType* vertT_out)
{
  int bidx = narrowband_list[blockIdx.x];
  int tidx = threadIdx.x;
  int ele_start = ele_offsets[bidx];
  int ele_end = ele_offsets[bidx + 1];
  int vert_start = vert_offsets[bidx];
  int vert_end = vert_offsets[bidx + 1];

  int nv = vert_end - vert_start;
  int ne = ele_end - ele_start;
  LevelsetValueType vertices[4][3];
  LevelsetValueType sigma[3] = {1.0, 0.0, 0.0};
  LevelsetValueType alphatuda[4] = {0.0, 0.0, 0.0, 0.0};
  LevelsetValueType volume, Hintegral, oldT;
  LevelsetValueType nablaPhi[3] = {0.0, 0.0, 0.0};
  LevelsetValueType nablaN[4][3];
  LevelsetValueType abs_nabla_phi;
  LevelsetValueType theta = 0;

  extern __shared__ char s_array[];
  LevelsetValueType* s_vertT = (LevelsetValueType*)s_array;
  LevelsetValueType* s_vert = (LevelsetValueType*)s_array;
  //temperarily hold the inside_mem_locations
  short* s_mem = (short*)&s_vertT[largest_vert_part];
  LevelsetValueType* s_alphatuda_Hintegral = (LevelsetValueType*)s_array;
  LevelsetValueType* s_alphatuda_volume = (LevelsetValueType*)s_array;
  LevelsetValueType* s_grad_phi = (LevelsetValueType*)s_array;
  LevelsetValueType* s_volume = (LevelsetValueType*)s_array;
  LevelsetValueType* s_curv_up = (LevelsetValueType*)s_array;
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

  LevelsetValueType eleT[4];
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

  if (tidx < nv)
  {
    for (int i = 0; i < 3; i++) s_vert[tidx * 3 + i] = vert[ i * nn + (vert_start + tidx) ];
  }
  __syncthreads();

  if (tidx < ne)
  {

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
      int global_vidx = ele[i * full_ele_num + ele_start + tidx];
      if (global_vidx >= vert_start && global_vidx < vert_end)
      {
#pragma unroll
        for (int j = 0; j < 3; j++)
        {
          vertices[i][j] = s_vert[ (global_vidx - vert_start)*3 + j];
        }
      }
      else
      {
#pragma unroll
        for (int j = 0; j < 3; j++)
        {
          vertices[i][j] = vert[j * nn + global_vidx];
        }
      }
    }
  }
  __syncthreads();

  if (tidx < ne)
  {
    LevelsetValueType cross[3];
    LevelsetValueType v31[3] = {vertices[1][0] - vertices[3][0], vertices[1][1] - vertices[3][1], vertices[1][2] - vertices[3][2]};
    LevelsetValueType v32[3] = {vertices[2][0] - vertices[3][0], vertices[2][1] - vertices[3][1], vertices[2][2] - vertices[3][2]};
    LevelsetValueType v30[3] = {vertices[0][0] - vertices[3][0], vertices[0][1] - vertices[3][1], vertices[0][2] - vertices[3][2]};
    CROSS_PRODUCT(v31, v32, cross);
    LevelsetValueType dotproduct = DOT_PRODUCT(cross, v30);

    volume = fabs(dotproduct) / 6.0;

    //compute inverse of 4 by 4 matrix
    LevelsetValueType a11 = vertices[0][0], a12 = vertices[0][1], a13 = vertices[0][2], a14 = 1.0;
    LevelsetValueType a21 = vertices[1][0], a22 = vertices[1][1], a23 = vertices[1][2], a24 = 1.0;
    LevelsetValueType a31 = vertices[2][0], a32 = vertices[2][1], a33 = vertices[2][2], a34 = 1.0;
    LevelsetValueType a41 = vertices[3][0], a42 = vertices[3][1], a43 = vertices[3][2], a44 = 1.0;

    LevelsetValueType det =
      a11 * a22 * a33 * a44 + a11 * a23 * a34 * a42 + a11 * a24 * a32 * a43
      + a12 * a21 * a34 * a43 + a12 * a23 * a31 * a44 + a12 * a24 * a33 * a41
      + a13 * a21 * a32 * a44 + a13 * a22 * a34 * a41 + a13 * a24 * a31 * a42
      + a14 * a21 * a33 * a42 + a14 * a22 * a31 * a43 + a14 * a23 * a32 * a41
      - a11 * a22 * a34 * a43 - a11 * a23 * a32 * a44 - a11 * a24 * a33 * a42
      - a12 * a21 * a33 * a44 - a12 * a23 * a34 * a41 - a12 * a24 * a31 * a43
      - a13 * a21 * a34 * a42 - a13 * a22 * a31 * a44 - a13 * a24 * a32 * a41
      - a14 * a21 * a32 * a43 - a14 * a22 * a33 * a41 - a14 * a23 * a31 * a42;

    LevelsetValueType b11 = a22 * a33 * a44 + a23 * a34 * a42 + a24 * a32 * a43 - a22 * a34 * a43 - a23 * a32 * a44 - a24 * a33 * a42;
    LevelsetValueType b12 = a12 * a34 * a43 + a13 * a32 * a44 + a14 * a33 * a42 - a12 * a33 * a44 - a13 * a34 * a42 - a14 * a32 * a43;
    LevelsetValueType b13 = a12 * a23 * a44 + a13 * a24 * a42 + a14 * a22 * a43 - a12 * a24 * a43 - a13 * a22 * a44 - a14 * a23 * a42;
    LevelsetValueType b14 = a12 * a24 * a33 + a13 * a22 * a34 + a14 * a23 * a32 - a12 * a23 * a34 - a13 * a24 * a32 - a14 * a22 * a33;

    LevelsetValueType b21 = a21 * a34 * a43 + a23 * a31 * a44 + a24 * a33 * a41 - a21 * a33 * a44 - a23 * a34 * a41 - a24 * a31 * a43;
    LevelsetValueType b22 = a11 * a33 * a44 + a13 * a34 * a41 + a14 * a31 * a43 - a11 * a34 * a43 - a13 * a31 * a44 - a14 * a33 * a41;
    LevelsetValueType b23 = a11 * a24 * a43 + a13 * a21 * a44 + a14 * a23 * a41 - a11 * a23 * a44 - a13 * a24 * a41 - a14 * a21 * a43;
    LevelsetValueType b24 = a11 * a23 * a34 + a13 * a24 * a31 + a14 * a21 * a33 - a11 * a24 * a33 - a13 * a21 * a34 - a14 * a23 * a31;


    LevelsetValueType b31 = a21 * a32 * a44 + a22 * a34 * a41 + a24 * a31 * a42 - a21 * a34 * a42 - a22 * a31 * a44 - a24 * a32 * a41;
    LevelsetValueType b32 = a11 * a34 * a42 + a12 * a31 * a44 + a14 * a32 * a41 - a11 * a32 * a44 - a12 * a34 * a41 - a14 * a31 * a42;
    LevelsetValueType b33 = a11 * a22 * a44 + a12 * a24 * a41 + a14 * a21 * a42 - a11 * a24 * a42 - a12 * a21 * a44 - a14 * a22 * a41;
    LevelsetValueType b34 = a11 * a24 * a32 + a12 * a21 * a34 + a14 * a22 * a31 - a11 * a22 * a34 - a12 * a24 * a31 - a14 * a21 * a32;

    nablaN[0][0] = b11 / det;
    nablaN[0][1] = b21 / det;
    nablaN[0][2] = b31 / det;
    nablaN[1][0] = b12 / det;
    nablaN[1][1] = b22 / det;
    nablaN[1][2] = b32 / det;
    nablaN[2][0] = b13 / det;
    nablaN[2][1] = b23 / det;
    nablaN[2][2] = b33 / det;
    nablaN[3][0] = b14 / det;
    nablaN[3][1] = b24 / det;
    nablaN[3][2] = b34 / det;

    //compuate grad of Phi
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
      nablaPhi[0] += nablaN[i][0] * eleT[i];
      nablaPhi[1] += nablaN[i][1] * eleT[i];
      nablaPhi[2] += nablaN[i][2] * eleT[i];
    }
    abs_nabla_phi = LENGTH(nablaPhi);

    //compute K and Kplus and Kminus
    LevelsetValueType Kplus[4];
    LevelsetValueType Kminus[4];
    LevelsetValueType K[4];
    Hintegral = 0.0;
    LevelsetValueType beta = 0;
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
      K[i] = volume * DOT_PRODUCT(sigma, nablaN[i]); // for H(\nabla u) = sigma DOT \nabla u
      Hintegral += K[i] * eleT[i];
      Kplus[i] = fmax(K[i], (LevelsetValueType)0.0);
      Kminus[i] = fmin(K[i], (LevelsetValueType)0.0);
      beta += Kminus[i];
    }

    beta = 1.0 / beta;

    if (fabs(Hintegral) > 1e-8)
    {
      LevelsetValueType delta[4];
#pragma unroll
      for (int i = 0; i < 4; i++)
      {
        delta[i] = Kplus[i] * beta * (Kminus[0] * (eleT[i] - eleT[0]) + Kminus[1] * (eleT[i] - eleT[1]) + Kminus[2] * (eleT[i] - eleT[2]) + Kminus[3] * (eleT[i] - eleT[3]));
      }

      LevelsetValueType alpha[4];
#pragma unroll
      for (int i = 0; i < 4; i++)
      {
        alpha[i] = delta[i] / Hintegral;
      }

#pragma unroll
      for (int i = 0; i < 4; i++)
      {
        theta += fmax((LevelsetValueType)0.0, alpha[i]);
      }
#pragma unroll
      for (int i = 0; i < 4; i++)
      {
        alphatuda[i] = fmax(alpha[i], (LevelsetValueType)0.0) / theta;
      }
    }
  }

  __syncthreads();

  if (tidx < ne)
  {
    for (int i = 0; i < 4; i++)
    {
      s_alphatuda_Hintegral[tidx * 4 + i] = alphatuda[i] * Hintegral;
    }

  }
  __syncthreads();

  LevelsetValueType up = 0.0, down = 0.0;
  if (tidx < nv)
  {
#pragma unroll
    for (int i = 0; i < 32; i++)
    {
      short lmem = l_mem[i];
      if (lmem > -1)
      {
        up += s_alphatuda_Hintegral[lmem];
      }
    }
  }
  __syncthreads();

  if (tidx < ne)
  {
    for (int i = 0; i < 4; i++)
    {
      s_alphatuda_volume[tidx * 4 + i] = alphatuda[i] * volume;
    }
  }
  __syncthreads();

  if (tidx < nv)
  {
#pragma unroll
    for (int i = 0; i < 32; i++)
    {
      short lmem = l_mem[i];
      if (lmem > -1)
      {
        down += s_alphatuda_volume[lmem];
      }
    }
  }
  __syncthreads();

  if (tidx < ne)
  {
    s_volume[tidx] = volume;
  }
  __syncthreads();

  LevelsetValueType sum_nb_volume = 0.0;
  if (tidx < nv)
  {
#pragma unroll
    for (int i = 0; i < 32; i++)
    {
      short lmem = l_mem[i];
      if (lmem > -1)
      {
        sum_nb_volume += s_volume[lmem / 4];
      }
    }
  }
  __syncthreads();

  if (tidx < ne)
  {
    s_grad_phi[tidx * 3 + 0] = nablaPhi[0] * volume;
    s_grad_phi[tidx * 3 + 1] = nablaPhi[1] * volume;
    s_grad_phi[tidx * 3 + 2] = nablaPhi[2] * volume;
  }
  __syncthreads();

  LevelsetValueType node_nabla_phi_up[3] = {0.0f, 0.0f, 0.0f};
  if (tidx < nv)
  {
#pragma unroll
    for (int i = 0; i < 32; i++)
    {
      short lmem = l_mem[i];
      if (lmem > -1)
      {
        node_nabla_phi_up[0] += s_grad_phi[lmem / 4 * 3 + 0];
        node_nabla_phi_up[1] += s_grad_phi[lmem / 4 * 3 + 1];
        node_nabla_phi_up[2] += s_grad_phi[lmem / 4 * 3 + 2];
      }
    }
  }
  __syncthreads();

  if (tidx < ne)
  {

    for (int i = 0; i < 4; i++)
    {
      s_curv_up[tidx * 4 + i] = volume * (DOT_PRODUCT(nablaN[i], nablaN[i]) / abs_nabla_phi * eleT[i] +
          DOT_PRODUCT(nablaN[i], nablaN[(i + 1) % 4]) / abs_nabla_phi * eleT[(i + 1) % 4] +
          DOT_PRODUCT(nablaN[i], nablaN[(i + 2) % 4]) / abs_nabla_phi * eleT[(i + 2) % 4] +
          DOT_PRODUCT(nablaN[i], nablaN[(i + 3) % 4]) / abs_nabla_phi * eleT[(i + 3) % 4]);
    }
  }
  __syncthreads();

  LevelsetValueType curv_up = 0.0f;
  if (tidx < nv)
  {
#pragma unroll
    for (int i = 0; i < 32; i++)
    {
      short lmem = l_mem[i];
      if (lmem > -1)
      {
        curv_up += s_curv_up[lmem];
      }
    }
    if (fabs(down) > 1e-8)
    {
      LevelsetValueType eikonal = up / down;
      LevelsetValueType node_eikonal = LENGTH(node_nabla_phi_up) / sum_nb_volume;
      vertT_out[vert_start + tidx] = oldT - timestep * eikonal;
    }
    else
    {
      vertT_out[vert_start + tidx] = oldT;
    }
  }
}

__global__ void kernel_compute_vert_ipermute(int nn, int* vert_permute, int* vert_ipermute)
{
  int bidx = blockIdx.x;
  int tidx = bidx * blockDim.x + threadIdx.x;
  for (int vidx = tidx; vidx < nn; vidx += blockDim.x * gridDim.x)
  {
    vert_ipermute[vert_permute[vidx]] = vidx;
  }
}

__global__ void kernel_ele_and_vert(int full_num_ele, int ne, int* ele, int* ele_after_permute, int* ele_permute, int nn, LevelsetValueType* vert, LevelsetValueType* vert_after_permute, LevelsetValueType* vertT, LevelsetValueType* vertT_after_permute, int* vert_permute, int* vert_ipermute)
{
  int bidx = blockIdx.x;
  int tidx = bidx * blockDim.x + threadIdx.x;
  for (int vidx = tidx; vidx < nn; vidx += blockDim.x * gridDim.x)
  {
    int old_vidx = vert_permute[vidx];
    for (int i = 0; i < 3; i++)
    {
      vert_after_permute[i * nn + vidx] = vert[i * nn + old_vidx];
      vertT_after_permute[vidx] = vertT[old_vidx];
    }
  }

  for (int eidx = tidx; eidx < full_num_ele; eidx += blockDim.x * gridDim.x)
  {
    int old_eidx = ele_permute[eidx];
    for (int i = 0; i < 4; i++)
    {
      int old_vidx = ele[i * ne + old_eidx];
      int new_vidx = vert_ipermute[old_vidx];
      ele_after_permute[i * full_num_ele + eidx] = new_vidx;
    }
  }
}

__global__ void kernel_compute_local_coords(int full_num_ele, int nn, int* ele, int* ele_offsets, LevelsetValueType* vert, LevelsetValueType* ele_local_coords,
    LevelsetValueType* cadv_global, LevelsetValueType* cadv_local)
{
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int eidx = tidx; eidx < full_num_ele; eidx += blockDim.x * gridDim.x)
  {
    int ele0 = ele[0 * full_num_ele + eidx];
    int ele1 = ele[1 * full_num_ele + eidx];
    int ele2 = ele[2 * full_num_ele + eidx];
    int ele3 = ele[3 * full_num_ele + eidx];

    LevelsetValueType x0 = vert[0 * nn + ele0];
    LevelsetValueType y0 = vert[1 * nn + ele0];
    LevelsetValueType z0 = vert[2 * nn + ele0];

    LevelsetValueType x1 = vert[0 * nn + ele1];
    LevelsetValueType y1 = vert[1 * nn + ele1];
    LevelsetValueType z1 = vert[2 * nn + ele1];

    LevelsetValueType x2 = vert[0 * nn + ele2];
    LevelsetValueType y2 = vert[1 * nn + ele2];
    LevelsetValueType z2 = vert[2 * nn + ele2];

    LevelsetValueType x3 = vert[0 * nn + ele3];
    LevelsetValueType y3 = vert[1 * nn + ele3];
    LevelsetValueType z3 = vert[2 * nn + ele3];

    LevelsetValueType AB[3] = {x1 - x0, y1 - y0, z1 - z0};
    LevelsetValueType AC[3] = {x2 - x0, y2 - y0, z2 - z0};
    LevelsetValueType AD[3] = {x3 - x0, y3 - y0, z3 - z0};
    LevelsetValueType lenAB = sqrt(AB[0] * AB[0] + AB[1] * AB[1] + AB[2] * AB[2]);

    LevelsetValueType planeN[3];
    CROSS_PRODUCT(AB, AC, planeN);
    LevelsetValueType lenN = sqrt(planeN[0] * planeN[0] + planeN[1] * planeN[1] + planeN[2] * planeN[2]);
    LevelsetValueType Z[3] = {planeN[0] / lenN, planeN[1] / lenN, planeN[2] / lenN};
    LevelsetValueType X[3] = {AB[0] / lenAB, AB[1] / lenAB, AB[2] / lenAB};
    LevelsetValueType Y[3];
    CROSS_PRODUCT(Z, X, Y);

    ele_local_coords[0 * full_num_ele + eidx] = lenAB;
    ele_local_coords[1 * full_num_ele + eidx] = DOT_PRODUCT(AC, X);
    ele_local_coords[2 * full_num_ele + eidx] = DOT_PRODUCT(AC, Y);
    ele_local_coords[3 * full_num_ele + eidx] = DOT_PRODUCT(AD, X);
    ele_local_coords[4 * full_num_ele + eidx] = DOT_PRODUCT(AD, Y);
    ele_local_coords[5 * full_num_ele + eidx] = DOT_PRODUCT(AD, Z);

    LevelsetValueType cadv_old[3] = {cadv_global[0 * full_num_ele + eidx], cadv_global[1 * full_num_ele + eidx], cadv_global[2 * full_num_ele + eidx]};
    cadv_local[0 * full_num_ele + eidx] = DOT_PRODUCT(cadv_old, X);
    cadv_local[1 * full_num_ele + eidx] = DOT_PRODUCT(cadv_old, Y);
    cadv_local[2 * full_num_ele + eidx] = DOT_PRODUCT(cadv_old, Z);
  }
}

__global__ void kernel_fill_ele_label(int ne, int* ele_permute, int* ele_offsets, int* npart, int* ele, int* ele_label)
{
  int bidx = blockIdx.x;
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;

  for (int eidx = tidx; eidx < ne; eidx += blockDim.x * gridDim.x)
  {
    int part0 = npart[ele[0 * ne + eidx]];
    int part1 = npart[ele[1 * ne + eidx]];
    int part2 = npart[ele[2 * ne + eidx]];
    int part3 = npart[ele[3 * ne + eidx]];
    int start = ele_offsets[eidx];
    int end = ele_offsets[eidx + 1];
    int n = end - start;
    for (int j = 0; j < n; j++) ele_permute[start + j] = eidx;
    ele_label[start] = part0;
    int i = 1;
    if (part1 != part0)
    {
      ele_label[start + i] = part1;
      i++;
    }
    if (part2 != part0 && part2 != part1)
    {
      ele_label[start + i] = part2;
      i++;
    }
    if (part3 != part0 && part3 != part1 && part3 != part2)
    {
      ele_label[start + i] = part3;
      i++;
    }

    if (i != n)
    {
      printf("bidx = %d, tidx = %d, i!=n!!\n", bidx, tidx);
    }
  }
}

__global__ void kernel_compute_ele_npart(int ne, int* npart, int* ele, int* ele_label)
{
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;

  for (int eidx = tidx; eidx < ne; eidx += blockDim.x * gridDim.x)
  {
    int part0 = npart[ele[0 * ne + eidx]];
    int part1 = npart[ele[1 * ne + eidx]];
    int part2 = npart[ele[2 * ne + eidx]];
    int part3 = npart[ele[3 * ne + eidx]];

    int n = 1;

    if (part1 != part0) n++;
    if (part2 != part0 && part2 != part1) n++;
    if (part3 != part0 && part3 != part1 && part3 != part2) n++;

    ele_label[eidx] = n;
  }
}

__global__ void permuteInitialAdjacencyKernel(int size, int *adjIndexesIn, int *adjacencyIn, int *permutedAdjIndexesIn, int *permutedAdjacencyIn, int *ipermutation, int *fineAggregate, int* neighbor_part)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    int oldBegin = adjIndexesIn[ipermutation[idx]];
    int oldEnd = adjIndexesIn[ipermutation[idx] + 1];
    int runSize = oldEnd - oldBegin;
    int newBegin = permutedAdjIndexesIn[idx];

    // Transfer old adjacency into new, while changing node id's with partition id's
    for (int i = 0; i < runSize; i++)
    {
      permutedAdjacencyIn[newBegin + i] = fineAggregate[ adjacencyIn[oldBegin + i] ];
    }
  }
}

__global__ void getInducedGraphNeighborCountsKernel(int size, int *aggregateIdx, int *adjIndexesOut, int *permutedAdjIndexes, int *permutedAdjacencyIn)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    int Begin = permutedAdjIndexes[ aggregateIdx[idx] ];
    int End = permutedAdjIndexes[ aggregateIdx[idx + 1] ];

    // Sort each section of the adjacency:
    for (int i = Begin; i < End - 1; i++)
    {
      for (int ii = i + 1; ii < End; ii++)
      {
        if (permutedAdjacencyIn[i] < permutedAdjacencyIn[ii])
        {
          int temp = permutedAdjacencyIn[i];
          permutedAdjacencyIn[i] = permutedAdjacencyIn[ii];
          permutedAdjacencyIn[ii] = temp;
        }
      }
    }

    // Scan through the sorted adjacency to get the condensed adjacency:
    int neighborCount = 1;
    if (permutedAdjacencyIn[Begin] == idx)
      neighborCount = 0;

    for (int i = Begin + 1; i < End; i++)
    {
      if (permutedAdjacencyIn[i] != permutedAdjacencyIn[i - 1] && permutedAdjacencyIn[i] != idx)
      {
        permutedAdjacencyIn[neighborCount + Begin] = permutedAdjacencyIn[i];
        neighborCount++;
      }
    }

    // Store the size
    adjIndexesOut[idx] = neighborCount;
  }
}

__global__ void fillCondensedAdjacencyKernel(int size, int *aggregateIdx, int *adjIndexesOut, int *adjacencyOut, int *permutedAdjIndexesIn, int *permutedAdjacencyIn)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    int oldBegin = permutedAdjIndexesIn[ aggregateIdx[idx] ];
    int newBegin = adjIndexesOut[idx];
    int runSize = adjIndexesOut[idx + 1] - newBegin;

    // Copy adjacency over
    for (int i = 0; i < runSize; i++)
    {
      adjacencyOut[newBegin + i] = permutedAdjacencyIn[oldBegin + i];
    }
  }
}

__global__ void mapAdjacencyToBlockKernel(int size, int *adjIndexes, int *adjacency, int *adjacencyBlockLabel, int *blockMappedAdjacency, int *fineAggregate)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    int begin = adjIndexes[idx];
    int end = adjIndexes[idx + 1];
    int thisBlock = fineAggregate[idx];

    // Fill block labeled adjacency and block mapped adjacency vectors
    for (int i = begin; i < end; i++)
    {
      int neighbor = fineAggregate[adjacency[i]];

      if (thisBlock == neighbor)
      {
        adjacencyBlockLabel[i] = -1;
        blockMappedAdjacency[i] = -1;
      }
      else
      {
        adjacencyBlockLabel[i] = thisBlock;
        blockMappedAdjacency[i] = neighbor;
      }
    }
  }
}

__global__ void findPartIndicesNegStartKernel(int size, int *array, int *partIndices)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (idx < size)
  {
    int value = array[idx];
    int nextValue = array[idx + 1];
    if (value != nextValue)
      partIndices[value + 1] = idx;
  }
}

__global__ void kernel_compute_vertT_before_permute(int nn, int* vert_permute, LevelsetValueType* vertT_after_permute, LevelsetValueType* vertT_before_permute)
{
  int bidx = blockIdx.x;
  int tidx = bidx * blockDim.x + threadIdx.x;
  for (int vidx = tidx; vidx < nn; vidx += blockDim.x * gridDim.x)
  {
    int old_vidx = vert_permute[vidx];
    vertT_before_permute[old_vidx] = vertT_after_permute[vidx];
  }
}

#endif  /* MESHFIM_KERNELS_H */

