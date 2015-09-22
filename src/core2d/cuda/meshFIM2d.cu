#include <meshFIM2d.h>
#include <Vec.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <meshFIM2d_kernels.h>
#include <mycutil.h>
#include <cusp/detail/format_utils.h>
#include <cusp/print.h>
#include <thrust/functional.h>

extern "C"
{
#include <metis.h>
}

void meshFIM2d::writeFLD()
{
  int nv = m_meshPtr->vertices.size();
  int nt = m_meshPtr->faces.size();
  FILE* matfile;
  FILE* fldfile;
  matfile = fopen("result.txt", "w+");
  fldfile = fopen("result.fld", "w+");
  fprintf(fldfile, "SCI\nASC\n2\n{@1 {GenericField<TriSurfMesh<TriLinearLgn<Point>>,ConstantBasis<float>,vector<float>> 3 {Field 3 {PropertyManager 2 1 \"name\" @2 {Property<string> 2 0 \"TorsoVolumeMesh\" }\n}\n}\n{@3 {TriSurfMesh<TriLinearLgn<Point>> 4 {Mesh 2 {PropertyManager 2 0 }\n}\n");
  fprintf(fldfile, "{STLVector 2 %d ", nv);
  for (int i = 0; i < nv; i++)
  {
    fprintf(fldfile, "{%.12f %.12f %.12f}", m_meshPtr->vertices[i][0], m_meshPtr->vertices[i][1], m_meshPtr->vertices[i][2]);
  }
  fprintf(fldfile, "}\n{STLIndexVector 1 %d 8 ", nt * 3);
  for (int i = 0; i < nt; i++)
  {
    fprintf(fldfile, "%d %d %d ", m_meshPtr->faces[i][0], m_meshPtr->faces[i][1], m_meshPtr->faces[i][2]);
  }
  fprintf(fldfile, "}\n");
  fprintf(fldfile, "{TriLinearLgn<Point>  1 }\n}\n}{ConstantBasis<float>  1 }\n");
  fprintf(fldfile, "{STLVector 2 %d ", nt);
  for (int i = 0; i < nt; i++)
  {
    fprintf(fldfile, " 0");
  }

  fprintf(fldfile, "}\n}\n}");
  for (int i = 0; i < nv; i++)
  {
    fprintf(matfile, "%.12f\n", m_meshPtr->vertT[i]);
  }
  fclose(matfile);
  fclose(fldfile);
}

void meshFIM2d::writeVTK()
{
  FILE* vtkfile;
  int nv = m_meshPtr->vertices.size();
  int nt = m_meshPtr->faces.size();
  vtkfile = fopen("result.vtk", "w+");
  fprintf(vtkfile, "# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET UNSTRUCTURED_GRID\n");
  fprintf(vtkfile, "POINTS %d float\n", nv);
  for (int i = 0; i < nv; i++)
  {
    fprintf(vtkfile, "%.12f %.12f %.12f\n", m_meshPtr->vertices[i][0], m_meshPtr->vertices[i][1], m_meshPtr->vertices[i][2]);
  }
  fprintf(vtkfile, "CELLS %d %d\n", nt, nt * 4);
  for (int i = 0; i < nt; i++)
  {
    fprintf(vtkfile, "3 %d %d %d\n", m_meshPtr->faces[i][0], m_meshPtr->faces[i][1], m_meshPtr->faces[i][2]);
  }

  fprintf(vtkfile, "CELL_TYPES %d\n", nt);
  for (int i = 0; i < nt; i++)
  {
    fprintf(vtkfile, "5\n");
  }
  fprintf(vtkfile, "POINT_DATA %d\nSCALARS traveltime float 1\nLOOKUP_TABLE default\n", nv);
  for (int i = 0; i < nv; i++)
  {
    fprintf(vtkfile, "%.12f\n", m_meshPtr->vertT[i]);
  }
  fclose(vtkfile);
}

void meshFIM2d::updateT_single_stage_d(LevelsetValueType timestep, int niter, IdxVector_d& narrowband, int num_narrowband)
{
  int nn = m_meshPtr->vertices.size();
  int ne = full_num_ele;
  int nblocks = num_narrowband;
  int nthreads = largest_ele_part;
  thrust::fill(vertT_out.begin(), vertT_out.end(), 0.0);
  int shared_size = sizeof (LevelsetValueType)* 3 * largest_ele_part + sizeof (short) *largest_vert_part*m_largest_num_inside_mem;
  //  int shared_size = sizeof(LevelsetValueType) * 3 * largest_vert_part;
  cudaSafeCall((kernel_updateT_single_stage << <nblocks, nthreads, shared_size >> >(timestep, CAST(narrowband), largest_ele_part, largest_vert_part, full_num_ele,
          CAST(m_ele_after_permute_d), CAST(m_ele_offsets_d), CAST(m_cadv_local_d),
          nn, CAST(m_vert_offsets_d), CAST(m_vert_after_permute_d), CAST(m_vertT_after_permute_d),
          CAST(m_ele_local_coords_d), m_largest_num_inside_mem, CAST(m_mem_locations), CAST(m_mem_location_offsets),
          CAST(vertT_out))));
  //  Vector_h tmp_h = vertT_out;
  //  for(int i=0; i<tmp_h.size(); i++)
  //  {
  //    printf("vertT_out[%d]=%f\n", i, tmp_h[i]);
  //  }

  nthreads = largest_vert_part;
  cudaSafeCall((CopyOutBack_levelset << <nblocks, nthreads >> >(CAST(narrowband),
          CAST(m_vert_offsets_d), CAST(m_vertT_after_permute_d), CAST(vertT_out))));
  //  m_vertT_after_permute_d = vertT_out;
  //  Vector_h tmp_h = m_vertT_after_permute_d;
  //  for(int i=0; i<tmp_h.size(); i++)
  //  {
  //    printf("m_vert_after_permute_d[%d]=%f\n", i, tmp_h[i]);
  //  }
}

//Single stage update

void meshFIM2d::updateT_single_stage(LevelsetValueType timestep, int nside, int niter, vector<int>& narrowband)
{
  vec3 sigma(1.0, 0.0, 1.0);
  LevelsetValueType epsilon = 1.0;
  int nv = m_meshPtr->vertices.size();
  int nt = m_meshPtr->faces.size();
  vector<LevelsetValueType> values(4);
  vector<LevelsetValueType> up(nv, 0.0);
  vector<LevelsetValueType> down(nv, 0.0);
  vector<vec3> node_grad_phi_up(nv, vec3(0.0, 0.0, 0.0));
  vector<LevelsetValueType> node_grad_phi_down(nv, 0.0);
  vector<LevelsetValueType> curv_up(nv, 0.0);


  for (int bandidx = 0; bandidx < narrowband.size(); bandidx++)
  {
    int tidx = narrowband[bandidx];
    for (int j = 0; j < 4; j++)
    {
      values[j] = m_meshPtr->vertT[m_meshPtr->faces[tidx][j]];
    }
    //compute ni normals
    vector<vec3> nodes(4);
    nodes[0] = (vec3) m_meshPtr->vertices[m_meshPtr->faces[tidx][0]];
    nodes[1] = (vec3) m_meshPtr->vertices[m_meshPtr->faces[tidx][1]];
    nodes[2] = (vec3) m_meshPtr->vertices[m_meshPtr->faces[tidx][2]];
    nodes[3] = (vec3) m_meshPtr->vertices[m_meshPtr->faces[tidx][3]];
    vec3 v31 = nodes[1] - nodes[3];
    vec3 v32 = nodes[2] - nodes[3];
    vec3 v30 = nodes[0] - nodes[3];
    vec3 crossproduct = v31 CROSS v32;
    LevelsetValueType dotproduct = crossproduct DOT v30;
    LevelsetValueType volume = fabs(dotproduct) / 6.0;

    //compute inverse of 4 by 4 matrix
    LevelsetValueType a11 = nodes[0][0], a12 = nodes[0][1], a13 = nodes[0][2], a14 = 1.0;
    LevelsetValueType a21 = nodes[1][0], a22 = nodes[1][1], a23 = nodes[1][2], a24 = 1.0;
    LevelsetValueType a31 = nodes[2][0], a32 = nodes[2][1], a33 = nodes[2][2], a34 = 1.0;
    LevelsetValueType a41 = nodes[3][0], a42 = nodes[3][1], a43 = nodes[3][2], a44 = 1.0;

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

    LevelsetValueType b41 = a21 * a33 * a42 + a22 * a31 * a43 + a23 * a32 * a41 - a21 * a32 * a43 - a22 * a33 * a41 - a23 * a31 * a42;
    LevelsetValueType b42 = a11 * a32 * a43 + a12 * a33 * a41 + a13 * a31 * a42 - a11 * a33 * a42 - a12 * a31 * a43 - a13 * a32 * a41;
    LevelsetValueType b43 = a11 * a23 * a42 + a12 * a21 * a43 + a13 * a22 * a41 - a11 * a22 * a43 - a12 * a23 * a41 - a13 * a21 * a42;
    LevelsetValueType b44 = a11 * a22 * a33 + a12 * a23 * a31 + a13 * a21 * a32 - a11 * a23 * a32 - a12 * a21 * a33 - a13 * a22 * a31;

    vector<vec4> Arows(4);
    Arows[0] = vec4(b11 / det, b12 / det, b13 / det, b14 / det);
    Arows[1] = vec4(b21 / det, b22 / det, b23 / det, b24 / det);
    Arows[2] = vec4(b31 / det, b32 / det, b33 / det, b34 / det);
    Arows[3] = vec4(b41 / det, b42 / det, b43 / det, b44 / det);

    vector<vec3> nablaN(4);
    for (int i = 0; i < 4; i++)
    {
      vec4 RHS(0.0, 0.0, 0.0, 0.0);
      RHS[i] = 1.0;
      nablaN[i][0] = Arows[0] DOT RHS;
      nablaN[i][1] = Arows[1] DOT RHS;
      nablaN[i][2] = Arows[2] DOT RHS;
    }

    //compuate grad of Phi
    vec3 nablaPhi(0.0, 0.0, 0.0);
    for (int i = 0; i < 4; i++)
    {
      nablaPhi[0] += nablaN[i][0] * values[i];
      nablaPhi[1] += nablaN[i][1] * values[i];
      nablaPhi[2] += nablaN[i][2] * values[i];
    }
    LevelsetValueType abs_nabla_phi = len(nablaPhi);

    //compute K and Kplus and Kminus
    vector<LevelsetValueType> Kplus(4);
    vector<LevelsetValueType> Kminus(4);
    vector<LevelsetValueType> K(4);
    LevelsetValueType Hintegral = 0.0;
    LevelsetValueType beta = 0;
    for (int i = 0; i < 4; i++)
    {
      K[i] = volume * (sigma DOT nablaN[i]); // for H(\nabla u) = sigma DOT \nabla u
      //      K[i] = volume * (nablaPhi DOT nablaN[i]) / len(nablaPhi); // for F(x) = 1
      //      K[i] = -volume* (nablaPhi DOT nablaN[i]) / len(nablaPhi); // for F(x) = -1
      Hintegral += K[i] * values[i];
      Kplus[i] = fmax(K[i], 0.0f);
      Kminus[i] = fmin(K[i], 0.0f);
      beta += Kminus[i];
    }
    beta = 1.0 / beta;

    if (fabs(Hintegral) > 1e-16)
    {
      vector<LevelsetValueType> delta(4);
      for (int i = 0; i < 4; i++)
      {
        delta[i] = Kplus[i] * beta * (Kminus[0] * (values[i] - values[0]) + Kminus[1] * (values[i] - values[1]) + Kminus[2] * (values[i] - values[2]) + Kminus[3] * (values[i] - values[3]));
      }

      vector<LevelsetValueType> alpha(4);
      for (int i = 0; i < 4; i++)
      {
        alpha[i] = delta[i] / Hintegral;
      }

      LevelsetValueType theta = 0;
      for (int i = 0; i < 4; i++)
      {
        theta += fmax(0.0f, alpha[i]);
      }

      vector<LevelsetValueType> alphatuda(4);
      for (int i = 0; i < 4; i++)
      {
        alphatuda[i] = fmax(alpha[i], 0.0f) / theta;
      }

      for (int i = 0; i < 4; i++)
      {
        up[m_meshPtr->faces[tidx][i]] += alphatuda[i] * Hintegral;
        down[m_meshPtr->faces[tidx][i]] += alphatuda[i] * volume;
        node_grad_phi_up[m_meshPtr->faces[tidx][i]] += volume* nablaPhi;
        node_grad_phi_down[m_meshPtr->faces[tidx][i]] += volume;
        curv_up[m_meshPtr->faces[tidx][i]] += volume * ((nablaN[i] DOT nablaN[i]) / abs_nabla_phi * values[i] +
            (nablaN[i] DOT nablaN[(i + 1) % 4]) / abs_nabla_phi * values[(i + 1) % 4] +
            (nablaN[i] DOT nablaN[(i + 2) % 4]) / abs_nabla_phi * values[(i + 2) % 4] +
            (nablaN[i] DOT nablaN[(i + 3) % 4]) / abs_nabla_phi * values[(i + 3) % 4]);
      }
    }
  }

  for (int vidx = 0; vidx < nv; vidx++)
  {
    LevelsetValueType eikonal = up[vidx] / down[vidx];
    LevelsetValueType curvature = curv_up[vidx] / node_grad_phi_down[vidx];
    LevelsetValueType node_eikonal = len(node_grad_phi_up[vidx]) / node_grad_phi_down[vidx];
    //    if(niter == 49) printf("%d %.15f %.15f %.15f\n", vidx, eikonal, up[vidx], down[vidx]);
    if (fabs(down[vidx]) > 1e-16)
    {
      //      m_meshPtr->vertT[vidx] -= epsilon * node_eikonal * curvature * timestep;
      m_meshPtr->vertT[vidx] -= eikonal * timestep;
      //      m_meshPtr->vertT[vidx] -= epsilon * node_eikonal * timestep;
    }
  }
}

void meshFIM2d::GraphPartition_Square(int squareLength, int squareWidth, int blockLength, int blockWidth)
{
  int nn = m_meshPtr->vertices.size();
  int numBlockLength = ceil((LevelsetValueType) squareLength / blockLength);
  int numBlockWidth = ceil((LevelsetValueType) squareWidth / blockWidth);
  int numBlock = numBlockLength * numBlockWidth;
  npart_h = IdxVector_h(nn);
  nparts = numBlock;

  int edgeCount = 0;
  for (int vIt = 0; vIt < nn; vIt++)
  {
    edgeCount += m_meshPtr->neighbors[vIt].size();
  }

  m_largest_num_inside_mem = 0;
  //for(int bidx = 0; bidx < nparts; bidx++)
  for (int i = 0; i < nn; i++)
  {
    if (m_meshPtr->adjacentfaces[i].size() > m_largest_num_inside_mem)
      m_largest_num_inside_mem = m_meshPtr->adjacentfaces[i].size();
  }
  printf("m_largest_num_inside_mem = %d\n", m_largest_num_inside_mem);

  //Allocating storage for array values of adjacency
  int* xadj = new int[nn + 1];
  int* adjncy = new int[edgeCount];

  // filling the arrays:
  xadj[0] = 0;
  int idx = 0;
  IdxVector_h neighbor_sizes(nn);
  // Populating the arrays:
  for (int i = 1; i < nn + 1; i++)
  {
    neighbor_sizes[i - 1] = m_meshPtr->neighbors[i - 1].size();
    xadj[i] = xadj[i - 1] + m_meshPtr->neighbors[i - 1].size();
    for (int j = 0; j < m_meshPtr->neighbors[i - 1].size(); j++)
    {
      adjncy[idx++] = m_meshPtr->neighbors[i - 1][j];
    }
  }

  m_neighbor_sizes_d = neighbor_sizes;

  for (int i = 0; i < squareWidth; i++)
    for (int j = 0; j < squareLength; j++)
    {
      int index = i * squareLength + j;
      int i2 = i;
      int j2 = j;
      npart_h[index] = (i2 / blockWidth) * numBlockLength + (j2 / blockLength);
    }

  //  cusp::print(npart_h);
  m_xadj_d = IdxVector_d(&xadj[0], &xadj[nn + 1]);
  m_adjncy_d = IdxVector_d(&adjncy[0], &adjncy[edgeCount]);

  IdxVector_h part_sizes(nparts, 0);
  for (int i = 0; i < nn; i++)
  {
    part_sizes[npart_h[i]]++;
  }
  int min_part_size = thrust::reduce(part_sizes.begin(), part_sizes.end(), 100000000, thrust::minimum<int>());
  largest_vert_part = thrust::reduce(part_sizes.begin(), part_sizes.end(), -1, thrust::maximum<int>());
  printf("Largest vertex partition size is: %d\n", largest_vert_part);
  if (min_part_size == 0) printf("Min partition size is 0!!\n");
  delete [] xadj;
  delete [] adjncy;
}

void meshFIM2d::Partition_METIS(int metissize)
{
  int options[10], pnumflag = 0, wgtflag = 0;
  options[0] = 0;
  int edgecut;
  int nn = m_meshPtr->vertices.size();
  npart_h = IdxVector_h(nn);
  nparts = nn / metissize;
  if (nparts < 2)
    nparts = 2;

  // Counting up edges for adjacency:
  int edgeCount = 0;
  for (int vIt = 0; vIt < nn; vIt++)
  {
    edgeCount += m_meshPtr->neighbors[vIt].size();
  }

  m_largest_num_inside_mem = 0;
  //for(int bidx = 0; bidx < nparts; bidx++)
  for (int i = 0; i < nn; i++)
  {
    if (m_meshPtr->adjacentfaces[i].size() > m_largest_num_inside_mem)
      m_largest_num_inside_mem = m_meshPtr->adjacentfaces[i].size();
  }

  printf("m_largest_num_inside_mem = %d\n", m_largest_num_inside_mem);

  //Allocating storage for array values of adjacency
  int* xadj = new int[nn + 1];
  int* adjncy = new int[edgeCount];

  // filling the arrays:
  xadj[0] = 0;
  int idx = 0;
  IdxVector_h neighbor_sizes(nn);
  // Populating the arrays:
  for (int i = 1; i < nn + 1; i++)
  {
    neighbor_sizes[i - 1] = m_meshPtr->neighbors[i - 1].size();
    xadj[i] = xadj[i - 1] + m_meshPtr->neighbors[i - 1].size();
    for (int j = 0; j < m_meshPtr->neighbors[i - 1].size(); j++)
    {
      adjncy[idx++] = m_meshPtr->neighbors[i - 1][j];
    }
  }

  m_neighbor_sizes_d = neighbor_sizes;
  int* npart_h_ptr = thrust::raw_pointer_cast(&npart_h[0]);

  METIS_PartGraphKway(&nn, xadj, adjncy, NULL, NULL, &wgtflag, &pnumflag, &nparts, options, &edgecut, npart_h_ptr);

  m_xadj_d = IdxVector_d(&xadj[0], &xadj[nn + 1]);
  m_adjncy_d = IdxVector_d(&adjncy[0], &adjncy[edgeCount]);

  IdxVector_h part_sizes(nparts, 0);
  for (int i = 0; i < nn; i++)
  {
    part_sizes[npart_h[i]]++;
  }
  int min_part_size = thrust::reduce(part_sizes.begin(), part_sizes.end(), 100000000, thrust::minimum<int>());
  largest_vert_part = thrust::reduce(part_sizes.begin(), part_sizes.end(), -1, thrust::maximum<int>());
  printf("Largest vertex partition size is: %d\n", largest_vert_part);
  if (min_part_size == 0) printf("Min partition size is 0!!\n");
  delete [] xadj;
  delete [] adjncy;
}

void meshFIM2d::InitPatches()
{
  int ne = m_meshPtr->faces.size();
  int nn = m_meshPtr->vertices.size();
  ele_d = IdxVector_d(3 * ne);
  ele_h = IdxVector_h(3 * ne);
  vert_d = Vector_d(3 * nn);
  m_vert_after_permute_d = Vector_d(3 * nn);
  Vector_h vert_h(3 * nn);
  //  thrust::fill(ele_label_d.begin(), ele_label_d.end(), 0);
  for (int eidx = 0; eidx < ne; eidx++)
  {
    for (int i = 0; i < 3; i++)
      ele_h[i * ne + eidx] = m_meshPtr->faces[eidx][i]; //interleaved storage
  }
  for (int vidx = 0; vidx < nn; vidx++)
  {
    for (int i = 0; i < 3; i++)
      vert_h[i * nn + vidx] = m_meshPtr->vertices[vidx][i]; //interleaved storage
  }
  ele_d = ele_h;
  vert_d = vert_h;
  m_npart_d = IdxVector_d(npart_h.begin(), npart_h.end());
  m_part_label_d = IdxVector_d(m_npart_d.begin(), m_npart_d.end());
  int nthreads = 256;
  int nblocks = min((int) ceil((LevelsetValueType) ne / nthreads), 65535);
  IdxVector_d ele_label_d(ne);
  cudaSafeCall((kernel_compute_ele_npart << <nblocks, nthreads >> >(ne, thrust::raw_pointer_cast(&m_npart_d[0]), thrust::raw_pointer_cast(&ele_d[0]), thrust::raw_pointer_cast(&ele_label_d[0]))));

  //  cudaThreadSynchronize();
  //  printf("line 437\n");
  //    IdxVector_h tmp_h;
  //    printf("size of ele_label_d is %d\n", ele_label_d.size());
  //    tmp_h = ele_label_d;
  //    for(int i=0; i<tmp_h.size(); i++)
  //    {
  //      printf("ele_label_d[%d] = %d\n",i, tmp_h[i]);
  //    }
  full_num_ele = thrust::reduce(ele_label_d.begin(), ele_label_d.end());
  //  cudaThreadSynchronize();
  //  printf("line 446\n");
  printf("full_num_ele = %d\n", full_num_ele);
  IdxVector_d ele_offsets_d(ne + 1);
  ele_offsets_d[0] = 0;
  thrust::inclusive_scan(ele_label_d.begin(), ele_label_d.end(), ele_offsets_d.begin() + 1);
  //  cudaThreadSynchronize();
  //  printf("line 450\n");
  IdxVector_d ele_full_label(full_num_ele);
  ele_permute = IdxVector_d(full_num_ele);
  cudaSafeCall((kernel_fill_ele_label << <nblocks, nthreads >> >(ne, thrust::raw_pointer_cast(&ele_permute[0]), thrust::raw_pointer_cast(&ele_offsets_d[0]),
          thrust::raw_pointer_cast(&m_npart_d[0]), thrust::raw_pointer_cast(&ele_d[0]),
          thrust::raw_pointer_cast(&ele_full_label[0]))));

  //  //  printf("line 453\n");
  //  //  tmp = ele_full_label;
  //  //  for(int i = 0; i < full_num_ele; i++)
  //  //  {
  //  //    printf("ele_full_label[%d] = %d\n", i, tmp[i]);
  //  //  }
  thrust::sort_by_key(ele_full_label.begin(), ele_full_label.end(), ele_permute.begin());
  //  cusp::print(ele_permute);
  m_ele_offsets_d = IdxVector_d(nparts + 1);
  IdxVector_d ones(full_num_ele, 1);
  IdxVector_d tmp(full_num_ele);
  IdxVector_d reduce_output(full_num_ele);
  thrust::reduce_by_key(ele_full_label.begin(), ele_full_label.end(), ones.begin(), tmp.begin(), reduce_output.begin());
  //  thrust::reduce_by_key(tmp_h.begin(), tmp_h.end(), ones.begin(), tmp.begin(), reduce_output_h.begin());
  //  IdxVector_d reduce_output = IdxVector_d(reduce_output_h.begin(), reduce_output_h.begin() + nparts);
  largest_ele_part = thrust::reduce(reduce_output.begin(), reduce_output.begin() + nparts, -1, thrust::maximum<int>());
  printf("Largest element partition size is: %d\n", largest_ele_part);
  if (largest_ele_part > 1024)
  {
    printf("Error: largest_ele_part > 1024 !!\n");
    exit(0);
  }
  m_ele_offsets_d[0] = 0;
  thrust::inclusive_scan(reduce_output.begin(), reduce_output.begin() + nparts, m_ele_offsets_d.begin() + 1);
}

void meshFIM2d::InitPatches2()
{
  int ne = m_meshPtr->faces.size();
  int nn = m_meshPtr->vertices.size();
  //compute invert vert permute
  IdxVector_d vert_permute(nn, 0);
  IdxVector_d vert_ipermute(nn, 0);
  //  thrust::sequence(vert_permute.begin(), vert_permute.end(), 0, 1);
  int nthreads = 256;
  int nblocks = min((int) ceil((LevelsetValueType) nn / nthreads), 65535);
  cudaSafeCall((kernel_fill_sequence << <nblocks, nthreads >> >(nn, CAST(vert_permute))));
  //  IdxVector_h tmp_h = vert_permute;
  //  for(int i = 0; i < tmp_h.size(); i++)
  //  {
  //    printf("vert_permute[%d] = %d\n", i, tmp_h[i]);
  //  }
  thrust::sort_by_key(m_part_label_d.begin(), m_part_label_d.end(), vert_permute.begin());
  nblocks = min((int) ceil((LevelsetValueType) nn / nthreads), 65535);



  cudaSafeCall((kernel_compute_vert_ipermute << <nblocks, nthreads >> >(nn, thrust::raw_pointer_cast(&vert_permute[0]), thrust::raw_pointer_cast(&vert_ipermute[0]))));
  m_vert_permute_d = IdxVector_d(vert_permute);
  m_vert_offsets_d = IdxVector_d(nparts + 1);
  cusp::detail::indices_to_offsets(m_part_label_d, m_vert_offsets_d);

  //  cudaThreadSynchronize();
  //  printf("line 491\n");
  //  tmp_h = m_vert_offsets_d;
  //  for(int i= 0; i < nparts + 1; i++)
  //  {
  //    printf("m_vert_offsets_d[%d] = %d\n", i, tmp_h[i]);
  //  }

  //permute the vert and ele values and Rinscribe
  m_ele_after_permute_d = IdxVector_d(3 * full_num_ele);
  m_vertT_after_permute_d = Vector_d(nn);
  nblocks = min((int) ceil((LevelsetValueType) full_num_ele / nthreads), 65535);
  cudaSafeCall((kernel_ele_and_vert << <nblocks, nthreads >> >(full_num_ele, ne,
          thrust::raw_pointer_cast(&ele_d[0]), thrust::raw_pointer_cast(&m_ele_after_permute_d[0]), thrust::raw_pointer_cast(&ele_permute[0]),
          nn, thrust::raw_pointer_cast(&vert_d[0]), thrust::raw_pointer_cast(&m_vert_after_permute_d[0]),
          thrust::raw_pointer_cast(&m_vertT_d[0]), thrust::raw_pointer_cast(&m_vertT_after_permute_d[0]),
          CAST(m_Rinscribe_before_permute_d), CAST(m_Rinscribe_d),
          CAST(vert_permute),
          thrust::raw_pointer_cast(&vert_ipermute[0]))));

  //  Vector_h LevelsetValueType_tmp_h = m_vert_after_permute_d;
  //  for(int i = 0; i < 3*nn ; i++)
  //  {
  //    printf("m_vert_after_permute_d[%d] = %f\n", i, LevelsetValueType_tmp_h[i]);
  //  }
  //
  //  tmp_h = vert_ipermute;
  //  for(int i = 0; i < nn; i++)
  //  {
  //    printf("vert_ipermute[%d] = %d\n", i, tmp_h[i]);
  //  }
  //
  //compute the local coords for each element
  m_ele_local_coords_d = Vector_d(3 * full_num_ele);
  m_cadv_local_d = Vector_d(2 * full_num_ele);
  nthreads = 256;
  nblocks = min((int) ceil((float) full_num_ele / nthreads), 65535);
  cudaSafeCall((kernel_compute_local_coords << <nblocks, nthreads >> >(full_num_ele, nn,
          thrust::raw_pointer_cast(&m_ele_after_permute_d[0]), thrust::raw_pointer_cast(&m_ele_offsets_d[0]),
          thrust::raw_pointer_cast(&m_vert_after_permute_d[0]),
          thrust::raw_pointer_cast(&m_ele_local_coords_d[0]),
          CAST(m_cadv_global_d),
          CAST(m_cadv_local_d))));
  //  Vector_h LevelsetValueType_tmp_h = m_ele_local_coords_d;
  //  for(int i = 0; i < 6 * full_num_ele; i++)
  //  {
  //    printf("m_ele_local_coords_d[%d] = %f\n", i, LevelsetValueType_tmp_h[i]);
  //  }

  //Generate redution list

  m_mem_locations = IdxVector_d(3 * full_num_ele);
  IdxVector_d tmp2 = m_ele_after_permute_d;
  thrust::sequence(m_mem_locations.begin(), m_mem_locations.end(), 0);
  thrust::sort_by_key(tmp2.begin(), tmp2.end(), m_mem_locations.begin());
  m_mem_location_offsets = IdxVector_d(nn + 1);
  cusp::detail::indices_to_offsets(tmp2, m_mem_location_offsets);

  //  tmp_h = tmp2;
  //  for(int i = 0; i < 4 * full_num_ele; i++)
  //  {
  //    printf("tmp2[%d] = %d\n", i, tmp_h[i]);
  //  }
  //
  //  tmp_h = m_mem_location_offsets;
  //  for(int i = 0; i < nn + 1; i++)
  //  {
  //    printf("m_mem_location_offsets[%d] = %d\n", i, tmp_h[i]);
  //  }
}

void meshFIM2d::GenerateBlockNeighbors()
{

  //Generate block neighbors
  // Declaring temporary vectors:
  IdxVector_d adjacencyBlockLabel, blockMappedAdjacency;
  adjacencyBlockLabel = IdxVector_d(m_adjncy_d.size(), 0);
  blockMappedAdjacency = IdxVector_d(m_adjncy_d.size(), 0);

  mapAdjacencyToBlock(m_xadj_d, m_adjncy_d, adjacencyBlockLabel, blockMappedAdjacency, m_npart_d);

  // Zip up the block label and block mapped vectors and sort:
  thrust::sort(thrust::make_zip_iterator(thrust::make_tuple(adjacencyBlockLabel.begin(), blockMappedAdjacency.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(adjacencyBlockLabel.end(), blockMappedAdjacency.end())));

  // Remove Duplicates and resize:
  int newSize = thrust::unique(thrust::make_zip_iterator(thrust::make_tuple(adjacencyBlockLabel.begin(), blockMappedAdjacency.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(adjacencyBlockLabel.end(), blockMappedAdjacency.end()))) -
    thrust::make_zip_iterator(thrust::make_tuple(adjacencyBlockLabel.begin(), blockMappedAdjacency.begin()));

  adjacencyBlockLabel.resize(newSize);
  blockMappedAdjacency.resize(newSize);
  getPartIndicesNegStart(adjacencyBlockLabel, m_block_xadj_d);
  m_block_adjncy_d.resize(blockMappedAdjacency.size() - 1);
  thrust::copy(blockMappedAdjacency.begin() + 1, blockMappedAdjacency.end(), m_block_adjncy_d.begin());

  //  cudaThreadSynchronize();
  //  tmp_h = m_block_xadj_d;
  //  for(int i = 0; i < m_block_xadj_d.size(); i++)
  //  {
  //    printf("m_block_xadj_d[%d] = %d\n", i, tmp_h[i]);
  //  }
  //
  //  tmp_h = m_block_adjncy_d;
  //  for(int i = 0; i < m_block_adjncy_d.size(); i++)
  //  {
  //    printf("m_block_adjncy_d[%d] = %d\n", i, tmp_h[i]);
  //  }
}

void meshFIM2d::compute_deltaT(int num_narrowband)
{
  int nn = m_meshPtr->vertices.size();
  int ne = m_meshPtr->faces.size();
  int nnb = num_narrowband;

  m_timestep = LARGENUM;
  int nblocks = nnb;
  int nthreads = largest_ele_part;
  std::cout << "nblocks: " << nblocks << ", nthreads: " << nthreads <<
    std::endl;

  if (nthreads <= 32)
  {
    cudaSafeCall((kernel_compute_timestep < 32 > << <nblocks, 32 >> >(full_num_ele, CAST(m_narrowband_d), CAST(m_ele_offsets_d), CAST(m_Rinscribe_d), CAST(m_cadv_local_d), CAST(m_ceik_global_d), CAST(m_ccurv_global_d),
            CAST(timestep_per_block), CAST(Rin_per_block))));
  }
  else if (nthreads <= 64)
  {
    cudaSafeCall((kernel_compute_timestep < 64 > << <nblocks, 64 >> >(full_num_ele, CAST(m_narrowband_d), CAST(m_ele_offsets_d), CAST(m_Rinscribe_d), CAST(m_cadv_local_d), CAST(m_ceik_global_d), CAST(m_ccurv_global_d),
            CAST(timestep_per_block), CAST(Rin_per_block))));
  }
  else if (nthreads <= 128)
  {
    cudaSafeCall((kernel_compute_timestep < 128 > << <nblocks, 128 >> >(full_num_ele, CAST(m_narrowband_d), CAST(m_ele_offsets_d), CAST(m_Rinscribe_d), CAST(m_cadv_local_d), CAST(m_ceik_global_d), CAST(m_ccurv_global_d),
            CAST(timestep_per_block), CAST(Rin_per_block))));
  }
  else if (nthreads <= 256)
  {
    cudaSafeCall((kernel_compute_timestep < 256 > << <nblocks, 256 >> >(full_num_ele, CAST(m_narrowband_d), CAST(m_ele_offsets_d), CAST(m_Rinscribe_d), CAST(m_cadv_local_d), CAST(m_ceik_global_d), CAST(m_ccurv_global_d),
            CAST(timestep_per_block), CAST(Rin_per_block))));
  }
  else if (nthreads <= 512)
  {
    cudaSafeCall((kernel_compute_timestep < 512 > << <nblocks, 512 >> >(full_num_ele, CAST(m_narrowband_d), CAST(m_ele_offsets_d), CAST(m_Rinscribe_d), CAST(m_cadv_local_d), CAST(m_ceik_global_d), CAST(m_ccurv_global_d),
            CAST(timestep_per_block), CAST(Rin_per_block))));
  }
  else if (nthreads <= 1024)
  {
    cudaSafeCall((kernel_compute_timestep < 1024 > << <nblocks, 1024 >> >(full_num_ele, CAST(m_narrowband_d), CAST(m_ele_offsets_d), CAST(m_Rinscribe_d), CAST(m_cadv_local_d), CAST(m_ceik_global_d), CAST(m_ccurv_global_d),
            CAST(timestep_per_block), CAST(Rin_per_block))));
  }
  else
  {
    printf("Error: nthreads greater than 256!!!\n");
  }

  m_timestep = thrust::reduce(timestep_per_block.begin(), timestep_per_block.begin() + nnb, (LevelsetValueType) LARGENUM, thrust::minimum<LevelsetValueType > ());
  m_maxRin = -1.0;
  //  Vector_h tmp = Rin_per_block;
  //  for(int i=0; i<nnb; i++)
  //  {
  //    m_maxRin = fmax(m_maxRin, tmp[i]);
  //  }
  m_maxRin = thrust::reduce(Rin_per_block.begin(), Rin_per_block.begin() + nnb, (LevelsetValueType) - 1.0, thrust::maximum<LevelsetValueType > ());

  //  m_maxRin = -1.0;
  //  for(int eidx = 0; eidx < nnb; eidx++)
  //  {
  //    LevelsetValueType Rin = m_meshPtr->radiusInscribe[eidx];
  //    m_maxRin = fmax(m_maxRin, Rin);
  //    LevelsetValueType minDeltaT1 = LARGENUM;
  //    if(fabs(Ccurv[eidx]) > _EPS) minDeltaT1 = (4.0 * Rin * Rin) / fabs(Ccurv[eidx]) / 2.0 / DIMENSION;
  //    LevelsetValueType minDeltaT2 = LARGENUM;
  //    if(fabs(Ceik[eidx]) + len(Cadv[eidx]) > _EPS) minDeltaT2 = (Rin * 2.0) / (fabs(Ceik[eidx]) + len(Cadv[eidx])) / DIMENSION;
  //    m_timestep = fmin(m_timestep, fmin(minDeltaT1, minDeltaT2));
  //  }

  //  printf("m_timestep is %f, m_maxRin is %f\n", m_timestep, m_maxRin);

}

void meshFIM2d::GenerateData(const char* filename, int nsteps,
    LevelsetValueType timestep, int inside_niter,
    int nside, int block_size, LevelsetValueType bandwidth,
    int part_type, int metis_size, int axis, double domain, bool verbose)
{
  printf("Starting meshFIM2d::GenerateData\n");
  LevelsetValueType oldT1, newT1, oldT2, newT2;
  int nv = m_meshPtr->vertices.size();
  int nt = m_meshPtr->faces.size();

  int squareLength = nside;
  int squareWidth = nside;
  int squareBlockLength = block_size;
  int squareBlockWidth = block_size;
  clock_t starttime, endtime, starttime1, endtime1;
  LevelsetValueType duration, duration1 = 0.0, duration2 = 0.0;

  if (part_type == 1)
    GraphPartition_Square(squareLength, squareWidth, squareBlockLength, squareBlockWidth);
  else //partition with METIS
  {
    Partition_METIS(metis_size);
  }

  if (m_meshPtr->vertT.size() == 0)
    m_meshPtr->vertT.resize(nv);

  Vector_h h_vertT(nv);
  for (int i = 0; i < nv; i++)
  {
    vec3 v1 = (vec3) m_meshPtr->vertices[i];
    m_meshPtr->vertT[i] = v1[axis] - domain;
    h_vertT[i] = m_meshPtr->vertT[i];
  }
  m_vertT_d = h_vertT;
  m_Rinscribe_before_permute_d = Vector_d(m_meshPtr->radiusInscribe);
  largest_Rin = thrust::reduce(m_Rinscribe_before_permute_d.begin(),
      m_Rinscribe_before_permute_d.end(), -1, thrust::maximum<LevelsetValueType > ());

  starttime = clock();
  //Init patches
  InitPatches();
  Vector_h cadv_h(3 * full_num_ele);
  Vector_h ceik_h(full_num_ele);
  Vector_h ccurv_h(full_num_ele);
  for (int i = 0; i < full_num_ele; i++)
  {
    ceik_h[i] = 0.0f;
    ccurv_h[i] = 0.0f;
    cadv_h[0 * full_num_ele + i] = 1.0;
    cadv_h[1 * full_num_ele + i] = 0.0;
    cadv_h[2 * full_num_ele + i] = 0.0;
  }
  m_cadv_global_d = Vector_d(cadv_h);
  m_ceik_global_d = Vector_d(ceik_h);
  m_ccurv_global_d = Vector_d(ccurv_h);
  m_Rinscribe_d = Vector_d(full_num_ele);
  InitPatches2();
  GenerateBlockNeighbors();
  cudaThreadSynchronize();
  printf("After  preprocessing\n");
  //  endtime = CLOCK();
  endtime = clock();
  duration = (LevelsetValueType) (endtime - starttime) / CLOCKS_PER_SEC;
  printf("pre processing time : %.10lf s\n", duration);

  //Inite redistance
  m_redist = new redistance(m_meshPtr, nparts, m_block_xadj_d, m_block_adjncy_d);

  //////////////////////////update values///////////////////////////////////////////
  m_narrowband_d = IdxVector_d(nparts);
  timestep_per_block = Vector_d(nparts);
  Rin_per_block = Vector_d(nparts, -1);
  int num_narrowband = 0;

  //  starttime = CLOCK();
  starttime = clock();
  for (int stepcount = 0; stepcount < nsteps; stepcount++)
  {
    //    starttime1 = clock();
    m_redist->FindSeedPoint(m_narrowband_d, num_narrowband, m_meshPtr, m_vertT_after_permute_d, nparts, largest_vert_part, largest_ele_part, m_largest_num_inside_mem, full_num_ele,
        m_vert_after_permute_d, m_vert_offsets_d, m_ele_after_permute_d, m_ele_offsets_d, m_ele_local_coords_d, m_mem_location_offsets, m_mem_locations,
        m_part_label_d, m_block_xadj_d, m_block_adjncy_d);
    m_redist->ReInitTsign(m_meshPtr, m_vertT_after_permute_d, nparts, largest_vert_part, largest_ele_part, m_largest_num_inside_mem, full_num_ele,
        m_vert_after_permute_d, m_vert_offsets_d, m_ele_after_permute_d, m_ele_offsets_d, m_ele_local_coords_d, m_mem_location_offsets, m_mem_locations,
        m_part_label_d, m_block_xadj_d, m_block_adjncy_d);

    m_redist->GenerateData(m_narrowband_d, num_narrowband, bandwidth, stepcount, m_meshPtr, m_vertT_after_permute_d, nparts, largest_vert_part, largest_ele_part, m_largest_num_inside_mem, full_num_ele,
        m_vert_after_permute_d, m_vert_offsets_d, m_ele_after_permute_d, m_ele_offsets_d, m_ele_local_coords_d, m_mem_location_offsets, m_mem_locations,
        m_part_label_d, m_block_xadj_d, m_block_adjncy_d);
    //    cudaThreadSynchronize();
    //    endtime1 = clock();
    //    duration1 += (LevelsetValueType)(endtime1 - starttime1) / CLOCKS_PER_SEC;

    compute_deltaT(num_narrowband);
    int ninside = bandwidth / m_maxRin / 4.0;
    //    printf("ninside = %d\n", ninside);
    //    starttime1 = clock();
    for (int niter = 0; niter < inside_niter; niter++)
      updateT_single_stage_d(timestep, stepcount, m_narrowband_d, num_narrowband);

    //    cudaThreadSynchronize();
    //    endtime1 = clock();
    //    duration2 += (LevelsetValueType)(endtime1 - starttime1) / CLOCKS_PER_SEC;
    //    tmp_h = m_vertT_after_permute_d;
    //    for(int i = 0; i < tmp_h.size(); i++)
    //    {
    //      printf("After: m_vertT_after_permute_d[%d] = %f\n", i, tmp_h[i]);
    //    }
    //    }
}

cudaThreadSynchronize();
//  endtime = CLOCK();
endtime = clock();
//  printf("redistance time : %.10lf s\n", duration1);
//  printf("levelset update time : %.10lf s\n", duration2);
duration = (LevelsetValueType) (endtime - starttime) / CLOCKS_PER_SEC;
printf("Processing time : %.10lf s\n", duration);
//  ////////////////////////done updating/////////////////////////////////////////////////
//
//  narrowband.clear();
//  m_vertT_after_permute_d = m_redist->m_DT_d;
int nthreads = 256;
int nblocks = min((int) ceil((LevelsetValueType) nv / nthreads), 655535);
cudaSafeCall((kernel_compute_vertT_before_permute << <nblocks, nthreads >> >(nv, CAST(m_vert_permute_d), CAST(m_vertT_after_permute_d), CAST(tmp_vertT_before_permute_d))));
Vector_h vertT_before_permute_h = tmp_vertT_before_permute_d;
for (int i = 0; i < nv; i++)
{
  m_meshPtr->vertT[i] = vertT_before_permute_h[i];
}
writeVTK();
writeFLD();
}

void meshFIM2d::getPartIndicesNegStart(IdxVector_d& sortedPartition, IdxVector_d& partIndices)
{
  // Sizing the array:
  int maxPart = sortedPartition[sortedPartition.size() - 1];
  partIndices.resize(maxPart + 2, 0);

  // Figuring out block sizes for kernel call:
  int size = sortedPartition.size();
  int blockSize = 256;
  int nBlocks = size / blockSize + (size % blockSize == 0 ? 0 : 1);

  // Getting pointers
  int *sortedPartition_d = thrust::raw_pointer_cast(&sortedPartition[0]);
  int *partIndices_d = thrust::raw_pointer_cast(&partIndices[0]);

  // Calling kernel to find indices for each part:
  findPartIndicesNegStartKernel << < nBlocks, blockSize >> > (size, sortedPartition_d, partIndices_d);
  partIndices[partIndices.size() - 1] = size - 1;
}

void meshFIM2d::mapAdjacencyToBlock(IdxVector_d &adjIndexes, IdxVector_d &adjacency, IdxVector_d &adjacencyBlockLabel, IdxVector_d &blockMappedAdjacency, IdxVector_d &fineAggregate)
{
  int size = adjIndexes.size() - 1;
  // Get pointers:adjacencyIn
  int *adjIndexes_d = thrust::raw_pointer_cast(&adjIndexes[0]);
  int *adjacency_d = thrust::raw_pointer_cast(&adjacency[0]);
  int *adjacencyBlockLabel_d = thrust::raw_pointer_cast(&adjacencyBlockLabel[0]);
  int *blockMappedAdjacency_d = thrust::raw_pointer_cast(&blockMappedAdjacency[0]);
  int *fineAggregate_d = thrust::raw_pointer_cast(&fineAggregate[0]);

  // Figuring out block sizes for kernel call:
  int blockSize = 256;
  int nBlocks = size / blockSize + (size % blockSize == 0 ? 0 : 1);

  // Calling kernel:
  mapAdjacencyToBlockKernel << < nBlocks, blockSize >> > (size, adjIndexes_d, adjacency_d, adjacencyBlockLabel_d, blockMappedAdjacency_d, fineAggregate_d);
}

