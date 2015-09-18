#include "meshFIM2d.h"
#include "Vec.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

void meshFIM2d::updateT(double timestep, int nside)
{
  vec2d sigma(1.0, 0.0);
  double epsilon = 1.0;
  int nv = m_meshPtr->vertices.size();
  int nf = m_meshPtr->faces.size();
  vector<double> values(3);
  vector<double> up(nv, 0.0);
  vector<double> down(nv, 0.0);
  vector<vec2d> node_grad_phi_up(nv, vec2d(0.0, 0.0));
  vector<double> node_grad_phi_down(nv, 0.0);
  vector<double> curv_up(nv, 0.0);

  for (int fidx = 0; fidx < nf; fidx++)
  {
    for (int j = 0; j < 3; j++)
    {
      values[j] = m_meshPtr->vertT[m_meshPtr->faces[fidx][j]];
    }
    //compute ni normals
    vector<vec3d> nodes(3);
    nodes[0] = (vec3d) m_meshPtr->vertices[m_meshPtr->faces[fidx][0]];
    nodes[1] = (vec3d) m_meshPtr->vertices[m_meshPtr->faces[fidx][1]];
    nodes[2] = (vec3d) m_meshPtr->vertices[m_meshPtr->faces[fidx][2]];
    double area = len((nodes[1] - nodes[0]) CROSS(nodes[2] - nodes[0])) / 2.0;

    //compute inverse of A
    double a, b, c, d, e, f, g, h, k;
    a = nodes[0][0], b = nodes[0][1], c = 1;
    d = nodes[1][0], e = nodes[1][1], f = 1;
    g = nodes[2][0], h = nodes[2][1], k = 1;
    double detA = a * (e * k - f * h) - b * (k * d - f * g) + c * (d * h - e * g);
    vector<vec3d> Arows(3);
    Arows[0] = vec3d(e * k - f * h, c * h - b * k, b * f - c * e);
    Arows[1] = vec3d(f * g - d * k, a * k - c * g, c * d - a * f);
    Arows[2] = vec3d(d * h - e * g, g * b - a * h, a * e - b * d);
    vector<vec2d> nablaN(3);
    for (int i = 0; i < 3; i++)
    {
      vec3d RHS(0.0, 0.0, 0.0);
      RHS[i] = 1.0;
      nablaN[i][0] = (Arows[0] DOT RHS) / detA;
      nablaN[i][1] = (Arows[1] DOT RHS) / detA;
    }

    //compuate grad of Phi
    vec2d nablaPhi(0.0, 0.0);
    for (int i = 0; i < 3; i++)
    {
      nablaPhi[0] += nablaN[i][0] * values[i];
      nablaPhi[1] += nablaN[i][1] * values[i];
    }
    double abs_nabla_phi = len(nablaPhi);
    //compute K and Kplus and Kminus
    vector<double> Kplus(3);
    vector<double> Kminus(3);
    vector<double> K(3);
    double Hintegral = 0.0;
    double beta = 0;
    for (int i = 0; i < 3; i++)
    {
      K[i] = area * (sigma DOT nablaN[i]); // for H(\nabla u) = sigma DOT \nabla u
      Hintegral += K[i] * values[i];
      Kplus[i] = fmax(K[i], 0.0);
      Kminus[i] = fmin(K[i], 0.0);
      beta += Kminus[i];
    }
    beta = 1.0 / beta;

    if (fabs(Hintegral) > 1e-16)
    {

      vector<double> delta(3);
      for (int i = 0; i < 3; i++)
      {
        delta[i] = Kplus[i] * beta * (Kminus[0] * (values[i] - values[0]) + Kminus[1] * (values[i] - values[1]) + Kminus[2] * (values[i] - values[2]));
      }

      vector<double> alpha(3);
      for (int i = 0; i < 3; i++)
      {
        alpha[i] = delta[i] / Hintegral;
      }

      double theta = 0;
      for (int i = 0; i < 3; i++)
      {
        theta += fmax(0.0, alpha[i]);
      }

      vector<double> alphatuda(3);
      for (int i = 0; i < 3; i++)
      {
        alphatuda[i] = fmax(alpha[i], 0.0) / theta;
      }

      for (int i = 0; i < 3; i++)
      {
        up[m_meshPtr->faces[fidx][i]] += alphatuda[i] * Hintegral;
        down[m_meshPtr->faces[fidx][i]] += alphatuda[i] * area;
        node_grad_phi_up[m_meshPtr->faces[fidx][i]] += area * nablaPhi;
        node_grad_phi_down[m_meshPtr->faces[fidx][i]] += area;
        curv_up[m_meshPtr->faces[fidx][i]] += area * ((nablaN[i] DOT nablaN[i]) / abs_nabla_phi * values[i] +
            (nablaN[i] DOT nablaN[(i + 1) % 3]) / abs_nabla_phi * values[(i + 1) % 3] +
            (nablaN[i] DOT nablaN[(i + 2) % 3]) / abs_nabla_phi * values[(i + 2) % 3]);
      }
    }
  }

  for (int vidx = 0; vidx < nv; vidx++)
  {
    if (fabs(down[vidx]) > 1e-16)
    {
      double eikonal = up[vidx] / down[vidx];
      double curvature = curv_up[vidx] / node_grad_phi_down[vidx];
      double node_eikonal = len(node_grad_phi_up[vidx]) / node_grad_phi_down[vidx];
      m_meshPtr->vertT[vidx] -= eikonal * timestep;
    }
  }

}

void meshFIM2d::GenerateData(char* filename, int nsteps, int nside)
{
  double oldT1, newT1, oldT2, newT2;
  index tmpIndex1, tmpIndex2;
  vector<int> nb;
  int currentVert;
  int nv = m_meshPtr->vertices.size();
  int nf = m_meshPtr->faces.size();
  NumComputation = 0;

  vec3d origin1 = vec3d(8, 8, 0);
  double radius = 3;
  double radiussqure = radius*radius;

  if (m_meshPtr->vertT.size() == 0)
    m_meshPtr->vertT.resize(nv);
  for (int i = 0; i < m_meshPtr->vertices.size(); i++)
  {
    vec3d v1 = (vec3d) m_meshPtr->vertices[i];
    vec3d v1o1 = v1 - origin1;
    m_meshPtr->vertT[i] = v1[0] - 7.7;
  }

  FILE* vtkfile;
  vtkfile = fopen("after0step.vtk", "w+");
  if (!vtkfile)
    printf("The vtk file was not opened\n");
  fprintf(vtkfile, "# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\n");
  fprintf(vtkfile, "POINTS %d float\n", nv);
  for (int i = 0; i < nv; i++)
  {
    fprintf(vtkfile, "%.12lf %.12lf %.12lf\n", m_meshPtr->vertices[i][0], m_meshPtr->vertices[i][1], m_meshPtr->vertices[i][2]);
  }
  fprintf(vtkfile, "POLYGONS %d %d\n", nf, nf * 4);
  for (int i = 0; i < nf; i++)
  {
    fprintf(vtkfile, "3 %d %d %d\n", m_meshPtr->faces[i][0], m_meshPtr->faces[i][1], m_meshPtr->faces[i][2]);
  }

  fprintf(vtkfile, "POINT_DATA %d\nSCALARS traveltime float 1\nLOOKUP_TABLE default\n", nv);
  for (int i = 0; i < nv; i++)
  {
    fprintf(vtkfile, "%.12lf\n", m_meshPtr->vertT[i]);
  }
  fclose(vtkfile);

  //////////////////////////update values///////////////////////////////////////////
  clock_t starttime, endtime;

  starttime = clock();
  double timestep = 0.05;
  for (int stepcount = 0; stepcount < nsteps; stepcount++)
  {
    updateT(timestep, nside);
  }

  endtime = clock();
  double duration = (double) (endtime - starttime) *1000 / CLOCKS_PER_SEC;
  printf("Processing time : %.10lf ms\n", duration);
  ////////////////////////done updating/////////////////////////////////////////////////

  vtkfile = fopen("after1000step.vtk", "w+");
  if (!vtkfile)
    printf("The vtk file was not opened\n");
  fprintf(vtkfile, "# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\n");
  fprintf(vtkfile, "POINTS %d float\n", nv);
  for (int i = 0; i < nv; i++)
  {
    fprintf(vtkfile, "%.12lf %.12lf %.12lf\n", m_meshPtr->vertices[i][0], m_meshPtr->vertices[i][1], m_meshPtr->vertices[i][2]);
  }
  fprintf(vtkfile, "POLYGONS %d %d\n", nf, nf * 4);
  for (int i = 0; i < nf; i++)
  {
    fprintf(vtkfile, "3 %d %d %d\n", m_meshPtr->faces[i][0], m_meshPtr->faces[i][1], m_meshPtr->faces[i][2]);
  }

  fprintf(vtkfile, "POINT_DATA %d\nSCALARS traveltime float 1\nLOOKUP_TABLE default\n", nv);
  for (int i = 0; i < nv; i++)
  {
    fprintf(vtkfile, "%.12lf\n", m_meshPtr->vertT[i]);
  }
  fclose(vtkfile);
}
