#ifndef __MESHFIM2D_H__
#define __MESHFIM2D_H__

#include "TriMesh.h"
#include "TriMesh_algo.h"

#include <typeinfo>
#include <functional>
#include <queue>
#include <list>
#include <time.h>

#ifndef _EPS
#define _EPS 1e-5
#endif

class meshFIM2d
{
  public:


    typedef int index;

    enum LabelType
    {
      MaskPoint, SeedPoint, ActivePoint, FarPoint, StopPoint, AlivePoint, ToBeAlivePoint
    };
    double Upwind(index , index );
    void updateT(double, int);
    void MeshReader(char * );

    double LocalSolver(index C, TriMesh::Face triangle, index currentVert);

    void SetSeedPoint(std::vector<index> SeedPoints)
    {
      m_SeedPoints = SeedPoints;
    }

    void setSpeedType(int st)
    {
    }

    void SetMesh(TriMesh* mesh, int nNoiseIter)
    {
      m_meshPtr = mesh;
      orient(m_meshPtr); //  Manasi

      // have to recompute the normals and other attributes required for rendering
      if (!m_meshPtr->normals.empty()) m_meshPtr->normals.clear(); //  Manasi
      m_meshPtr->need_normals(); //  Manasi
      if (!m_meshPtr->adjacentfaces.empty()) m_meshPtr->adjacentfaces.clear(); //  Manasi
      m_meshPtr->need_adjacentfaces(); //  Manasi
      if (!m_meshPtr->across_edge.empty()) m_meshPtr->across_edge.clear(); //  Manasi
      m_meshPtr->need_across_edge(); //  Manasi
      if (!m_meshPtr->tstrips.empty()) m_meshPtr->tstrips.clear(); //  Manasi
      m_meshPtr->need_tstrips(); //  Manasi

      if (/*m_meshPtr->*/SPEEDTYPE == CURVATURE)
      {
        m_meshPtr->need_curvatures();

      }
      if (/*m_meshPtr->*/SPEEDTYPE == NOISE)
      {
        m_meshPtr->need_noise(nNoiseIter);

      }
      m_meshPtr->need_speed();

      m_meshPtr->need_faceedges();

    }

    void InitializeLabels()
    {
      if (!m_meshPtr)
      {
        std::cout << "Label-vector size unknown, please set the mesh first..." << std::endl;
      }
      else
      {
        // initialize all labels to 'Far'
        int nv = m_meshPtr->vertices.size();
        if (m_Label.size() != nv) m_Label.resize(nv);
        for (int l = 0; l < nv; l++)
        {
          m_Label[l] = FarPoint;
        }

        // if seeed-points are present, treat them differently
        if (!m_SeedPoints.empty())
        {
          for (int s = 0; s < m_SeedPoints.size(); s++)
          {
            m_Label[m_SeedPoints[s]] = SeedPoint; //m_Label[s] = LabelType::SeedPoint;
          }
        }
      }
    }

    void InitializeActivePoints()
    {
      if (!m_SeedPoints.empty())
      {
        int ns = m_SeedPoints.size();
        vector<index> nb;
        for (int s = 0; s < ns; s++)
        {
          nb = m_meshPtr->neighbors[m_SeedPoints[s]];
          for (int i = 0; i < nb.size(); i++)
          {
            if (m_Label[nb[i]] != SeedPoint)
            {
              m_ActivePoints.push_back(nb[i]);
              m_Label[nb[i]] = ActivePoint;
            }

          }
        }
      }
    }

    float PointLength(point v)
    {
      float length = 0;
      for (int i = 0; i < 3; i++)
      {
        length += v[i] * v[i];
      }

      return sqrt(length);

    }

    void SetStopDistance(float d)
    {
      m_StopDistance = d;
    }


    void GenerateData(char* filename, int nsteps, int nside);

    meshFIM2d()
    {
      m_meshPtr = NULL;
    };

    ~meshFIM2d()
    {
    };

    TriMesh* m_meshPtr;
    int NumComputation;

  protected:

    std::list<index> m_ActivePoints;
    std::vector<index> m_SeedPoints;
    std::vector<LabelType> m_Label;
    float m_StopDistance;
};


#endif
