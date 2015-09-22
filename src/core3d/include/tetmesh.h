#ifndef TETMESH_H
#define TETMESH_H
/*
   Szymon Rusinkiewicz
   Princeton University

   TriMesh.h
   Class for triangle meshes.
 */

#ifndef  LARGENUM
#define  LARGENUM  1000000000
#endif
#ifndef  SMALLNUM
#define  SMALLNUM  1e-6
#endif
#define  ONE       1
#define  CURVATURE 2
#define  NOISE     3
#define  SPEEDTYPE ONE

#include "Vec.h"
//#include "Color.h"
#include "math.h"
#include <vector>
#include <list>
#include <types.h>

class TetMesh
{
  //protected:
  //  static bool read_helper(const char *filename, TetMesh *mesh);

  public:
    // Types

    struct Face
    {
      int v[3];

      LevelsetValueType speedInv;
      LevelsetValueType T[3];
      Vec < 3, LevelsetValueType> edgeLens; // edge length for 01, 12, 20

      Face()
      {
        speedInv = 0;
      }

      Face(const int &v0, const int &v1, const int &v2)
      {
        v[0] = v0;
        v[1] = v1;
        v[2] = v2;
        speedInv = 0;
      }

      Face(const int *v_)
      {
        v[0] = v_[0];
        v[1] = v_[1];
        v[2] = v_[2];
        speedInv = 0;
      }

      int &operator[] (int i)
      {
        return v[i];
      }

      const int &operator[] (int i)const
      {
        return v[i];
      }

      operator const int * () const
      {
        return &(v[0]);
      }

      operator const int * ()
      {
        return &(v[0]);
      }

      operator int * ()
      {
        return &(v[0]);
      }

      int indexof(int v_) const
      {
        return (v[0] == v_) ? 0 :
          (v[1] == v_) ? 1 :
          (v[2] == v_) ? 2 : -1;
      }

    };

    struct Tet
    {
      int v[4];
      Face f[4];
      LevelsetValueType speedInv;
      //LevelsetValueType T[3];
      //Vec<3,LevelsetValueType> edgeLens;  // edge length for 01, 12, 20

      Tet()
      {
        speedInv = 0;
      }

      Tet(const int &v0, const int &v1, const int &v2, const int &v3)
      {
        v[0] = v0;
        v[1] = v1;
        v[2] = v2;
        v[3] = v3;
        speedInv = 0;
      }

      Tet(const int *v_)
      {
        v[0] = v_[0];
        v[1] = v_[1];
        v[2] = v_[2];
        v[3] = v_[3];
        speedInv = 0;
      }

      int &operator[] (int i)
      {
        return v[i];
      }

      const int &operator[] (int i)const
      {
        return v[i];
      }

      operator const int * () const
      {
        return &(v[0]);
      }

      operator const int * ()
      {
        return &(v[0]);
      }

      operator int * ()
      {
        return &(v[0]);
      }

      int indexof(int v_) const
      {
        return (v[0] == v_) ? 0 :
          (v[1] == v_) ? 1 :
          (v[2] == v_) ? 2 :
          (v[3] == v_) ? 3 : -1;
      }
    };

    // The basics: vertices and faces
    std::vector<point> vertices;
    std::vector<Face> faces;
    std::vector<Tet> tets;

    // Computed per-vertex properties
    std::vector<vec> normals;
    std::vector<vec> pdir1, pdir2;
    std::vector<LevelsetValueType> curv1, curv2;
    std::vector< Vec < 4, LevelsetValueType> > dcurv;
    std::vector<vec> cornerareas;
    std::vector<LevelsetValueType> pointareas;
    std::vector<LevelsetValueType> vertT;

    // Connectivity structures:
    //  For each vertex, all neighboring vertices
    std::vector< std::vector<int> > neighbors;
    //  For each vertex, all neighboring faces
    std::vector< std::vector<int> > adjacenttets;
    std::vector<Tet> across_face;

    std::vector<LevelsetValueType> radiusInscribe;


    std::vector< std::vector<Tet> > vertOneringTets;

    std::vector<LevelsetValueType> noiseOnVert;

    void reorient();
    void rescale(LevelsetValueType size);
    void need_faceedges();
    void need_speed();
    void need_noise();
    void need_oneringtets();
    void need_normals();
    void need_pointareas();
    void need_neighbors(bool verbose = false);
    void need_adjacenttets(bool verbose = false);
    void need_across_face();
    //void need_across_edge();
    void need_meshinfo();
    void need_Rinscribe();
    bool IsNonObtuse(int v, Tet t);
    void SplitFace(std::vector<Tet> &acTets, int v, Tet ct, int nfAdj);
    std::vector<Tet> GetOneRing(int v);

    // FIM: check angle for at a given vertex, for a given face

    bool IsNonObtuse(int v, Face f)
    {
      int iV = f.indexof(v);

      point A = this->vertices[v];
      point B = this->vertices[f[(iV + 1) % 3]];
      point C = this->vertices[f[(iV + 2) % 3]];

      LevelsetValueType a = dist(B, C);
      LevelsetValueType b = dist(A, C);
      LevelsetValueType c = dist(A, B);

      LevelsetValueType angA = 0.0; /* = acos( (b*b + c*c - a*a) / (2*b*c) )*/

      if ((a > 0) && (b > 0) && (c > 0))//  Manasi stack overflow
      {
        angA = acos((b * b + c * c - a * a) / (2 * b * c)); //  Manasi stack overflow
      }

      //return 1;
      return (angA < M_PI / 2.0f);
    }

    // Debugging printout, controllable by a "verbose"ness parameter
    static int verbose;
    static void set_verbose(int);
    static int dprintf(const char *format, ...);

    void init(float* pointlist, int numpoint, int*trilist, int numtri, int* tetlist, int numtet, int numattr, float* attrlist, bool verbose = false);

    //Constructor
    TetMesh()
    {
    }
};

#endif
