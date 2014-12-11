#ifndef TETMESH_H
#define TETMESH_H
/*
Szymon Rusinkiewicz
Princeton University

TriMesh.h
Class for triangle meshes.
 */

#define  LARGENUM  1000000000
#define  SMALLNUM  1e-6
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
using std::vector;

class TetMesh
{
  //protected:
  //	static bool read_helper(const char *filename, TetMesh *mesh);

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
    }

    Face(const int &v0, const int &v1, const int &v2)
    {
      v[0] = v0;
      v[1] = v1;
      v[2] = v2;
    }

    Face(const int *v_)
    {
      v[0] = v_[0];
      v[1] = v_[1];
      v[2] = v_[2];
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
    }

    Tet(const int &v0, const int &v1, const int &v2, const int &v3)
    {
      v[0] = v0;
      v[1] = v1;
      v[2] = v2;
      v[3] = v3;
    }

    Tet(const int *v_)
    {
      v[0] = v_[0];
      v[1] = v_[1];
      v[2] = v_[2];
      v[3] = v_[3];
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
  vector<point> vertices;
  vector<Face> faces;
  vector<Tet> tets;
//  vector<Color> colors;

  // Computed per-vertex properties
  vector<vec> normals;
  vector<vec> pdir1, pdir2;
  vector<LevelsetValueType> curv1, curv2;
  vector< Vec < 4, LevelsetValueType> > dcurv;
  vector<vec> cornerareas;
  vector<LevelsetValueType> pointareas;
  vector<LevelsetValueType> vertT;

  // Connectivity structures:
  //  For each vertex, all neighboring vertices
  vector< vector<int> > neighbors;
  //  For each vertex, all neighboring faces
  vector< vector<int> > adjacenttets;
  vector<Tet> across_face;

  vector<LevelsetValueType> radiusInscribe;


  vector< vector<Tet> > vertOneringTets;
  //  For each face, the three faces attached to its edges
  //  (for example, across_edge[3][2] is the number of the face
  //   that's touching the edge opposite vertex 2 of face 3)
  //vector<Face> across_edge;

  vector<LevelsetValueType> noiseOnVert;
  //vector<LevelsetValueType> noiseOnFace;


  //int SPEEDTYPE;
  // Compute all this stuff...
  //void setSpeedType(int st)
  //{
  //ST = st;
  //}
  //void need_tstrips();
  //void convert_strips(tstrip_rep rep);
  //void unpack_tstrips();
  //void triangulate_grid();
  //void need_faces()
  //{
  //	if (!faces.empty())
  //		return;
  //	if (!tstrips.empty())
  //		unpack_tstrips();
  //	else if (!grid.empty())
  //		triangulate_grid();
  //}
	void reorient();
  void rescale(LevelsetValueType size);
  void need_faceedges();
  void need_speed();
  void need_noise();
  void need_oneringtets();
  void need_normals();
  void need_pointareas();
  void need_neighbors();
  void need_adjacenttets();
  void need_across_face();
  //void need_across_edge();
  void need_meshinfo();
  void need_Rinscribe();
  bool IsNonObtuse(int v, Tet t);
  void SplitFace(vector<Tet> &acTets, int v, Tet ct, int nfAdj);
  vector<Tet> GetOneRing(int v);

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

  void init(LevelsetValueType* pointlist, int numpoint, int*trilist, int numtri, int* tetlist, int numtet, int numattr, LevelsetValueType* attrlist);

  //Constructor
  TetMesh()
  {
  }
};

#endif
