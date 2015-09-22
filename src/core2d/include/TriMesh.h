#ifndef TRIMESH_H
#define TRIMESH_H
/*
   Szymon Rusinkiewicz
   Princeton University

   TriMesh.h
   Class for triangle meshes.
 */

#ifndef  LARGENUM
#define  LARGENUM  10000000.0
#endif
#define  ONE       1
#define  CURVATURE 2
#define  NOISE     3
#define  SPEEDTYPE ONE

#include "Vec.h"
#include "Color.h"
#include "math.h"
#include <vector>
#include <list>
#include <stdio.h>
using std::vector;



#define MIN(a,b) ( (a)< (b) )?(a):(b)
#define MAX(a,b) ((a)>(b))?(a):(b)

class TriMesh
{
  protected:
    static bool read_helper(const char *filename, TriMesh *mesh);

  public:
    // Types

    struct Face
    {
      int v[3];
      float speedInv;
      float T[3];
      Vec < 3, float> edgeLens; // edge length for 01, 12, 20

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

    struct BBox
    {
      point min, max;

      point center() const
      {
        return point((min[0] + max[0]) * 0.5f,
            (min[1] + max[1]) * 0.5f,
            (min[2] + max[2]) * 0.5f);
      }

      vec size() const
      {
        return max - min;
      }
      bool valid;

      BBox() : valid(false)
      {
      }
    };

    struct BSphere
    {
      point center;
      float r;
      bool valid;

      BSphere() : valid(false)
      {
      }
    };

    // Enums

    enum tstrip_rep
    {
      TSTRIP_LENGTH, TSTRIP_TERM
    };

    enum
    {
      GRID_INVALID = -1
    };
    //enum speed_type { ONE = 0, CURVATURE, NOISE };

    // The basics: vertices and faces
    vector<point> vertices;
    vector<Face> faces;


    // Triangle strips
    vector<int> tstrips;

    // Grid, if present
    vector<int> grid;
    int grid_width, grid_height;

    // Other per-vertex properties
    vector<Color> colors;
    vector<float> confidences;
    vector<unsigned> flags;
    unsigned flag_curr;

    // Computed per-vertex properties
    vector<vec> normals;
    vector<vec> pdir1, pdir2;
    vector<float> curv1, curv2;
    vector< Vec < 4, float> > dcurv;
    vector<vec> cornerareas;
    vector<float> pointareas;
    vector< float> vertT;

    // Bounding structures
    BBox bbox;
    BSphere bsphere;

    // Connectivity structures:
    //  For each vertex, all neighboring vertices
    vector< vector<int> > neighbors;
    //  For each vertex, all neighboring faces
    vector< vector<int> > adjacentfaces;

    vector<float> radiusInscribe;


    vector< vector<Face> > vertOneringFaces;
    //  For each face, the three faces attached to its edges
    //  (for example, across_edge[3][2] is the number of the face
    //   that's touching the edge opposite vertex 2 of face 3)
    vector<Face> across_edge;

    vector<float> noiseOnVert;
    vector<float> noiseOnFace;


    //int SPEEDTYPE;
    // Compute all this stuff...

    void setSpeedType(int st)
    {
      //ST = st;
    }
    void need_tstrips();
    void convert_strips(tstrip_rep rep);
    void unpack_tstrips();
    void triangulate_grid();

    void need_faces()
    {
      if (!faces.empty())
        return;
      if (!tstrips.empty())
        unpack_tstrips();
      else if (!grid.empty())
        triangulate_grid();
    }

    void need_faceedges();
    void need_speed();
    void need_noise();
    void need_oneringfaces();
    void need_normals();
    void need_pointareas();
    void need_curvatures();
    void need_dcurv();
    void need_bbox();
    void need_bsphere();
    void need_neighbors();
    void need_adjacentfaces();
    void need_across_edge();
    void need_meshinfo();
    void need_Rinscribe();

    // Input and output
    static TriMesh *read(const char *filename);
    void write(const char *filename);

    // Statistics
    // XXX - Add stuff here
    float feature_size();

    // Useful queries
    // XXX - Add stuff here

    bool is_bdy(int v)
    {
      if (neighbors.empty()) need_neighbors();
      if (adjacentfaces.empty()) need_adjacentfaces();
      return neighbors[v].size() != adjacentfaces[v].size();
    }

    vec trinorm(int f)
    {
      if (faces.empty()) need_faces();
      return ::trinorm(vertices[faces[f][0]], vertices[faces[f][1]],
          vertices[faces[f][2]]);
    }

    // FIM: check angle for at a given vertex, for a given face

    bool IsNonObtuse(int v, Face f)
    {
      int iV = f.indexof(v);

      point A = this->vertices[v];
      point B = this->vertices[f[(iV + 1) % 3]];
      point C = this->vertices[f[(iV + 2) % 3]];

      float a = dist(B, C);
      float b = dist(A, C);
      float c = dist(A, B);

      float angA = 0.0; /* = acos( (b*b + c*c - a*a) / (2*b*c) )*/

      if ((a > 0) && (b > 0) && (c > 0))//  Manasi stack overflow
      {//  Manasi stack overflow
        angA = acos((b * b + c * c - a * a) / (2 * b * c)); //  Manasi stack overflow
      }//  Manasi stack overflow

      //return 1;

      return (angA < M_PI / 2.0f);
    }

    // FIM: given a vertex, find an all-acute neighborhood of faces

    void SplitFace(vector<Face> &acFaces, int v, Face cf, int nfAdj/*, int currentVert*/)
    {
      // get all the four vertices in order
      /* v1         v4
         +-------+
         \     . \
         \   .   \
         \ .     \
         +-------+
         v2         v3 */

      need_neighbors();
      int iV = cf.indexof(v); // get index of v in terms of cf
      int v1 = v;
      int v2 = cf[(iV + 1) % 3];
      int v4 = cf[(iV + 2) % 3];
      iV = this->faces[nfAdj].indexof(v2); // get index of v in terms of adjacent face
      int v3 = this->faces[nfAdj][(iV + 1) % 3];
      neighbors[v3].push_back(v1);
      Face af = faces[nfAdj];

      // create faces (v1,v3,v4) and (v1,v2,v3), check angle at v1
      Face f1(v1, v3, v4);
      //f1.T[f1.indexof(v1)] = this->vertT[currentVert][v1];
      //f1.T[f1.indexof(v3)] = this->vertT[currentVert][v3];
      //f1.T[f1.indexof(v4)] = this->vertT[currentVert][v4];


      Face f2(v1, v2, v3);
      //f2.T[f2.indexof(v1)] = this->vertT[currentVert][v1];
      //f2.T[f2.indexof(v2)] = this->vertT[currentVert][v2];
      //f2.T[f2.indexof(v3)] = this->vertT[currentVert][v3];



      if (IsNonObtuse(v, f1))
      {
        switch (SPEEDTYPE)
        {


        case CURVATURE:
          f1.speedInv = (abs(curv1[f1[0]] + curv2[f1[0]]) + abs(curv1[f1[1]] + curv2[f1[1]]) + abs(curv1[f1[2]] + curv2[f1[2]])) / 6;
          break;
        case ONE:
          f1.speedInv = 1.0;
          break;
        case NOISE:
          f1.speedInv = /*0.5*(cf.speedInv + af.speedInv);*/(noiseOnVert[f1[0]] + noiseOnVert[f1[1]] + noiseOnVert[f1[2]]) / 3;
          break;
        default:
          f1.speedInv = 1.0;
          break;

        }
        point edge01 = vertices[f1[1]] - vertices[f1[0]];
        point edge12 = vertices[f1[2]] - vertices[f1[1]];
        point edge20 = vertices[f1[0]] - vertices[f1[2]];
        f1.edgeLens[0] = sqrt(edge01[0] * edge01[0] + edge01[1] * edge01[1] + edge01[2] * edge01[2]);
        f1.edgeLens[1] = sqrt(edge12[0] * edge12[0] + edge12[1] * edge12[1] + edge12[2] * edge12[2]);
        f1.edgeLens[2] = sqrt(edge20[0] * edge20[0] + edge20[1] * edge20[1] + edge20[2] * edge20[2]);
        acFaces.push_back(f1);
      }
      else
      {
        int nfAdj_new = this->across_edge[nfAdj][this->faces[nfAdj].indexof(v2)];
        if (nfAdj_new > -1)
        {
          SplitFace(acFaces, v, f1, nfAdj_new/*, currentVert*/);

        }
        else
          printf("NO cross edge!!! Maybe a hole!!\n");
        //SplitFace(acFaces,v,f1,nfAdj_new, currentVert);
      }

      if (IsNonObtuse(v, f2))
      {
        switch (SPEEDTYPE)
        {


        case CURVATURE:
          f2.speedInv = (abs(curv1[f2[0]] + curv2[f2[0]]) + abs(curv1[f2[1]] + curv2[f2[1]]) + abs(curv1[f2[2]] + curv2[f2[2]])) / 6;
          break;
        case ONE:
          f2.speedInv = 1.0;
          break;
        case NOISE:
          f2.speedInv = /*f1.speedInv = 0.5*(cf.speedInv + af.speedInv);*/(noiseOnVert[f2[0]] + noiseOnVert[f2[1]] + noiseOnVert[f2[2]]) / 3;
          break;
        default:
          f2.speedInv = 1.0;
          break;

        }
        point edge01 = vertices[f2[1]] - vertices[f2[0]];
        point edge12 = vertices[f2[2]] - vertices[f2[1]];
        point edge20 = vertices[f2[0]] - vertices[f2[2]];
        f2.edgeLens[0] = sqrt(edge01[0] * edge01[0] + edge01[1] * edge01[1] + edge01[2] * edge01[2]);
        f2.edgeLens[1] = sqrt(edge12[0] * edge12[0] + edge12[1] * edge12[1] + edge12[2] * edge12[2]);
        f2.edgeLens[2] = sqrt(edge20[0] * edge20[0] + edge20[1] * edge20[1] + edge20[2] * edge20[2]);
        acFaces.push_back(f2);
      }
      else
      {
        int nfAdj_new = this->across_edge[nfAdj][this->faces[nfAdj].indexof(v4)];
        if (nfAdj_new > -1)
        {
          SplitFace(acFaces, v, f2, nfAdj_new/*,currentVert*/);
        }
        else
          printf("NO cross edge!!! Maybe a hole!!\n");
        //SplitFace(acFaces,v,f2,nfAdj_new,currentVert);
      }
    }

    // FIM: one ring function

    vector<Face> GetOneRing(int v/*, int currentVert*/)
    {
      // make sure we have the across-edge map
      if (this->across_edge.empty())
        this->need_across_edge();

      // variables required
      vector<Face> oneRingFaces;
      vector<Face> t_faces;

      // get adjacent faces
      int naf = this->adjacentfaces[v].size();

      if (!naf)
      {
        std::cout << "vertex " << v << " has 0 adjacent faces..." << std::endl;
      }
      else
      {
        for (int af = 0; af < naf; af++)
        {
          Face cf = this->faces[adjacentfaces[v][af]];

          t_faces.clear();
          if (IsNonObtuse(v, cf))// check angle: if non-obtuse, return existing face
          {
            //this->colors[cf[0]] = Color::red();
            //this->colors[cf[1]] = Color::red();
            //this->colors[cf[2]] = Color::red();
            t_faces.push_back(cf);
          }
          else
          {
            int nfae = this->across_edge[this->adjacentfaces[v][af]][cf.indexof(v)];
            if (nfae > -1)
            {
              SplitFace(t_faces, v, cf, nfae/*,currentVert*/); // if obtuse, split face till we get all acute angles
            }
            else
              printf("NO cross edge!!! Maybe a hole!!\n");
            //SplitFace(t_faces,v,cf,nfae,currentVert);// if obtuse, split face till we get all acute angles
          }

          for (int tf = 0; tf < t_faces.size(); tf++)
          {
            //this->colors[t_faces[tf][0]] = Color::red();
            //this->colors[t_faces[tf][1]] = Color::red();
            //this->colors[t_faces[tf][2]] = Color::red();
            oneRingFaces.push_back(t_faces[tf]);
          }
        }
      }
      //this->colors[v] = Color::green();
      return oneRingFaces;
    }

    // FIM: initialize attributes
    //typedef std::<int> ListType;

    //  void InitializeAttributes(int currentVert, std::vector<int> seeds = vector<int>())
    //  {
    //    // initialize the travel times over all vertices...
    //    int nv = this->vertices.size();
    //    vertT.resize(nv);
    //    vertT.assign(nv, LARGENUM);
    //
    //    //    for (int v = 0; v < nv; v++)
    //    //    {
    //    //      this->vertT.push_back(LARGENUM);  //modified from this->vertT[v] = 1000000.0)
    //    //    }
    //
    //    //vector<int> nb;
    //
    //    // initialize seed points if present...
    //    if (!seeds.empty())
    //    {
    //      int ns = seeds.size();
    //      for (int s = 0; s < ns; s++)
    //      {
    //        this->vertT[seeds[s]] = 0.0; //modified from this->vertT[s] = 0.0;
    //        //nb = this->neighbors[seeds[s]];
    //
    //      }
    //
    //
    //    }
    //
    //    // pre-compute faces, normals, and other per-vertex properties that may be needed
    //    this->need_neighbors();
    //    this->need_normals();
    //    this->need_adjacentfaces();
    //    this->need_across_edge();
    //    this->need_faces();
    //
    //    // for all faces: initialize per-vertex travel time and face-speed
    //    int nf = this->faces.size();
    //    for (int f = 0; f < nf; f++)
    //    {
    //      Face cf = this->faces[f];
    //
    //      // travel time
    //      faces[f].T[0] = this->vertT[cf[0]];
    //      faces[f].T[1] = this->vertT[cf[1]];
    //      faces[f].T[2] = this->vertT[cf[2]];
    //
    //      // speed
    //      //faces[f].speedInv = 1.0;
    //    }
    //  }

    // Debugging printout, controllable by a "verbose"ness parameter
    static int verbose;
    static void set_verbose(int);
    static int dprintf(const char *format, ...);

    // Constructor

    TriMesh() : grid_width(-1), grid_height(-1), flag_curr(0)
  {
  }
};

#endif
