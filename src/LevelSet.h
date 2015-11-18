#ifndef __LEVELSET_H__
#define __LEVELSET_H__

#include <cstdio>
#include <cstdlib>
#include <meshFIM2d.h>
#include <meshFIM3d.h>
#include <math.h>
#include <TriMesh.h>
#include <tetmesh.h>
#include <tetgen.h>
#include <cuda_runtime.h>
#include <time.h>
#include <cutil.h>
#include <cstring>
#include <limits>

/** The class that represents all of the available options for LevelSet */
class LevelSet{
public:
  LevelSet(bool isTriMesh, 
    std::string fname = "../src/test/test_data/sphere_266verts.ply",
    bool verbose = false);
  virtual ~LevelSet();

  //helper functions
  std::vector < float >& getResultAtIteration(size_t i);
  size_t numIterations();
  void writeVTK();
  void writeFLD();
  //main functions
  //initialize the vertex values
  void initializeVertices(std::vector<float> values);
  //initialize the element advection
  void initializeAdvection(std::vector<point> values);
  void initializeMesh();
  /**
  * Creates the mesh, partitions the mesh, and runs the algorithm.
  */
  void solveLevelSet();
  /**
  * This function uses the provided analytical solutions to
  * visualize the algorithm's error after each iteration.
  *
  * @param solution The vector of expected solutions.
  */
  void printErrorGraph(std::vector<float> solution); 
  //data
  bool verbose_;
  std::string filename_;
  int partitionType_;
  int numSteps_;
  double timeStep_;
  int insideIterations_;
  int blockSize_;
  int sideLengths_;
  double bandwidth_;
  int metisSize_;
  bool userSetInitial_;
  bool userSetAdvection_;
  TriMesh* triMesh_;
  TetMesh* tetMesh_;
  std::vector < std::vector <float> > time_values_;
  meshFIM2d *FIMPtr2d_;
  meshFIM3d *FIMPtr3d_;
  bool isTriMesh_;
};

#endif
