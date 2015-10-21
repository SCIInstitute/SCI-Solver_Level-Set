#ifndef __LEVELSET3D_H__
#define __LEVELSET3D_H__

#include <cstdio>
#include <cstdlib>
#include <meshFIM.h>
#include <math.h>
#include <tetmesh.h>
#include <tetgen.h>
#include <cuda_runtime.h>
#include <time.h>
#include <mycutil.h>
#include <cstring>
#include <limits>

namespace LevelSet3d {
  /** The class that represents all of the available options for LevelSet */
  class LevelSet3d {
    public:
      LevelSet3d(std::string fname = "../src/test/test_data/sphere334",
          bool verbose = false) :
        verbose_(verbose),
        filename_(fname),
        partitionType_(0),
        numSteps_(10),
        timeStep_(1.),
        insideIterations_(1),
        blockSize_(16),
        sideLengths_(1),
        bandwidth_(16.),
        metisSize_(16),
        userSetInitial_(false),
        userSetAdvection_(false)
    {}
      //data
      bool verbose_;
      std::string filename_;
      int partitionType_;
      int numSteps_;
      double timeStep_;
      int insideIterations_;
      int blockSize_;
      int sideLengths_;
      LevelsetValueType bandwidth_;
      int metisSize_;
      bool userSetInitial_;
      bool userSetAdvection_;
  };

  //The static pointer to the mesh
  static TetMesh * mesh_ = NULL;
  //the answer vector
  static std::vector < std::vector <LevelsetValueType> > time_values_;
  //accessor functions to the results.
  std::vector < LevelsetValueType >& getResultAtIteration(size_t i) {
    return time_values_.at(i);
  }
  size_t numIterations() { return time_values_.size(); }
  void writeVTK() {
    meshFIM FIMPtr(mesh_);
    FIMPtr.writeVTK(time_values_);
  }
  void writeFLD() {
    meshFIM FIMPtr(mesh_);
    FIMPtr.writeFLD();
  }
  //initialize the vertex values
  void initializeVertices(LevelSet3d &data, std::vector<double> values) {
    if (mesh_ == NULL) {
      std::cerr << "You must initialize the mesh first!" << std::endl;
      exit(0);
    }
    if (values.size() != mesh_->vertices.size()) {
      std::cerr << "Initialize values size does not match number of vertices!"
        << std::endl;
      exit(0);
    }
    mesh_->vertT.resize(mesh_->vertices.size());
    for (size_t i = 0; i < values.size(); i++) {
      mesh_->vertT[i] = values[i];
    }
    data.userSetInitial_ = true;
  }
  //initialize the element advection
  void initializeAdvection(LevelSet3d &data, std::vector<point> values) {
    if (mesh_ == NULL) {
      std::cerr << "You must initialize the mesh first!" << std::endl;
      exit(0);
    }
    if (values.size() != mesh_->tets.size()) {
      std::cerr << "Initialize values size does not match number of elements!"
        << std::endl;
      exit(0);
    }
    mesh_->normals.resize(mesh_->tets.size());
    for (size_t i = 0; i < values.size(); i++) {
      mesh_->normals[i] = values[i];
    }
    data.userSetAdvection_ = true;
  }

  void initializeMesh(LevelSet3d data = LevelSet3d()) {
    if (mesh_ == NULL) {
      mesh_ = new TetMesh();
      tetgenio in, addin, bgmin, out;
      if(!in.load_tetmesh((char*)data.filename_.c_str(),data.verbose_))
      {
        printf("File open failed!!\n");
        exit(0);
      }

      mesh_->init(in.pointlist, in.numberofpoints, in.trifacelist,
          in.numberoffacets, in.tetrahedronlist, in.numberoftetrahedra,
          in.numberoftetrahedronattributes, in.tetrahedronattributelist,
          data.verbose_);
      mesh_->reorient();
      mesh_->need_neighbors(data.verbose_);
      mesh_->need_adjacenttets(data.verbose_);
    }
  }

  /**
   * Creates the mesh, partitions the mesh, and runs the algorithm.
   *
   * @data The set of options for the LevelSet algorithm.
   *       The defaults are used if nothing is provided.
   */
  void solveLevelSet(LevelSet3d data = LevelSet3d()) {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device;
    for(device = 0; device < deviceCount; ++device)
    {
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, device);
      if (data.verbose_)
        printf("Device %d has compute capability %d.%d.\n",
            device, deviceProp.major, deviceProp.minor);
    }
    device = 0;

    cudaSetDevice(device);
    if(cudaDeviceReset() != cudaSuccess) {
      std::cout << "There is a problem with CUDA device " << device << std::endl;
      exit(0);
    }

    cudaSafeCall((cudaDeviceSetCacheConfig(cudaFuncCachePreferShared)));
    if (data.verbose_)
      printf("Narrowband width is %f\n", data.bandwidth_);
    clock_t starttime, endtime;
    starttime = clock();

    if (mesh_ == NULL) {
      initializeMesh(data);
    }
    float mn = std::numeric_limits<float>::max();
    float mx = std::numeric_limits<float>::min();
    for (size_t i = 0; i < mesh_->vertices.size(); i++) {
      mn = std::min(mn, static_cast<float>(mesh_->vertices[i][0]));
      mx = std::max(mx, static_cast<float>(mesh_->vertices[i][0]));
    }
    //populate advection if it's empty
    if (!data.userSetAdvection_) {
      mesh_->normals.resize(mesh_->tets.size());
      for (size_t i = 0; i < mesh_->tets.size(); i++) {
        mesh_->normals[i] =  point((mx - mn) / 100.,0,0);
      }
    }
    //fill in initial values for the mesh if not given by the user
    if (!data.userSetInitial_) {
      mesh_->vertT.resize(mesh_->vertices.size());
      for (size_t i = 0; i < mesh_->vertices.size(); i++) {
        mesh_->vertT[i] = mesh_->vertices[i][0] -  (mx + mn) / 2.;
      }
    }
    meshFIM FIMPtr(mesh_);
    time_values_ =
      FIMPtr.GenerateData((char*)data.filename_.c_str(), data.numSteps_,
          data.timeStep_, data.insideIterations_, data.sideLengths_,
          data.blockSize_, data.bandwidth_, data.partitionType_,
          data.metisSize_, data.verbose_);

    endtime = clock();
    double duration = (double)(endtime - starttime) * 1000/ CLOCKS_PER_SEC;

    if (data.verbose_)
      printf("Computing time : %.10lf ms\n",duration);
  }

  /**
   * This function uses the provided analytical solutions to
   * visualize the algorithm's error after each iteration.
   *
   * @param solution The vector of expected solutions.
   */
  void printErrorGraph(std::vector<float> solution) {

    // now calculate the RMS error for each iteration
    std::vector<float> rmsError;
    rmsError.resize(numIterations());
    for (size_t i = 0; i < numIterations(); i++) {
      float sum = 0.f;
      std::vector<LevelsetValueType> result = getResultAtIteration(i);
      for (size_t j = 0; j < solution.size(); j++) {
        float err = std::abs(solution[j] - result[j]);
        sum +=  err * err;
      }
      rmsError[i] = std::sqrt(sum / static_cast<float>(solution.size()));
    }
    //determine the log range
    float max_err = rmsError[0];
    float min_err = rmsError[rmsError.size() - 1];
    int max_log = -10, min_log = 10;
    while (std::pow(static_cast<float>(10),max_log) < max_err) max_log++;
    while (std::pow(static_cast<float>(10),min_log) > min_err) min_log--;
    // print the error graph
    printf("\n\nlog(Err)|\n");
    bool printTick = true;
    for(int i = max_log ; i >= min_log; i--) {
      if (printTick) {
        printf("   10^%2d|",i);
      } else {
        printf("        |");
      }
      for (size_t j = 0; j < numIterations(); j++) {
        if (rmsError[j] > std::pow(static_cast<float>(10),i) &&
            rmsError[j] < std::pow(static_cast<float>(10),i+1))
          printf("*");
        else
          printf(" ");
      }
      printf("\n");
      printTick = !printTick;
    }
    printf("--------|------------------------------------------");
    printf("  Converged to: %.4f\n",rmsError[rmsError.size() - 1]);
    printf("        |1   5    10   15   20   25   30   35\n");
    printf("                   Iteration\n");
  }
}

#endif
