#ifndef __LEVELSET_H__
#define __LEVELSET_H__

#include <cstdio>
#include <cstdlib>
#include <meshFIM2d.h>
#include <math.h>
#include <TriMesh.h>
#include <cuda_runtime.h>
#include <time.h>
#include <cutil.h>
#include <cstring>
#include <limits>

/** The class that represents all of the available options for LevelSet */
class LevelSet2d{
public:
  LevelSet2d(std::string fname = "../src/test/test_data/sphere_266verts.ply",
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
    userSetAdvection_(false),
    mesh_(NULL)
  {}
  //helper functions
  std::vector < float >& getResultAtIteration(size_t i) {
    return this->time_values_.at(i);
  }
  size_t numIterations() { return this->time_values_.size(); }
  void writeVTK() {
    meshFIM2d FIMPtr(this->mesh_);
    FIMPtr.writeVTK(this->time_values_);
  }
  void writeFLD() {
    meshFIM2d FIMPtr(this->mesh_);
    FIMPtr.writeFLD();
  }
  //main functions
  //initialize the vertex values
  void initializeVertices(std::vector<float> values) {
    if (this->mesh_ == NULL) {
      std::cerr << "You must initialize the mesh first!" << std::endl;
      exit(0);
    }
    if (values.size() != this->mesh_->vertices.size()) {
      std::cerr << "Initialize values size does not match number of vertices!"
        << std::endl;
      exit(0);
    }
    this->mesh_->vertT.resize(this->mesh_->vertices.size());
    for (size_t i = 0; i < values.size(); i++) {
      this->mesh_->vertT[i] = values[i];
    }
    this->userSetInitial_ = true;
  }
  //initialize the element advection
  void initializeAdvection(std::vector<point> values) {
    if (this->mesh_ == NULL) {
      std::cerr << "You must initialize the mesh first!" << std::endl;
      exit(0);
    }
    if (values.size() != this->mesh_->faces.size()) {
      std::cerr << "Initialize values size does not match number of elements!"
        << std::endl;
      exit(0);
    }
    this->mesh_->normals.resize(this->mesh_->faces.size());
    for (size_t i = 0; i < values.size(); i++) {
      this->mesh_->normals[i] = values[i];
    }
    this->userSetAdvection_ = true;
  }

  void initializeMesh() {
    if (this->mesh_ == NULL) {
      this->mesh_ = TriMesh::read(this->filename_.c_str(), this->verbose_);
      if (this->mesh_ == NULL)
      {
        printf("File open failed!!\n");
        exit(0);
      }
      this->mesh_->need_neighbors(this->verbose_);
      this->mesh_->need_adjacentfaces(this->verbose_);
      this->mesh_->need_Rinscribe();
    }
  }

  /**
  * Creates the mesh, partitions the mesh, and runs the algorithm.
  *
  * @data The set of options for the LevelSet algorithm.
  *       The defaults are used if nothing is provided.
  */
  void solveLevelSet() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device;
    for (device = 0; device < deviceCount; ++device)
    {
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, device);
      if (this->verbose_)
        printf("Device %d has compute capability %d.%d.\n",
        device, deviceProp.major, deviceProp.minor);
    }
    device = 0;

    cudaSetDevice(device);
    if (cudaDeviceReset() != cudaSuccess) {
      std::cout << "There is a problem with CUDA device " << device << std::endl;
      exit(0);
    }

    cudaSafeCall((cudaDeviceSetCacheConfig(cudaFuncCachePreferShared)));
    if (this->verbose_)
      printf("Narrowband width is %f\n", this->bandwidth_);
    clock_t starttime, endtime;
    starttime = clock();

    if (this->mesh_ == NULL) {
      initializeMesh();
    }
    float mn = std::numeric_limits<float>::max();
    float mx = std::numeric_limits<float>::min();
    for (size_t i = 0; i < mesh_->vertices.size(); i++) {
      mn = std::min(mn, static_cast<float>(mesh_->vertices[i][0]));
      mx = std::max(mx, static_cast<float>(mesh_->vertices[i][0]));
    }
    //populate advection if it's empty
    if (!this->userSetAdvection_) {
      this->mesh_->normals.resize(this->mesh_->faces.size());
      for (size_t i = 0; i < this->mesh_->faces.size(); i++) {
        this->mesh_->normals[i] = point((mx - mn) / 40., 0, 0);
      }
    }
    //fill in initial values for the mesh if not given by the user
    if (!this->userSetInitial_) {
      this->mesh_->vertT.resize(this->mesh_->vertices.size());
      for (size_t i = 0; i < this->mesh_->vertices.size(); i++) {
        this->mesh_->vertT[i] = -(this->mesh_->vertices[i][0] - mn) * 40. / (mx - mn) + 1.;
      }
    }
    meshFIM2d FIMPtr(this->mesh_);
    this->time_values_ =
      FIMPtr.GenerateData((char*)this->filename_.c_str(), this->numSteps_,
      this->timeStep_, this->insideIterations_, this->sideLengths_,
      this->blockSize_, this->bandwidth_, this->partitionType_,
      this->metisSize_, this->verbose_);

    endtime = clock();
    double duration = (double)(endtime - starttime) * 1000 / CLOCKS_PER_SEC;

    if (this->verbose_)
      printf("Computing time : %.10lf ms\n", duration);
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
    rmsError.resize(this->numIterations());
    for (size_t i = 0; i < this->numIterations(); i++) {
      float sum = 0.f;
      std::vector<float> result = this->getResultAtIteration(i);
      for (size_t j = 0; j < solution.size(); j++) {
        float err = std::abs(solution[j] - result[j]);
        sum += err * err;
      }
      rmsError[i] = std::sqrt(sum / static_cast<float>(solution.size()));
    }
    //determine the log range
    float max_err = rmsError[0];
    float min_err = rmsError[rmsError.size() - 1];
    int max_log = -10, min_log = 10;
    while (std::pow(static_cast<float>(10), max_log) < max_err) max_log++;
    while (std::pow(static_cast<float>(10), min_log) > min_err) min_log--;
    // print the error graph
    printf("\n\nlog(Err)|\n");
    bool printTick = true;
    for (int i = max_log; i >= min_log; i--) {
      if (printTick) {
        printf("   10^%2d|", i);
      } else {
        printf("        |");
      }
      for (size_t j = 0; j < this->numIterations(); j++) {
        if (rmsError[j] > std::pow(static_cast<float>(10), i) &&
          rmsError[j] < std::pow(static_cast<float>(10), i + 1))
          printf("*");
        else
          printf(" ");
      }
      printf("\n");
      printTick = !printTick;
    }
    printf("--------|------------------------------------------");
    printf("  Converged to: %.4f\n", rmsError[rmsError.size() - 1]);
    printf("        |1   5    10   15   20   25   30   35\n");
    printf("                   Iteration\n");
  }
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
  TriMesh* mesh_;
  std::vector < std::vector <float> > time_values_;
};
#endif
