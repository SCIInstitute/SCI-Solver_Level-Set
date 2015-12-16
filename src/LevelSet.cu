#include "LevelSet.h"
#include <limits>

LevelSet::LevelSet(bool isTriMesh, std::string fname, bool verbose) :
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
  triMesh_(NULL),
  tetMesh_(NULL),
  FIMPtr2d_(NULL),
  FIMPtr3d_(NULL),
  isTriMesh_(isTriMesh)
{}

LevelSet::~LevelSet() {
  if (this->tetMesh_ != NULL)
    delete this->tetMesh_;
  if (this->triMesh_ != NULL)
    delete this->triMesh_;
  if (this->FIMPtr2d_ != NULL)
    delete this->FIMPtr2d_;
  if (this->FIMPtr3d_ != NULL)
    delete this->FIMPtr3d_;
}

std::vector < float >& LevelSet::getResultAtIteration(size_t i) {
  return this->time_values_.at(i);
}
size_t LevelSet::numIterations() { return this->time_values_.size(); }
void LevelSet::writeVTK() {
  if (FIMPtr2d_ != NULL)
    FIMPtr2d_->writeVTK(this->time_values_);
  else
    FIMPtr3d_->writeVTK(this->time_values_);
}
void LevelSet::writeFLD() {
  if (FIMPtr2d_ != NULL)
    FIMPtr2d_->writeFLD();
  else
    FIMPtr3d_->writeFLD();
}

void LevelSet::initializeVertices(std::vector<float> values) {
  if (this->triMesh_ == NULL && this->tetMesh_ == NULL) {
    std::cerr << "You must initialize the mesh first!" << std::endl;
    exit(0);
  }
  if (this->triMesh_ != NULL) {
    if (values.size() != this->triMesh_->vertices.size()) {
      std::cerr << "Initialize values size does not match number of vertices!"
        << std::endl;
      exit(0);
    }
    this->triMesh_->vertT.resize(this->triMesh_->vertices.size());
    for (size_t i = 0; i < values.size(); i++) {
      this->triMesh_->vertT[i] = values[i];
    }
  } else {
    if (values.size() != this->tetMesh_->vertices.size()) {
      std::cerr << "Initialize values size does not match number of vertices!"
        << std::endl;
      exit(0);
    }
    this->tetMesh_->vertT.resize(this->tetMesh_->vertices.size());
    for (size_t i = 0; i < values.size(); i++) {
      this->tetMesh_->vertT[i] = values[i];
    }

  }
  this->userSetInitial_ = true;
}

void LevelSet::initializeAdvection(std::vector<point> values) {
  if (this->triMesh_ == NULL && this->tetMesh_ == NULL) {
    std::cerr << "You must initialize the mesh first!" << std::endl;
    exit(0);
  }
  if (this->triMesh_ != NULL) {
    if (values.size() != this->triMesh_->faces.size()) {
      std::cerr << "Initialize values size does not match number of elements!"
        << std::endl;
      exit(0);
    }
    this->triMesh_->normals.resize(this->triMesh_->faces.size());
    for (size_t i = 0; i < values.size(); i++) {
      this->triMesh_->normals[i] = values[i];
    }
  } else {
    if (values.size() != this->tetMesh_->tets.size()) {
      std::cerr << "Initialize values size does not match number of elements!"
        << std::endl;
      exit(0);
    }
    this->tetMesh_->normals.resize(this->tetMesh_->tets.size());
    for (size_t i = 0; i < values.size(); i++) {
      this->tetMesh_->normals[i] = values[i];
    }
  }
  this->userSetAdvection_ = true;
}

void LevelSet::initializeMesh() {
  if (this->isTriMesh_) {
    if (this->triMesh_ == NULL) {
      this->triMesh_ = TriMesh::read(this->filename_.c_str(), this->verbose_);
      if (this->triMesh_ == NULL)
      {
        printf("File open failed!!\n");
        exit(0);
      }
      this->triMesh_->need_neighbors(this->verbose_);
      this->triMesh_->need_adjacentfaces(this->verbose_);
      this->triMesh_->need_Rinscribe();
    }
  } else {
    if (this->tetMesh_ == NULL) {
      this->tetMesh_ = new TetMesh();
      tetgenio in, addin, bgmin, out;
      if (!in.load_tetmesh((char*)this->filename_.c_str(), this->verbose_))
      {
        printf("File open failed!!\n");
        exit(0);
      }

      this->tetMesh_->init(in.pointlist, in.numberofpoints, in.trifacelist,
          in.numberoffacets, in.tetrahedronlist, in.numberoftetrahedra,
          in.numberoftetrahedronattributes, in.tetrahedronattributelist,
          this->verbose_);
      this->tetMesh_->reorient();
      this->tetMesh_->need_neighbors(this->verbose_);
      this->tetMesh_->need_adjacenttets(this->verbose_);
    }
  }
}

void LevelSet::solveLevelSet() {
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

  if ((this->triMesh_ == NULL && this->isTriMesh_) ||
      (this->tetMesh_ == NULL && !this->isTriMesh_)) {
    initializeMesh();
  }
  double mn = std::numeric_limits<double>::max();
  double mx = (mn - 1.) * -1.;
  if (this->isTriMesh_) {
    for (size_t i = 0; i < this->triMesh_->vertices.size(); i++) {
      mn = std::min(mn, static_cast<double>(this->triMesh_->vertices[i][0]));
      mx = std::max(mx, static_cast<double>(this->triMesh_->vertices[i][0]));
    }
    double midX = (mn + mx) / 2.;
    double rangeX = mx - mn;
    //populate advection if it's empty
    if (!this->userSetAdvection_) {
      this->triMesh_->normals.resize(this->triMesh_->faces.size());
      for (size_t i = 0; i < this->triMesh_->faces.size(); i++) {
        this->triMesh_->normals[i] = point((mx - mn) / 40., 0, 0);
      }
    }
    //fill in initial values for the mesh if not given by the user
    if (!this->userSetInitial_) {
      this->triMesh_->vertT.resize(this->triMesh_->vertices.size());
      for (size_t i = 0; i < this->triMesh_->vertices.size(); i++) {
        this->triMesh_->vertT[i] =
          (this->triMesh_->vertices[i][0] - midX) / rangeX;
      }
    }
    if (FIMPtr2d_ != NULL) delete FIMPtr2d_;
    FIMPtr2d_ = new meshFIM2d(this->triMesh_);
    this->time_values_ =
      FIMPtr2d_->GenerateData((char*)this->filename_.c_str(), this->numSteps_,
          this->timeStep_, this->insideIterations_, this->sideLengths_,
          this->blockSize_, this->bandwidth_, this->partitionType_,
          this->metisSize_, this->verbose_);

    endtime = clock();
    double duration = (double)(endtime - starttime) * 1000 / CLOCKS_PER_SEC;

    if (this->verbose_)
      printf("Computing time : %.10lf ms\n", duration);
  } else {
    for (size_t i = 0; i < this->tetMesh_->vertices.size(); i++) {
      mn = std::min(mn, static_cast<double>(this->tetMesh_->vertices[i][0]));
      mx = std::max(mx, static_cast<double>(this->tetMesh_->vertices[i][0]));
    }
    double midX = (mn + mx) / 2.;
    double rangeX = mx - mn;
    //populate advection if it's empty
    if (!this->userSetAdvection_) {
      this->tetMesh_->normals.resize(this->tetMesh_->tets.size());
      for (size_t i = 0; i < this->tetMesh_->tets.size(); i++) {
        this->tetMesh_->normals[i] = point((mx - mn) / 100., 0, 0);
      }
    }
    //fill in initial values for the mesh if not given by the user
    if (!this->userSetInitial_) {
      this->tetMesh_->vertT.resize(this->tetMesh_->vertices.size());
      for (size_t i = 0; i < this->tetMesh_->vertices.size(); i++) {
        this->tetMesh_->vertT[i] = (this->tetMesh_->vertices[i][0] - midX) / rangeX;
      }
    }
    if (FIMPtr3d_ != NULL) delete FIMPtr3d_;
    FIMPtr3d_ = new meshFIM3d(this->tetMesh_);
    time_values_ =
      FIMPtr3d_->GenerateData((char*)this->filename_.c_str(), this->numSteps_,
          this->timeStep_, this->insideIterations_, this->sideLengths_,
          this->blockSize_, this->bandwidth_, this->partitionType_,
          this->metisSize_, this->verbose_);

  }
}

void LevelSet::printErrorGraph(std::vector<float> solution) {

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
