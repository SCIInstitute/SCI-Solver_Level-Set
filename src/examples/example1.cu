#include <LevelSet3d.h>

int main(int argc, char *argv[])
{
  LevelSet3d::LevelSet3d data;
  bool fromCenter = false;
  //input filename (minus extension)
  std::string filename;
  for (int i = 0; i < argc; i++) {
    if (strcmp(argv[i],"-v") == 0) {
      data.verbose_ = true;
    } else if (strcmp(argv[i],"-i") == 0) {
      if (i+1 >= argc) break;
      data.filename_ = std::string(argv[i+1]);
      if (data.filename_.substr(data.filename_.size()-5,5) == ".node")
        data.filename_ = data.filename_.substr(0,data.filename_.size() - 5);
      if (data.filename_.substr(data.filename_.size()-4,4) == ".ele")
        data.filename_ = data.filename_.substr(0,data.filename_.size() - 4);
      i++;
    } else if (strcmp(argv[i],"-n") == 0) {
      if (i+1 >= argc) break;
      data.numSteps_ = atoi(argv[i+1]);
      i++;
    } else if (strcmp(argv[i],"-t") == 0) {
      if (i+1 >= argc) break;
      data.timeStep_ = atof(argv[i+1]);
      i++;
    } else if (strcmp(argv[i],"-s") == 0) {
      if (i+1 >= argc) break;
      data.insideIterations_ = atoi(argv[i+1]);
      i++;
    } else if (strcmp(argv[i],"-d") == 0) {
      if (i+1 >= argc) break;
      data.sideLengths_ = atoi(argv[i+1]);
      i++;
    } else if (strcmp(argv[i],"-p") == 0) {
      if (i+1 >= argc) break;
      data.partitionType_ = atoi(argv[i+1]);
      i++;
    } else if (strcmp(argv[i],"-m") == 0) {
      if (i+1 >= argc) break;
      data.metisSize_ = atoi(argv[i+1]);
      i++;
    } else if (strcmp(argv[i],"-b") == 0) {
      if (i+1 >= argc) break;
      data.blockSize_ = atoi(argv[i+1]);
      i++;
    } else if (strcmp(argv[i],"-w") == 0) {
      if (i+1 >= argc) break;
      data.bandwidth_ = atof(argv[i+1]);
      i++;
    } else if (strcmp(argv[i],"-c") == 0) {
      fromCenter = true;
    } else if (strcmp(argv[i],"-h") == 0) {
      std::cout << "Usage: ./Example1 [OPTIONS]" << std::endl;
      std::cout << "   -h                 Print this help message." << std::endl;
      std::cout << "   -v                 Print verbose runtime information." << std::endl;
      std::cout << "   -i FILENAME        Use this input tet mesh (node/ele)." << std::endl;
      std::cout << "   -n NSTEPS          # of steps to take of TIMESTEP amount." << std::endl;
      std::cout << "   -t TIMESTEP        Duration of a timestep." << std::endl;
      std::cout << "   -s INSIDE_NITER    # of inside iterations." << std::endl;
      std::cout << "   -d NSIDE           # of sides for Square partition type." << std::endl;
      std::cout << "   -p PARTITION_TYPE  1 for Square, otherwise is it METIS." << std::endl;
      std::cout << "   -b NUM_BLOCKS      # of blocks for Square partition type." << std::endl;
      std::cout << "   -m METIS_SIZE      The size for METIS partiation type." << std::endl;
      std::cout << "   -w BANDWIDTH       The Bandwidth for the algorithm." << std::endl;
      std::cout << "   -c                 Initialize data to move away from center." << std::endl;
      exit(0);
    }
  }
  if (fromCenter) {
    //find the center, max from center
    LevelSet3d::initializeMesh(data);
    point center(0, 0, 0);
    for (size_t i = 0; i < LevelSet3d::mesh_->vertices.size(); i++) {
      center = center + LevelSet3d::mesh_->vertices[i];
    }
    center = center / static_cast<double>(LevelSet3d::mesh_->vertices.size());
    double max = 0.;
    for (size_t i = 0; i < LevelSet3d::mesh_->vertices.size(); i++) {
      point p = LevelSet3d::mesh_->vertices[i] - center;
      double mag = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
      max = std::max(max, mag);
    }
    //initialize values of verts
    std::vector<double> vals;
    for (size_t i = 0; i < LevelSet3d::mesh_->vertices.size(); i++) {
      point p = LevelSet3d::mesh_->vertices[i] - center;
      double mag = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
      vals.push_back(mag - max / 2.);
    }
    //initialize advection to be away from the center.
    std::vector<point> adv;
    for (size_t i = 0; i < LevelSet3d::mesh_->tets.size(); i++) {
      point p = (LevelSet3d::mesh_->vertices[LevelSet3d::mesh_->tets[i][0]] +
        LevelSet3d::mesh_->vertices[LevelSet3d::mesh_->tets[i][1]] +
        LevelSet3d::mesh_->vertices[LevelSet3d::mesh_->tets[i][2]])
        / 3. - center;
      double mag = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
      adv.push_back(p / mag / mag);
    }
    LevelSet3d::initializeVertices(data, vals);
    LevelSet3d::initializeAdvection(data, adv);
  }
  LevelSet3d::solveLevelSet(data);
  LevelSet3d::writeVTK();
  return 0;
}

