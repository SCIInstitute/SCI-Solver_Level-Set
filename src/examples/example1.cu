#include <LevelSet.h>
#include <cmath>

int main(int argc, char *argv[])
{
  LevelSet data(false,"../src/test/test_data/sphere334",false);
  std::string type = "x";
  //input filename (minus extension)
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
    } else if (strcmp(argv[i], "-y") == 0) {
      if (i + 1 >= argc) break;
      type = std::string(argv[++i]);
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
      std::cout << "   -y EXAMPLE_TYPE    Example type: 'center', 'revolve', 'x'" << std::endl;
      exit(0);
    }
  }
  if (type == "center"  || type == "revolve") {
    //find the center, max from center
    data.initializeMesh();
    point center(0, 0, 0);
    for (size_t i = 0; i < data.tetMesh_->vertices.size(); i++) {
      center = center + data.tetMesh_->vertices[i];
    }
    center = center / static_cast<float>(data.tetMesh_->vertices.size());
    float max = 0.;
    for (size_t i = 0; i < data.tetMesh_->vertices.size(); i++) {
      point p = data.tetMesh_->vertices[i] - center;
      float mag = len(p);
      max = std::max(max, mag);
    }
    //initialize values of verts
    std::vector<float> vals;
    for (size_t i = 0; i < data.tetMesh_->vertices.size(); i++) {
      point p = data.tetMesh_->vertices[i] - center;
      double mag = len(p);
      if (type == "revolve") {
        //get the angle with (+/-1,0,0)
        float val = p[0];
        if (val < 0.) val *= -1.;
        float theta = std::acos(val / std::sqrt(p[0] * p[0] + p[1] * p[1]));
        if (p[1] < 0.f) theta *= -1.f;
        vals.push_back(10.f * theta);
      } else {
        vals.push_back(mag - max / 2.);
      }
    }
    //initialize advection to be away from the center.
    std::vector<point> adv;
    for (size_t i = 0; i < data.tetMesh_->tets.size(); i++) {
      point p = (data.tetMesh_->vertices[data.tetMesh_->tets[i][0]] +
        data.tetMesh_->vertices[data.tetMesh_->tets[i][1]] +
        data.tetMesh_->vertices[data.tetMesh_->tets[i][2]] +
        data.tetMesh_->vertices[data.tetMesh_->tets[i][3]])
        / 4.f - center;
      float mag = len(p);
      mag /= max / 20.f;
      if (type == "revolve") {
        //only care about XY plane angle
        //get the tangent to the central circle
        point p2 = p;
        p2[2] = 0.f;
        point p3 = p2 CROSS point(0, 0, 1);
        adv.push_back(p3 * len(p2) / (100.f * len (p3)));
      } else {
        adv.push_back(p / mag / mag);
      }
    }
    data.initializeVertices(vals);
    data.initializeAdvection(adv);
  }
  data.solveLevelSet();
  data.writeVTK();
  return 0;
}

