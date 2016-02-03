#include <LevelSet.h>

int main(int argc, char *argv[])
{
  LevelSet data(true);
  std::string type = "x";
  float isovalue = 0.;
  //input filename (minus extension)
  std::string filename;
  for (int i = 0; i < argc; i++) {
    if (strcmp(argv[i], "-v") == 0) {
      data.verbose_ = true;
    }
    else if (strcmp(argv[i], "-i") == 0) {
      if (i + 1 >= argc) break;
      data.filename_ = std::string(argv[i + 1]);
      i++;
    }
    else if (strcmp(argv[i], "-n") == 0) {
      if (i + 1 >= argc) break;
      data.numSteps_ = atoi(argv[i + 1]);
      i++;
    }
    else if (strcmp(argv[i], "-t") == 0) {
      if (i + 1 >= argc) break;
      data.timeStep_ = atof(argv[i + 1]);
      i++;
    } else if (strcmp(argv[i], "-o") == 0) {
      if (i + 1 >= argc) break;
      isovalue = atof(argv[i + 1]);
      i++;
    }
    else if (strcmp(argv[i], "-s") == 0) {
      if (i + 1 >= argc) break;
      data.insideIterations_ = atoi(argv[i + 1]);
      i++;
    }
    else if (strcmp(argv[i], "-d") == 0) {
      if (i + 1 >= argc) break;
      data.sideLengths_ = atoi(argv[i + 1]);
      i++;
    }
    else if (strcmp(argv[i], "-p") == 0) {
      if (i + 1 >= argc) break;
      data.partitionType_ = atoi(argv[i + 1]);
      i++;
    }
    else if (strcmp(argv[i], "-m") == 0) {
      if (i + 1 >= argc) break;
      data.metisSize_ = atoi(argv[i + 1]);
      i++;
    }
    else if (strcmp(argv[i], "-b") == 0) {
      if (i + 1 >= argc) break;
      data.blockSize_ = atoi(argv[i + 1]);
      i++;
    }
    else if (strcmp(argv[i], "-w") == 0) {
      if (i + 1 >= argc) break;
      data.bandwidth_ = atof(argv[i + 1]);
      i++;
    } else if (strcmp(argv[i], "-y") == 0) {
      if (i + 1 >= argc) break;
      type = std::string(argv[++i]);
    } else if (strcmp(argv[i], "-h") == 0) {
      std::cout << "Usage: ./Example2 [OPTIONS]" << std::endl;
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
      std::cout << "   -y EXAMPLE_TYPE    Example type: 'revolve', 'x', 'center, 'curvature'" << std::endl;
      std::cout << "   -o ISOVALUE        The isovalue for curvature." << std::endl;
      exit(0);
    }
  }
  if (type == "center" || type == "revolve" || type == "curvature") {
    //find the center, max from center
    data.initializeMesh();
    point center(0, 0, 0);
    for (size_t i = 0; i < data.triMesh_->vertices.size(); i++) {
      center = center + data.triMesh_->vertices[i];
    }
    center = center / static_cast<float>(data.triMesh_->vertices.size());
    float max = 0.;
    for (size_t i = 0; i < data.triMesh_->vertices.size(); i++) {
      point p = data.triMesh_->vertices[i] - center;
      float mag = len(p);
      max = std::max(max, mag);
    }
    //initialize values of verts
    std::vector<float> vals;
    for (size_t i = 0; i < data.triMesh_->vertices.size(); i++) {
      point p = data.triMesh_->vertices[i] - center;
      if (type == "center") {
        float mag = len(p);
        vals.push_back(mag - max / 2.);
      } else {
        //get the angle with (+/-1,0,0)
        float val = p[0];
        if (val < 0.) val *= -1.;
        float theta = std::acos(val / std::sqrt(p[0] * p[0] + p[1] * p[1]));
        if (p[1] < 0.f) {
          theta *= -1.f;
        }
        if (type == "revolve") {
          vals.push_back(10.f * theta);
        } else {
          vals.push_back(std::max(std::abs(p[0]),
            std::max(std::abs(p[1]), std::abs(p[2]))) - isovalue);
        }
      }
    }
    //initialize advection to be away from the center.
    std::vector<point> adv;
    for (size_t i = 0; i < data.triMesh_->faces.size(); i++) {
      point p = (data.triMesh_->vertices[data.triMesh_->faces[i][0]] +
        data.triMesh_->vertices[data.triMesh_->faces[i][1]] +
        data.triMesh_->vertices[data.triMesh_->faces[i][2]])
        / 3.f;
      if (type == "center") {
        point pt = p - center;
        float mag = len(pt);
        mag /= max /10.f;
        adv.push_back(pt / mag);
      } else {
        //get the tangent to the central circle
        point p2 = p CROSS point(0, 0, 1);
        if (type == "revolve") {
          adv.push_back(p2 * len(p) / 100.f);
        } else {
          adv.push_back(point(0.f, 0.f, 0.f));
        }
      }
    }
    data.initializeVertices(vals);
    data.initializeAdvection(adv);
  }
  data.solveLevelSet();
  data.writeVTK();
  return 0;
}

