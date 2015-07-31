#include <stdio.h>
#include <stdlib.h>
#include <meshFIM.h>
#include <math.h>
#include <tetmesh.h>
#include <tetgen.h>
#include <cuda_runtime.h>
#include <time.h>
#include <mycutil.h>
#include <cstring>

int main(int argc, char *argv[])
{
  //Verbose option
  bool verbose = false;
  int nsteps = 10;//atoi(argv[2]);
  LevelsetValueType timestep = 1;//atof(argv[3]);
  int inside_niter = 10;//atoi(argv[4]);
  int nside = 1;//atoi(argv[5]);
  int block_size = 16;//atoi(argv[6]);
  LevelsetValueType bandwidth = 16.;//atof(argv[7]);
  int part_type = 0;//atoi(argv[8]);
  int metis_size = 16;//atoi(argv[9]);
  //input filename (minus extension)
  std::string filename;
  for (int i = 0; i < argc; i++)
    if (strcmp(argv[i],"-v") == 0) {
      verbose = true;
    } else if (strcmp(argv[i],"-i") == 0) {
      if (i+1 >= argc) break;
      filename = std::string(argv[i+1]);
      if (filename.substr(filename.size()-5,5) == ".node") 
        filename = filename.substr(0,filename.size() - 5);
      if (filename.substr(filename.size()-4,4) == ".ele") 
        filename = filename.substr(0,filename.size() - 4);
        std::cout << filename << std::endl;
      i++;
    } else if (strcmp(argv[i],"-n") == 0) {
      if (i+1 >= argc) break;
      nsteps = atoi(argv[i+1]);
      i++;
    } else if (strcmp(argv[i],"-t") == 0) {
      if (i+1 >= argc) break;
      timestep = atof(argv[i+1]);
      i++;
    } else if (strcmp(argv[i],"-s") == 0) {
      if (i+1 >= argc) break;
      inside_niter = atoi(argv[i+1]);
      i++;
    } else if (strcmp(argv[i],"-d") == 0) {
      if (i+1 >= argc) break;
      nside = atoi(argv[i+1]);
      i++;
    } else if (strcmp(argv[i],"-p") == 0) {
      if (i+1 >= argc) break;
      part_type = atoi(argv[i+1]);
      i++;
    } else if (strcmp(argv[i],"-m") == 0) {
      if (i+1 >= argc) break;
      metis_size = atoi(argv[i+1]);
      i++;
    } else if (strcmp(argv[i],"-b") == 0) {
      if (i+1 >= argc) break;
      block_size = atoi(argv[i+1]);
      i++;
    } else if (strcmp(argv[i],"-w") == 0) {
      if (i+1 >= argc) break;
      bandwidth = atof(argv[i+1]);
      i++;
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
      exit(0);
    }
  if (filename.empty())
    filename = "../example_data/cube_unstruc_size256_s2";

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  int device;
  for(device = 0; device < deviceCount; ++device)
  {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    if (verbose)
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

  TetMesh themesh;
  tetgenio in, addin, bgmin, out;
  if(!in.load_tetmesh((char*)filename.c_str(),verbose))
  {
    printf("File open failed!!\n");
    exit(0);
  }
  if (verbose)
    printf("Narrowband width is %f\n", bandwidth);
  clock_t starttime, endtime;
  starttime = clock();

  themesh.init(in.pointlist, in.numberofpoints, in.trifacelist, in.numberoffacets, in.tetrahedronlist, in.numberoftetrahedra, in.numberoftetrahedronattributes, in.tetrahedronattributelist, verbose);
  themesh.reorient();
  themesh.need_neighbors(verbose);
  themesh.need_adjacenttets(verbose);
  meshFIM* FIMPtr = new meshFIM(&themesh);
  FIMPtr->GenerateData((char*)filename.c_str(), nsteps, timestep, inside_niter, nside, block_size, bandwidth, part_type, metis_size, verbose);

  endtime = clock();
  double duration = (double)(endtime - starttime) * 1000/ CLOCKS_PER_SEC;

  if (verbose)
    printf("Computing time : %.10lf ms\n",duration);

  delete FIMPtr;
  return 0;
}

