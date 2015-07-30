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
  //input filename (minus extension)
  std::string filename;
  for (int i = 0; i < argc; i++)
    if (strcmp(argv[i],"-v") == 0) {
      verbose = true;
    } else if (strcmp(argv[i],"-i") == 0) {
      if (i+1 >= argc) break;
      filename = std::string(argv[i+1]);
      i++;
    }
  if (filename.empty())
    filename = "../example_data/aorta";

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
  if(!in.load_tetmesh((char*)filename.c_str()))
  {
    printf("File open failed!!\n");
    exit(0);
  }
  int nsteps = 10;//atoi(argv[2]);
  LevelsetValueType timestep = 1;//atof(argv[3]);
  int inside_niter = 10;//atoi(argv[4]);
  int nside = 1;//atoi(argv[5]);
  int block_size = 16;//atoi(argv[6]);
  LevelsetValueType bandwidth = 16.;//atof(argv[7]);
  int part_type = 0;//atoi(argv[8]);
  int metis_size = 16;//atoi(argv[9]);
  if (verbose)
    printf("Narrowband width is %f\n", bandwidth);
  clock_t starttime, endtime;
  starttime = clock();

  themesh.init(in.pointlist, in.numberofpoints, in.trifacelist, in.numberoffacets, in.tetrahedronlist, in.numberoftetrahedra, in.numberoftetrahedronattributes, in.tetrahedronattributelist);
  themesh.reorient();
  themesh.need_neighbors();
  themesh.need_adjacenttets();
  meshFIM* FIMPtr = new meshFIM(&themesh);
  FIMPtr->GenerateData((char*)filename.c_str(), nsteps, timestep, inside_niter, nside, block_size, bandwidth, part_type, metis_size);

  endtime = clock();
  double duration = (double)(endtime - starttime) * 1000/ CLOCKS_PER_SEC;

  if (verbose)
    printf("Computing time : %.10lf ms\n",duration);

  delete FIMPtr;
  return 0;
}

