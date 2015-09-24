#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <meshFIM2d.h>
#include <math.h>
#include <TriMesh.h>
#include <cuda_runtime.h>
#include <mycutil.h>
using std::string;

void usage(const char *myname)
{
  fprintf(stderr, "Usage: %s infile ntimestep timestep niter side_vert_num block_size bandwidth part_type(0 is metis and 1 is square) metis_size axis domain verbose\n", myname);
  exit(1);
}

int main(int argc, char *argv[])
{
  if(argc != 13)
    usage(argv[0]);

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  int device;
  for(device = 0; device < deviceCount; ++device)
  {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Device %d has compute capability %d.%d.\n",
        device, deviceProp.major, deviceProp.minor);
  }

  cudaSetDevice(0);
  if(cudaDeviceReset() != cudaSuccess)
    exit(0);

  cudaSafeCall((cudaDeviceSetCacheConfig(cudaFuncCachePreferShared)));

  TriMesh* themesh;
  string filename = argv[1];
  filename += ".ply";
  int nsteps = atoi(argv[2]);
  LevelsetValueType timestep = atof(argv[3]);
  int inside_niter = atoi(argv[4]);
  int nside = atoi(argv[5]);
  int block_size = atoi(argv[6]);
  LevelsetValueType bandwidth = atof(argv[7]);
  int part_type =atoi(argv[8]);
  int metis_size = atoi(argv[9]);
  int axis = atoi(argv[10]);
  double domain = atof(argv[11]);
  bool verbose = atoi(argv[12])==1?true:false;
  //  themesh.init(in.pointlist, in.numberofpoints, in.trifacelist, in.numberoffacets, in.tetrahedronlist, in.numberoftetrahedra, in.numberoftetrahedronattributes, in.tetrahedronattributelist);
  //  themesh.reorient();
  themesh = TriMesh::read(filename.c_str());
  //  themesh.rescale(domain_size);
  themesh->need_neighbors();
  themesh->need_adjacentfaces();
  themesh->need_Rinscribe();
  meshFIM2d* FIMPtr = new meshFIM2d(themesh);
  //  FIMPtr->SetMesh(&themesh, 1);
  FIMPtr->GenerateData((char*)filename.c_str(), nsteps,
      timestep, inside_niter, nside, block_size,
      bandwidth, part_type, metis_size, axis, domain, verbose);
  return 0;
}

