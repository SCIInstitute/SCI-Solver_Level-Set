#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <meshFIM.h>
#include <math.h>
#include <tetmesh.h>
#include <tetgen.h>
#include <cuda_runtime.h>
#include <mycutil.h>
#include <mytimer.h>
using std::string;

void usage(const char *myname)
{
  fprintf(stderr, "Usage: %s infile ntimestep timestep niter side_vert_num block_size bandwidth part_type metis_size\n", myname);
  exit(1);
}

int main(int argc, char *argv[])
{
  if(argc != 10)
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

  clock_t starttime, endtime;

  TetMesh themesh;
  char* filename = argv[1];
  tetgenio in, addin, bgmin, out;
  if(!in.load_tetmesh(filename))
  {
    printf("File open failed!!\n");
    exit(0);
  }
  int nsteps = atoi(argv[2]);
	LevelsetValueType timestep = atof(argv[3]);
//	printf("timestep=%f\n", timestep);
	int inside_niter = atoi(argv[4]);
	int nside = atoi(argv[5]);
	int block_size = atoi(argv[6]);
	LevelsetValueType bandwidth = atof(argv[7]);
	int part_type =atoi(argv[8]);
	int metis_size = atoi(argv[9]);
  printf("Narrowband width is %f\n", bandwidth);
	LevelsetValueType domain_size = 16.0;
  themesh.init(in.pointlist, in.numberofpoints, in.trifacelist, in.numberoffacets, in.tetrahedronlist, in.numberoftetrahedra, in.numberoftetrahedronattributes, in.tetrahedronattributelist);
  themesh.reorient();
//  themesh.rescale(domain_size);
  themesh.need_neighbors();
  themesh.need_adjacenttets();
  meshFIM* FIMPtr = new meshFIM(&themesh);
  //  FIMPtr->SetMesh(&themesh, 1);
  FIMPtr->GenerateData(filename, nsteps, timestep, inside_niter, nside, block_size, bandwidth, part_type, metis_size);
  return 0;
}

