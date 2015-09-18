/*
Szymon Rusinkiewicz
Princeton University

mesh_view.cc
Simple viewer
 */

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <meshFIM2d.h>
#include <math.h>
#include <TriMesh.h>
#include <timer.h>

using std::string;

void usage(const char *myname)
{
  fprintf(stderr, "Usage: %s infile ntimestep nside\n", myname);
  exit(1);
}

int main(int argc, char *argv[])
{
  if(argc != 4)
    usage(argv[0]);

  clock_t starttime, endtime;
  TriMesh* themesh;
  char* filename = argv[1];
	int nsteps = atoi(argv[2]);
	int nside  = atoi(argv[3]);
  themesh = TriMesh::read(filename);
  if(!themesh)
    exit(1);

  meshFIM2d FIMPtr;
  FIMPtr.SetMesh(themesh, 1);
  FIMPtr.GenerateData(filename, nsteps, nside);
  return 0;
}

