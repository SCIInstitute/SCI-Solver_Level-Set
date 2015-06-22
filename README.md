SCI-Solver_Level-Set
====================

**Currently in pre-alpha stage, estimated stable release: October 2015**

SCI-Solver_Level-Set is a C++/CUDA library written to solve a system of level set equations on unstructured meshes. It is designed to solve the equations quickly by using GPU hardware.

The code was written by Zhisong Fu. The theory behind this code is published in the paper: "Fast Parallel Solver for the Levelset Equations on Unstructured Meshes"

**AUTHORS:** Zhisong Fu(*a*,*b*), Sergey Yakovlev(*b*), Robert M. Kirby(*a*,*b*), Ross T. Whitaker(*a*,*b*)

  - School of Computing, University of Utah, Salt Lake City, UT, USA

  - Scientific Computing and Imaging Institute, University of Utah, Salt Lake City, USA

**ABSTRACT:**
Levelset method is a numerical technique for tracking interfaces and shapes. It is actively used within various areas such as physics, chemistry, fluid mechanics, computer vision and microchip fabrication to name a few. Applying the levelset method entails solving the levelset partial differential equation. This paper presents a parallel algorithm for solving the levelset equation on fully unstructured 2D or 3D meshes or manifolds. By taking into account constraints and capabilities of two different computing platforms, the method is suitable for both the coarse-grained parallelism found on CPU-based systems and the fine-grained parallelism of modern massively-SIMD architectures such as graphics processors. In order to solve levelset equation efficiently, we combine the narrowband scheme with domain decomposition: the narrowband fast iterative method (nbFIM) to compute the distance transform by solving an eikonal equation and the patched narrowband (patchNB) scheme to evolve the embedding are proposed in this paper. We also introduce the Hybrid Gathering parallelism strategy to enable regular and lock-free computations in both the nbFIM and patchNB. Finally, we provide the detailed description of the implementation and data structures for the proposed strategies, as well as the performance data for both CPU and GPU implementations.

Requirements
==============

* Git, CMake (3.0+ recommended), and the standard system build environment tools.
* You will need a CUDA Compatible Graphics card. See <a href="https://developer.nvidia.com/cuda-gpus">here</a> You will also need to be sure your card has CUDA compute capability of at least 2.0.
* SCI-Solver_Eikonal is compatible with the latest CUDA toolkit (7.0). Download <a href="https://developer.nvidia.com/cuda-downloads">here</a>.
* This project has been tested on OpenSuse 13.1 (Bottle) on NVidia GeForce GTX 570 HD, Windows 7 on NVidia GeForce GTX 775M, and OSX 10.10 on NVidia GeForce GTX 775M.
* If you have a CUDA compatible card with the above operating systems, and are experiencing issues, please contact the repository owners.
* Windows: You will need Microsoft Visual Studio 2010+ build tools. This document describes the "NMake" process.
* OSX: Please be sure to follow setup for CUDA <a href="http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-mac-os-x/#axzz3W4nXNNin">here</a>. There are several compatability requirements for different MAC machines, including using a different version of CUDA (ie. 5.5).

Building
==============

<h3>Unix / OSX</h3>
In a terminal:
```c++
mkdir SCI-SOLVER_Level-Set/build
cd SCI-SOLVER_Level-Set/build
cmake ../src
make
```

<h3>Windows</h3>
Open a Visual Studio (32 or 64 bit) Native Tools Command Prompt.
Follow these commands:
```c++
mkdir C:\Path\To\SCI-Solver_Level-Set\build
cd C:\Path\To\SCI-Solver_Level-Set\build
cmake -G "NMake Makefiles" ..\src
nmake
```

**Note:** For all platforms, you may need to specify your CUDA toolkit location (especially if you have multiple CUDA versions installed):
  ```c++
     cmake -DCUDA_TOOLKIT_ROOT_DIR="~/NVIDIA/CUDA-7.0" ../src
     ```
     (Assuming this is the location).

     **Note:** If you have compile errors such as <code>undefined reference: atomicAdd</code>, it is likely you need to set your compute capability manually. CMake outputs whether compute capability was determined automatically, or if you need to set it manually. The default (and known working) minimum compute capability is 2.0.

     ```c++
     cmake -DCUDA_COMPUTE_CAPABILITY=20 ../src
     make
     ```


     Running Examples
     ==============

     You will need to enable examples in your build to compile and run them

     ```c++
     cmake -DBUILD_EXAMPLES=ON ../src
     make
     ```

     You will find the example binaries built in the <code>build/examples</code> directory.

     Run the examples in the build directory:

     ```c++
     examples/Example1
     examples/Example2
     ...
     ```

     Follow the example source code in <code>src/examples</code> to learn how to use the library.

     Using the Library
     ==============

     A basic usage of the library links to the <code>libLEVELSET_CORE</code> library during build and includes the headers needed, which are usually no more than:

     ```c++
#include "meshFIM.h"
     ```
     TODO?:

     Then a program would setup the FEM parameters using the
     <code>AMG_Config</code> object and call <code>setup_solver()</code> to generate
     the answer matrices.

     You will need to make sure your CMake/Makfile/Build setup knows where to point for the library and header files. See the examples and their CMakeLists.txt.

     Testing
     ==============
     The repo comes with a set of regression tests to see if recent changes break expected results. To build the tests, you will need to set <code>BUILD_TESTING</code> to "ON" in either <code>ccmake</code> or when calling CMake:

     ```c++
     cmake -DBUILD_TESTING=ON ../src
     ```
     <h4>Windows</h4>
     The gtest library included in the repo needs to be built with forced shared libraries on Windows, so use the following:

     ```c++
     cmake -DBUILD_TESTING=ON -Dgtest_forced_shared_crt=ON ../src
     ```
     Be sure to include all other necessary CMake definitions as annotated above.
