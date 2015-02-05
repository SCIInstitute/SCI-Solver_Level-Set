SCI-Solver_Level-Set
====================

**Currently in pre-alpha stage, estimated stable release: April 2015**

SCI-Solver_Level-Set is a C++/CUDA library written to solve a system of level set equations on unstructured meshes. It is designed to solve the equations quickly by using GPU hardware.

The code was written by Zhisong Fu. The theory behind this code is published in the paper: "Fast Parallel Solver for the Levelset Equations on Unstructured Meshes"

**AUTHORS:** Zhisong Fu(*a*,*b*), Sergey Yakovlev(*b*), Robert M. Kirby(*a*,*b*), Ross T. Whitaker(*a*,*b*)

`  `*a*-School of Computing, University of Utah, Salt Lake City, UT, USA

`  `*b*-Scientific Computing and Imaging Institute, University of Utah, Salt Lake City, USA

**ABSTRACT:**
Levelset method is a numerical technique for tracking interfaces and shapes. It is actively used within various areas such as physics, chemistry, fluid mechanics, computer vision and microchip fabrication to name a few. Applying the levelset method entails solving the levelset partial differential equation. This paper presents a parallel algorithm for solving the levelset equation on fully unstructured 2D or 3D meshes or manifolds. By taking into account constraints and capabilities of two different computing platforms, the method is suitable for both the coarse-grained parallelism found on CPU-based systems and the fine-grained parallelism of modern massively-SIMD architectures such as graphics processors. In order to solve levelset equation efficiently, we combine the narrowband scheme with domain decomposition: the narrowband fast iterative method (nbFIM) to compute the distance transform by solving an eikonal equation and the patched narrowband (patchNB) scheme to evolve the embedding are proposed in this paper. We also introduce the Hybrid Gathering parallelism strategy to enable regular and lock-free computations in both the nbFIM and patchNB. Finally, we provide the detailed description of the implementation and data structures for the proposed strategies, as well as the performance data for both CPU and GPU implementations.
