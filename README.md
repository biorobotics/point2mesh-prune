## point2mesh
Release code for Gutow, G and Howie Choset, "Fast Point to Mesh Distance by Domain Voxelization," 2023. Provides python+numba CPU and GPU implementations of the voxelization scheme proposed in the paper, as well as the reimplementations of RSStrees, BSPtrees, and linked voxel structure compared against.
# Folder Structure
point2triangle_comparisons.py implements convenience code for timing different point2triangle methods, including several third party methods.

triangle_mesh.py implements point to triangle distance and closest point operations for a variety of call signatures and approaches, including GPU versions.

assets contains .PLY files of the three triangle meshes used in the paper

test contains a script to build a voxelization for the smallest of the three meshes and query points in // on the GPU

util contain broadly applicable, generic, documented, and well-tested functions for import and use throughout the code base. Many of the functions are numba accelerated. Includes the BSP and RSStree implementations used in the paper.

voxelized implements the proposed voxelization scheme and the extended linked voxel structure from Hauth 2009 (without the correction for empty voxels).

# Dependencies

Core (versions given are for those used to generate results in the paper)
    python=3.8.10
    numba=0.56.4 (conda or pip, see numba documentation for enabling CUDA support depending on choice)
    trimesh[all] (pip)
    python-fcl (pip)

Comparisons
    pysdf (pip)
    cgal-bindings (pip)
    VCGLib (github)