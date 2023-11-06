import sys
if len(sys.argv)>1:
    tpb=int(sys.argv[1])
    npts=int(sys.argv[2])
else:
    tpb=128
    npts=100
print(f"threads per block={tpb}")
print(f"number of query points={npts}")
import trimesh
import numpy as np
from point2mesh.util import bounding_volume_hierarchy
mesh=trimesh.load_mesh("point2mesh/assets/coarse_hinge_base.ply")
rss=bounding_volume_hierarchy.construct_rss_tree(np.arange(len(mesh.triangles)),mesh.triangles,max_depth=10,leaf_size=10)
tree_as_arrays=bounding_volume_hierarchy.rsstree_to_arrays(rss,np.float32)
from numba import cuda
on_device=[cuda.to_device(arr) for arr in tree_as_arrays]
tree_on_device=bounding_volume_hierarchy.array_rsstree(*on_device)
triangles_on_device=cuda.to_device(mesh.triangles.astype(np.float32))

kernel=bounding_volume_hierarchy.make_rss_points2mesh_kernel(100)
pts=np.linspace(np.zeros(3),np.ones(3),npts)
distances=bounding_volume_hierarchy.point2mesh_via_rss_gpu(pts,triangles_on_device,tree_on_device,kernel,threads_per_block=tpb)

cpu_distance,_=bounding_volume_hierarchy.point2mesh_via_rss_serial(pts,mesh.triangles,rss)

errors=np.abs(distances-cpu_distance)
print(f"max error {np.max(errors)}")