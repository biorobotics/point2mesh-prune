import trimesh

from point2mesh.point2triangle_comparisons import load_mesh,GPU_32bit_GVD_initialize,GPU_32bit_GVD_test

mesh_path="point2mesh/assets/coarse_hinge_base.ply"
expansion_factor=1.5
n_query=2000000
spacing=0.01
threads_per_block=512

mesh=load_mesh(mesh_path)
#scale bounding box
box_transform=mesh.bounding_box.transform
box_size=expansion_factor*mesh.bounding_box.extents
big_box=trimesh.primitives.Box(extents=box_size,transform=box_transform)
#sample points
query_points=big_box.sample_volume(n_query)
#initialize
spacing32,ct32,vt32,m32,dw32,tri32=GPU_32bit_GVD_initialize(spacing,mesh,expansion_factor)
#run
GPU_32bit_GVD_test(query_points,tri32,threads_per_block,ct32,vt32,m32,spacing32,dw32)