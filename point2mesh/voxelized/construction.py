'''
code for computing voxelized approximations of the Generalized Voronoi Diagram of a triangular mesh
'''
from multiprocessing import Pool
from typing import List,Tuple

from numpy.typing import ArrayLike,NDArray
import numpy as np
import fcl
import trimesh

from numba import cuda,njit

from point2mesh.util.triangle_geometry import parallel_squared_distance_triangles_to_aligned_box
from point2mesh.util import bounding_volume_hierarchy

from point2mesh.triangle_mesh import multiple_points_to_triangle_squared_distance,tri_to_points_squared_hausdorff

def make_pruned_point_to_mesh_ragged_array(spacing:float,meshes:List[trimesh.Trimesh],expansion_factor:float)->Tuple[List[NDArray[np.intp]],List[NDArray[np.intp]],NDArray[np.float_],NDArray[np.intp]]:
    '''
    voxelize domains around a collection of meshes and for each mesh, return the pruned triangle sets for each voxel

    Parameters: spacing : float
                    the edge length of the voxels to use
                meshes : n element list[trimesh.Trimesh]
                    list of triangle meshes to process
                expansion_factor : scalar
                    the domain voxelized is expansion_factor*axis aligned bounding box of each mesh
    Returns:    candidate_triangles : list[NDArray[np.intp]]
                    each list entry corresponds to a mesh. It records as a 1D array the ids of the triangles tested for each voxel. 
                    The offset at which each voxel starts is recorded in voxel2triangles.
                    Thus, for mesh 0, candidate_triangles[0][voxel2triangles[0][i]:voxel2triangles[0][i+1]] is the triangles that need to be tested for points in voxel i
                voxel2triangles : list[NDArray[np.intp]]
                    each list entry corresponds to a mesh. It records as a 1D array the offset at which voxel i begins in candidate_triangles.
                minimums : (n,3) float array
                    for each mesh, the smallest xyz value contained in the voxelization
                domain_widths : (n,3) int array
                    for each mesh, the number of voxels along each axis         
    '''
    candidate_triangles=[]
    voxel2triangles=[]
    minimums=[]
    domain_widths=[]
    for mesh in meshes:
        scaled_half_edge=mesh.bounding_box.extents*expansion_factor/2
        maximums=scaled_half_edge+mesh.bounding_box.transform[:3,3]
        minimums.append(mesh.bounding_box.transform[:3,3]-scaled_half_edge)
        gridvectors=tuple(np.arange(minimums[-1][i],maximums[i],spacing) for i in range(3))
        domain_width=np.array([len(gv) for gv in gridvectors])
        domain_widths.append(domain_width)
        npts=np.prod(domain_width)
        gridpoints=np.stack(np.meshgrid(*gridvectors,indexing='ij'),axis=-1).reshape(npts,3)
        integer_positions=np.stack(np.meshgrid(*tuple(np.arange(len(gv)) for gv in gridvectors),indexing='ij'),axis=-1).reshape(npts,3)
        
        triangle_list=[None]*npts
        triangle_counts=[None]*npts
        
        #compute which triangles of the mesh can be nearest to a point in a given voxel
        collision_mesh=fcl.CollisionObject(trimesh.collision.mesh_to_BVH(mesh),fcl.Transform())
        for i,pt in enumerate(gridpoints):
            flat_index=np.ravel_multi_index(integer_positions[i],domain_width)
            triangle_list[flat_index]=get_candidate_triangles_with_pruning(pt,mesh,collision_mesh,spacing)
            triangle_counts[flat_index]=len(triangle_list[flat_index])
        #convert list of arrays to flat array plus an array of offsets
        voxel2triangles.append(np.concatenate(([0],np.cumsum(triangle_counts[:-1]))))
        flat_array=np.concatenate(triangle_list)
        candidate_triangles.append(flat_array)
    minimums=np.array(minimums)
    domain_widths=np.array(domain_widths)
    return candidate_triangles,voxel2triangles,minimums,domain_widths

def multiprocess_make_pruned_point_to_mesh_ragged_array(spacing:float,meshes:List[trimesh.Trimesh],expansion_factor:float,n_workers:int)->Tuple[List[NDArray[np.intp]],List[NDArray[np.intp]],NDArray[np.float_],NDArray[np.intp]]:
    '''
    use multiprocessing to voxelize domains around a collection of meshes and for each mesh, return the pruned triangle sets for each voxel

    Parameters: spacing : float
                    the edge length of the voxels to use
                meshes : n element list[trimesh.Trimesh]
                    list of triangle meshes to process
                expansion_factor : scalar
                    the domain voxelized is expansion_factor*axis aligned bounding box of each mesh
                n_workers : int
                    number of worker processes to use
    Returns:    candidate_triangles : list[NDArray[np.intp]]
                    each list entry corresponds to a mesh. It records as a 1D array the ids of the triangles tested for each voxel. 
                    The offset at which each voxel starts is recorded in voxel2triangles.
                    Thus, for mesh 0, candidate_triangles[0][voxel2triangles[0][i]:voxel2triangles[0][i+1]] is the triangles that need to be tested for points in voxel i
                voxel2triangles : list[NDArray[np.intp]]
                    each list entry corresponds to a mesh. It records as a 1D array the offset at which voxel i begins in candidate_triangles.
                minimums : (n,3) float array
                    for each mesh, the smallest xyz value contained in the voxelization
                domain_widths : (n,3) int array
                    for each mesh, the number of voxels along each axis  
    '''
    candidate_triangles=[]
    voxel2triangles=[]
    minimums=[]
    domain_widths=[]
    for mesh in meshes:
        scaled_half_edge=mesh.bounding_box.extents*expansion_factor/2
        maximums=scaled_half_edge+mesh.bounding_box.transform[:3,3]
        minimums.append(mesh.bounding_box.transform[:3,3]-scaled_half_edge)
        gridvectors=tuple(np.arange(minimums[-1][i],maximums[i],spacing) for i in range(3))
        domain_width=np.array([len(gv) for gv in gridvectors])
        domain_widths.append(domain_width)
        npts=np.prod(domain_width)
        gridpoints=np.stack(np.meshgrid(*gridvectors,indexing='ij'),axis=-1).reshape(npts,3)
        
        triangle_list=[None]*npts
        triangle_counts=[None]*npts
        
        #compute which triangles of the mesh can be nearest to a point in a given voxel
        with Pool(n_workers,initializer=init_get_candidates_pruning,initargs=(mesh,spacing)) as pool:
            triangle_list=pool.map(worker_get_candidates_pruning,gridpoints)
        triangle_counts=[len(tl) for tl in triangle_list]
        #convert list of arrays to flat array plus an array of offsets
        voxel2triangles.append(np.concatenate(([0],np.cumsum(triangle_counts[:-1]))))
        flat_array=np.concatenate(triangle_list)
        candidate_triangles.append(flat_array)
    minimums=np.array(minimums)
    domain_widths=np.array(domain_widths)
    return candidate_triangles,voxel2triangles,minimums,domain_widths

def make_pruned_point_to_mesh_gpu_array(spacing:float,meshes:List[trimesh.Trimesh],expansion_factor:float):
    '''
    voxelize domains around a collection of meshes and for each mesh, return the pruned triangle sets for each voxel as GPU arrays

    Parameters: spacing : float
                    the edge length of the voxels to use
                meshes : n element list[trimesh.Trimesh]
                    list of triangle meshes to process
                expansion_factor : scalar
                    the domain voxelized is expansion_factor*axis aligned bounding box of each mesh
    Returns:    candidate_triangles : list[DeviceNDArray[np.intp]]
                    each list entry corresponds to a mesh. It records as a 1D array the ids of the triangles tested for each voxel. 
                    The offset at which each voxel starts is recorded in voxel2triangles.
                    Thus, for mesh 0, candidate_triangles[0][voxel2triangles[0][i]:voxel2triangles[0][i+1]] is the triangles that need to be tested for points in voxel i
                voxel2triangles : list[DeviceNDArray[np.intp]]
                    each list entry corresponds to a mesh. It records as a 1D array the offset at which voxel i begins in candidate_triangles.
                minimums : (n,3) float Device array
                    for each mesh, the smallest xyz value contained in the voxelization
                domain_widths : (n,3) int Device array
                    for each mesh, the number of voxels along each axis         
    '''
    candidate_triangles,voxel2triangles,minimums,domain_widths=make_pruned_point_to_mesh_ragged_array(spacing,meshes,expansion_factor)
    #put arrays on GPU
    candidates_gpu=[cuda.to_device(ct) for ct in candidate_triangles]
    voxel2triangles_gpu=[cuda.to_device(vt) for vt in voxel2triangles]
    return candidates_gpu,voxel2triangles_gpu,cuda.to_device(minimums),cuda.to_device(domain_widths)

def init_get_candidates_pruning(mesh,spacing):
    '''
    initialize worker for computing triangle sets for voxels

    Parameters: mesh : trimesh.Trimesh
                    the mesh to consider
                spacing : float
                    the edge length of the voxels to use
    defines globals g_mesh, g_spacing, and g_collision_mesh
    '''
    global g_mesh,g_spacing,g_collision_mesh
    g_collision_mesh=fcl.CollisionObject(trimesh.collision.mesh_to_BVH(mesh),fcl.Transform())
    g_mesh=mesh
    g_spacing=spacing

def worker_get_candidates_pruning(pt):
    '''
    call get_candidate_triangles_with_pruning for a specific point using globals g_mesh, g_collision_mesh, and g_spacing
    '''
    return get_candidate_triangles_with_pruning(pt,g_mesh,g_collision_mesh,g_spacing)

def get_candidate_triangles_with_pruning(pt:ArrayLike,mesh:trimesh.Trimesh,collision_mesh:fcl.CollisionObject,spacing:float):
    '''
    core function to compute for a voxel a small set of triangles that contains the triangle closest to each point in the voxel

    Parameters: pt : (3,) float array
                    the corner of the voxel with the smallest coordinate values
                mesh : trimesh.Trimesh
                    the mesh
                collision_mesh : fcl.CollisionObject
                    the mesh with FCL collision preprocessing done
                spacing : float or (3,) float array
                    the edge lengths of the voxel. Not tested with non-singleton value!
    Returns: integer array containing the ids of the triangles in the set
    '''
    corner_shifts=np.stack(np.meshgrid(np.array([0,1]),np.array([0,1]),np.array([0,1])),axis=-1).reshape(8,3)
    #make voxel
    voxel=fcl.Box(*(spacing*np.ones((3,))))
    corners=pt+corner_shifts*spacing
    tf_centered=fcl.Transform(np.eye(3),pt+spacing/2)
    collision_voxel=fcl.CollisionObject(voxel,tf_centered)
    #test if any triangles intersect the voxel
    collision_request=fcl.CollisionRequest(num_max_contacts=len(mesh.faces)*6,enable_contact=True)
    collision_result=fcl.CollisionResult()
    ret=fcl.collide(collision_mesh,collision_voxel,collision_request,collision_result)#Step 1
    if ret>0:
        #Step 2
        #for each intersecting triangle, compute the maximum distance to a corner of the voxel
        #then set l1 to the minimum of that over all intersecting triangles
        nintersection=len(collision_result.contacts)
        intersecting_tris=[contact.b1 for contact in collision_result.contacts]

        l1squared,triangle_idx=tri_to_points_squared_hausdorff(mesh.triangles[intersecting_tris],corners)
        T1=collision_result.contacts[triangle_idx].b1
        l1=np.sqrt(l1squared)
    else:
        #no intersecting T1, so for step 1 we instead use the nearest triangle to the voxel
        distance_request=fcl.DistanceRequest()
        distance_result=fcl.DistanceResult()
        ret=fcl.distance(collision_mesh,collision_voxel,distance_request,distance_result)
        T1=distance_result.b1
        triangle=mesh.triangles[distance_result.b1]
        #Step 2
        #get the maximum distance between the triangle nearest the voxel and a point in the voxel
        #this is equal to the maximum distance from a corner of the uninflated voxel to the triangle
        #this is still conservative in that another triangle with the same or slightly larger minimum distance could have smaller max corner distance
        l1=np.sqrt(np.max(multiple_points_to_triangle_squared_distance(triangle[0],triangle[1],triangle[2],corners,1e-12)))
    #Step 3
    #inflate the voxel by f1, such that the shortest distance from any point in the UNINFLATED voxel to the inflated boundary is at least l1
    #that distance is (f1-1)*spacing/2
    f1=l1*2/spacing+1
    box_B=fcl.CollisionObject(fcl.Box(*(f1*spacing*np.ones((3,)))),tf_centered)

    #Step 4
    #get all triangles that could potentially be closer to a point in V than T1 is by intersecting with box B to return a superset, S
    collision_request=fcl.CollisionRequest(num_max_contacts=len(mesh.faces)*6,enable_contact=True)
    collision_result=fcl.CollisionResult()
    ret=fcl.collide(collision_mesh,box_B,collision_request,collision_result)
    S={contact.b1 for contact in collision_result.contacts}
    S.add(T1)
    S=list(S)
    Striangles=mesh.triangles[S]
    #Step 5
    #over every triangle Ti in S, find the maximum distance between any point in V and Ti (the one-sided Hausdorff distance from V to Ti)
    upper_bound,_=tri_to_points_squared_hausdorff(Striangles,corners)
    #Step 6
    #any triangle whose minimum distance to V is larger than the smallest hausdorff distance is redundant
    #so compute those minimum distances
    sides=spacing*np.ones((3,))
    center=pt+spacing/2
    minimum_distances=parallel_squared_distance_triangles_to_aligned_box(Striangles,sides,center)
    #Step 7
    #keep only those triangles whose minimum distance is less than the upper bound
    return np.array([Si for idx,Si in enumerate(S) if upper_bound>minimum_distances[idx]])

@njit
def get_candidate_triangles_via_rss(pt:ArrayLike,triangles:ArrayLike,rsstree:List[bounding_volume_hierarchy.rssnode],spacing:float):
    '''
    core function to compute for a voxel a small set of triangles that contains the triangle closest to each point in the voxel

    Parameters: pt : (3,) float array
                    the corner of the voxel with the smallest coordinate values
                mesh : trimesh.Trimesh
                    the mesh
                collision_mesh : fcl.CollisionObject
                    the mesh with FCL collision preprocessing done
                spacing : float or (3,) float array
                    the edge lengths of the voxel. Not tested with non-singleton value!
    Returns: integer array containing the ids of the triangles in the set
    '''
    corner_shifts=np.array([[0,0,0],[0,0,1],[0,1,0],[1,0,0],[0,1,1],[1,0,1],[1,1,0],[1,1,1]])
    #make voxel
    side_lengths=spacing*np.ones(3)
    center=pt+side_lengths/2
    corners=pt+corner_shifts*spacing
    #test if any triangles intersect the voxel
    triangle_ids,_,_=bounding_volume_hierarchy.aligned_box_get_nearby_triangles(side_lengths,center,rsstree,triangles,0.0)
    if len(triangle_ids)>0:
        #Step 2
        #for each intersecting triangle, compute the maximum distance to a corner of the voxel
        #then set l1 to the minimum of that over all intersecting triangles
        l1squared,triangle_idx=tri_to_points_squared_hausdorff(triangles[np.array(triangle_ids)],corners)
        T1=triangle_ids[triangle_idx]
        l1=np.sqrt(l1squared)
    else:
        #no intersecting T1, so for step 1 we instead use the nearest triangle to the voxel
        _,_,T1=bounding_volume_hierarchy.get_aligned_box_distance_queue(side_lengths,center,rsstree,triangles)
        triangle=triangles[T1]
        #Step 2
        #get the maximum distance between the triangle nearest the voxel and a point in the voxel
        #this is equal to the maximum distance from a corner of the uninflated voxel to the triangle
        #this is still conservative in that another triangle with the same or slightly larger minimum distance could have smaller max corner distance
        l1=np.sqrt(np.max(multiple_points_to_triangle_squared_distance(triangle[0],triangle[1],triangle[2],corners,1e-12)))

    #Step 2, 3, and 6
    #get all triangles that could potentially be closer to a point in V than T1 is by returning all triangles whose min distance is not larger than l1
    S,minimum_distances,_=bounding_volume_hierarchy.aligned_box_get_nearby_triangles(side_lengths,center,rsstree,triangles,l1)
    Striangles=triangles[np.array(S)]
    #Step 5
    #over every triangle Ti in S, find the maximum distance between any point in V and Ti (the one-sided Hausdorff distance from V to Ti)
    upper_bound_squared,_=tri_to_points_squared_hausdorff(Striangles,corners)
    upper_bound=np.sqrt(upper_bound_squared)
    #Step 7
    #keep only those triangles whose minimum distance is less than the upper bound
    return np.array([Si for idx,Si in enumerate(S) if upper_bound>minimum_distances[idx]])