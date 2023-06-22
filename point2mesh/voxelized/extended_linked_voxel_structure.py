'''
Implements the "Linked Voxel Structure" of 
Steffen Hauth, Yavuz Murtezaoglu, Lars Linsen,
Extended linked voxel structure for point-to-mesh distance computation and its application to NC collision detection,
Computer-Aided Design,
Volume 41, Issue 12,
2009,
Pages 896-906,
ISSN 0010-4485,
https://doi.org/10.1016/j.cad.2009.06.007.
'''
from math import fabs,sqrt,ceil

import numpy as np

from numba import njit,prange,cuda
from numba.typed import List
from numba.types import int64 as nb_int64
from point2mesh.util.Math import massnorm
from point2mesh.util.ArrayTricks import threeD_multi_to_1D
from point2mesh.util import binary_space_partition_tree,bounding_volume_hierarchy
from point2mesh.triangle_mesh import cuda_point_to_triangle_squared_distance,point_to_triangle_squared_distance


def make_linked_voxels_ragged_array(spacing,meshes,expansion_factor,use_gpu=False,use_caching=False,exact=True,use_rss=True):
    candidate_triangles=[]
    voxel2triangles=[]
    linked_voxels=[]
    voxel2voxel=[]
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

        triangle_list=[np.empty(0,np.int64) for _ in range(npts)]
        triangle_counts=np.zeros(npts,dtype=np.int64)

        linked_voxel_list=[np.empty(0,np.int64) for _ in range(npts)]
        linked_voxel_counts=triangle_counts.copy()
        if use_gpu:
            gpu_triangles=cuda.to_device(mesh.triangles.reshape((len(mesh.triangles)*9,)).astype(np.float32))
        elif use_caching:
            ntri=len(mesh.triangles)
            distance_cache=np.empty(ntri)
        if use_rss:
            rsstree=bounding_volume_hierarchy.construct_rss_tree(np.arange(len(mesh.triangles)),mesh.triangles,leaf_size=20)
        for i,pt in enumerate(gridpoints):
            flat_index=np.ravel_multi_index(integer_positions[i],domain_width)
            #voxel is non-empty if there is a triangle inside the circumscribed sphere of the voxel
            if use_rss:
                voxel_nonempty,_,_=bounding_volume_hierarchy.is_distance_lte(pt+spacing/2,rsstree,mesh.triangles,(np.sqrt(3)/2*spacing)**2)
            elif use_gpu:
                distances=point2face_distances(pt+spacing/2,gpu_triangles,512)
                tris_found=get_candidate_triangles_in_sphere_gpu(np.sqrt(3)/2*spacing,distances)
                voxel_nonempty=len(tris_found)>0
            elif use_caching:
                distance_cache[:]=-1.0
                tris_found,distance_cache=get_candidate_triangles_in_sphere_nofcl(pt+spacing/2,np.sqrt(3)/2*spacing,mesh.triangles,distance_cache)
                voxel_nonempty=len(tris_found)>0
            elif exact:
                tris_found=get_candidate_triangles_in_sphere_nocache(pt+spacing/2,np.sqrt(3)/2*spacing,mesh.triangles)
                voxel_nonempty=len(tris_found)>0
            else:
                tris_found=get_candidate_triangles_in_sphere_just_vertex(pt+spacing/2,np.sqrt(3)/2*spacing,mesh.triangles)#only check if vertices lie in the sphere
                voxel_nonempty=len(tris_found)>0
            if voxel_nonempty:
                #compute which triangles of the mesh intersect a sphere of radius 3*sqrt(3)*spacing/2 centered on the voxel
                if use_gpu:
                    if use_rss:
                        #haven't computed distances yet
                        distances=point2face_distances(pt+spacing/2,gpu_triangles,512)
                    triangle_list[flat_index]=get_candidate_triangles_in_sphere_gpu(3*np.sqrt(3)/2*spacing,distances)
                elif use_caching:
                    triangle_list[flat_index],_=get_candidate_triangles_in_sphere_nofcl(pt+spacing/2,3*np.sqrt(3)*spacing/2,mesh.triangles,distance_cache)#use cached distances to accelerate intersection test
                elif exact:
                    triangle_list[flat_index]=get_candidate_triangles_in_sphere_nocache(pt+spacing/2,3*np.sqrt(3)*spacing/2,mesh.triangles)
                else:
                    triangle_list[flat_index]=get_candidate_triangles_in_sphere_just_vertex(pt+spacing/2,3*np.sqrt(3)/2*spacing,mesh.triangles)#only check if vertices lie in the sphere
                triangle_counts[flat_index]=len(triangle_list[flat_index])
        #convert lists of arrays to flat array plus an array of offsets
        voxel2triangles.append(np.concatenate(([0],np.cumsum(triangle_counts[:-1]))))
        flat_array=np.concatenate(triangle_list)
        candidate_triangles.append(flat_array)
        #delete what we can to save memory; fine grids lead to lots and lots of voxel links
        if use_gpu:
            del distances
            del gpu_triangles
        if use_rss:
            del rsstree
        del triangle_list
        print("Starting empty voxels")
        #for every empty voxel, find the minimal radius of a sphere of voxels centered on the empty voxel that contains at least one non-empty voxel
        for i,pt in enumerate(gridpoints):
            flat_index=np.ravel_multi_index(integer_positions[i],domain_width)
            if triangle_counts[flat_index]==0:
                r=0
                all_empty=True
                while all_empty:
                    r+=1
                    all_empty=all((triangle_counts[idx]==0 for idx in flat_sphere_coord_generator(integer_positions[i],r,domain_width)))
                #link to all non empty voxels within 3 layers of that minimal sphere
                linked_voxel_list[flat_index]=get_non_empty_comprehension(integer_positions[i],r,domain_width,triangle_counts)
                linked_voxel_counts[flat_index]=len(linked_voxel_list[flat_index])
        voxel2voxel.append(np.concatenate(([0],np.cumsum(linked_voxel_counts[:-1]))))
        linked_voxels.append(np.concatenate(linked_voxel_list))
    minimums=np.array(minimums)
    domain_widths=np.array(domain_widths)
    return candidate_triangles,voxel2triangles,linked_voxels,voxel2voxel,minimums,domain_widths

@njit(parallel=True)
def get_candidate_triangles_in_sphere_nofcl(center,radius,triangles,distance_cache):
    to_include=np.sum(massnorm(triangles-center)<radius,1)>0
    for i in prange(len(triangles)):
        if not to_include[i]:
            dist=distance_cache[i]
            if dist<0:
                dist=point_to_triangle_squared_distance(triangles[i][0],triangles[i][1],triangles[i][2],center,1e-12)
                distance_cache[i]=dist
            if radius*radius>dist:
                #triangle intersects sphere
                to_include[i]=True
    return np.nonzero(to_include)[0],distance_cache

@njit(parallel=True)
def get_candidate_triangles_in_sphere_nocache(center,radius,triangles):
    to_include=np.sum(massnorm(triangles-center)<radius,1)>0
    for i in prange(len(triangles)):
        if not to_include[i]:
            dist=point_to_triangle_squared_distance(triangles[i][0],triangles[i][1],triangles[i][2],center,1e-12)
            if radius*radius>dist:
                #triangle intersects sphere
                to_include[i]=True
    return np.nonzero(to_include)[0]

@njit(parallel=True)
def get_candidate_triangles_in_sphere_just_vertex(center,radius,triangles):
    return np.nonzero(np.sum(massnorm(triangles-center)<radius,1)>0)[0]

def get_candidate_triangles_in_sphere_gpu(radius,distances):
    return np.nonzero(distances<radius)[0]

def point2face_distances(point,triangles_on_device,threads_per_block=1024):
    '''
    compute distance from each of a set of points to a collection of triangles

    Parameters: points : 3, float array
                    the point to compute distances for
                triangles_on_device : (m*9,) float device array
                    the vertices of the triangles, as a flattened C-order device array
                threads_per_block : int or empty (default 1024)
                    # of threads to use per block (only obeyed approximately)
    Return:     distances : (m,) float array
                    the actual, guaranteed non-negative, distance from the point to each triangle
    '''
    m=int(len(triangles_on_device)//9)

    #use a 2D grid, with first dim used for stepping along points and the second stepping along triangles
    tpb=int(sqrt(threads_per_block))
    blocks_per_grid=ceil(m/tpb)
    kernel_signature=(blocks_per_grid,(tpb,tpb))

    distances=cuda.device_array((m,))
    point2face_kernel[kernel_signature](point,triangles_on_device,distances)
    return distances.copy_to_host()

@cuda.jit
def point2face_kernel(query,triangles,distances):
    tol=1e-12
    tid=cuda.grid(1)
    if tid*9>=len(triangles):
        return
    else:
        distances[tid]=sqrt(abs(cuda_point_to_triangle_squared_distance(triangles[tid*9:tid*9+9],query,tol)))

@njit
def permutation3D_with_sign_flips_generator(iterable):
    #hard code sign flips
    sign_flips=np.array([[-1, -1, -1],
       [-1, -1,  1],
       [ 1, -1, -1],
       [ 1, -1,  1],
       [-1,  1, -1],
       [-1,  1,  1],
       [ 1,  1, -1],
       [ 1,  1,  1]])
    #hard code permutations since itertools.permutations not supported
    permuted=((iterable[0],iterable[1],iterable[2]),(iterable[0],iterable[2],iterable[1]),(iterable[1],iterable[0],iterable[2]),(iterable[1],iterable[2],iterable[0]),(iterable[2],iterable[0],iterable[1]),(iterable[2],iterable[1],iterable[0]))
    for item in permuted:
        for s in sign_flips:
            yield (item[0]*s[0],item[1]*s[1],item[2]*s[2])

@njit
def flat_sphere_coord_generator(integer_position,r,domain_width):
    for z in range(int(np.ceil(r/np.sqrt(3)))):
        x=r
        y=nb_int64(0)
        d=1/4-r
        for i in range(1,z):
            d=d-2*(i-1)+1
        while d>0:
            d=d-2*x+1
            x-=1
        while x>=y:
            if y>=z:
                for coord in permutation3D_with_sign_flips_generator((x,y,z)):
                    ball_coord=np.array(coord)+integer_position
                    if np.all(ball_coord<domain_width) and np.all(ball_coord>=0):
                        yield threeD_multi_to_1D(ball_coord[0],ball_coord[1],ball_coord[2],domain_width)
            d+=2*y+1
            y+=1
            if d>0:
                d=d-2*x+1
                x-=1

@njit
def get_non_empty_comprehension(integer_position,r,domain_width,triangle_counts):
    return np.array([idx for ri in range(r,r+4) for idx in flat_sphere_coord_generator(integer_position,ri,domain_width) if triangle_counts[idx]>0],dtype=np.int64)

@njit
def linked_voxels_point2mesh_distance_nobsp(point,triangles,voxels2triangles,candidate_triangles,voxel2voxel,linked_voxels,minimums,spacing,domain_width,tol):
    count=0
    int_idx=((point-minimums)//spacing).astype(np.int64)
    if np.any(int_idx>=domain_width) or np.any(int_idx<0):
        return np.inf,0
    else:
        distance_cache=np.full(int(len(triangles)//9),-1.0)
        #get the 1D index of this voxel
        array_pos=threeD_multi_to_1D(int_idx[0],int_idx[1],int_idx[2],domain_width)
        mindistsquared,tested=voxel_point2mesh_distance_squared(point,triangles,voxels2triangles,candidate_triangles,array_pos,tol,distance_cache)
        count+=tested
        #if the voxel is empty, mindistsquared is still infinity
        if not np.isfinite(mindistsquared):
            #get the voxels linked to this voxel
            voxel_id_start=voxel2voxel[array_pos]
            voxel_id_end=voxel2voxel[array_pos+1]
            voxel_ids=linked_voxels[voxel_id_start:voxel_id_end]
            #compute shortest distance to each triangle assigned to those voxels
            for voxel_id in voxel_ids:
                candidate_dist_squared,tested=voxel_point2mesh_distance_squared(point,triangles,voxels2triangles,candidate_triangles,voxel_id,tol,distance_cache)
                count+=tested
                if candidate_dist_squared<mindistsquared:
                    mindistsquared=candidate_dist_squared
        return sqrt(mindistsquared),count
        
@njit
def voxel_point2mesh_distance_squared(point,triangles,voxels2triangles,candidate_triangles,voxel_id,tol,distance_cache):
    count=0
    #get the triangles assigned to this voxel
    triangle_id_start=voxels2triangles[voxel_id]
    triangle_id_end=voxels2triangles[voxel_id+1]
    triangle_ids=candidate_triangles[triangle_id_start:triangle_id_end]
    mindistsquared=np.inf
    #compute the shortest distance to those triangles
    for triangle_id in triangle_ids:
        candidate_dist_squared=distance_cache[triangle_id]
        if candidate_dist_squared<0:
            idx=triangle_id*9
            candidate_dist_squared=fabs(cuda_point_to_triangle_squared_distance(triangles[idx:idx+9],point,tol))
            count+=1
            distance_cache[triangle_id]=candidate_dist_squared
        if candidate_dist_squared<mindistsquared:
            mindistsquared=candidate_dist_squared
    return mindistsquared,count

@njit
def make_voxel_bsptrees(max_depth,rng,voxels2triangles,candidate_triangles,triangle_vertices):
    bsptrees=List()
    for voxel_id in range(len(voxels2triangles)):
        triangle_id_start=voxels2triangles[voxel_id]
        triangle_id_end=voxels2triangles[voxel_id+1]
        triangle_ids=candidate_triangles[triangle_id_start:triangle_id_end]
        bsptrees.append(binary_space_partition_tree.construct_on_subset(binary_space_partition_tree.split_random,rng,triangle_vertices,triangle_ids,max_depth,1))
    return bsptrees
        
@njit
def voxel_tree_point2mesh_distance_squared(point,triangle_vertices,bsptrees,voxel_id,distance_cache,best_distance_squared):
    return binary_space_partition_tree.get_distance_squared(point,bsptrees[voxel_id],triangle_vertices,distance_cache,best_distance_squared)

@njit
def linked_voxels_point2mesh_distance_bsp(point,triangle_vertices,bsptrees,voxel2voxel,linked_voxels,minimums,spacing,domain_width):
    count=0
    int_idx=((point-minimums)//spacing).astype(np.int64)
    mindistsquared=np.inf
    if np.any(int_idx>=domain_width) or np.any(int_idx<0):
        return mindistsquared,0
    else:
        distance_cache=np.full(len(triangle_vertices),-1.0)
        #get the 1D index of this voxel
        array_pos=threeD_multi_to_1D(int_idx[0],int_idx[1],int_idx[2],domain_width)
        mindistsquared,tested=voxel_tree_point2mesh_distance_squared(point,triangle_vertices,bsptrees,array_pos,distance_cache,mindistsquared)
        count+=tested
        #if the voxel is empty, mindistsquared is still infinity
        if not np.isfinite(mindistsquared):
            #get the voxels linked to this voxel
            voxel_id_start=voxel2voxel[array_pos]
            voxel_id_end=voxel2voxel[array_pos+1]
            voxel_ids=linked_voxels[voxel_id_start:voxel_id_end]
            #compute shortest distance to each triangle assigned to those voxels
            for voxel_id in voxel_ids:
                candidate_dist_squared,tested=voxel_tree_point2mesh_distance_squared(point,triangle_vertices,bsptrees,voxel_id,distance_cache,mindistsquared)
                count+=tested
                if candidate_dist_squared<mindistsquared:
                    mindistsquared=candidate_dist_squared
        return sqrt(mindistsquared),count

@njit
def point2mesh_via_linked_cpu_serial(points,triangles,voxel2triangles,candidate_triangles,voxel2voxels,linked_voxels,minimums,spacing,domain_width):
    distances=np.empty(len(points))
    counts=np.empty(len(points),dtype=np.int64)
    tol=1e-12
    for i,point in enumerate(points):
        distances[i],counts[i]=linked_voxels_point2mesh_distance_nobsp(point,triangles,voxel2triangles,candidate_triangles,voxel2voxels,linked_voxels,minimums,spacing,domain_width,tol)
    return distances,counts

@njit
def point2mesh_via_linked_bsp_cpu_serial(points,triangle_vertices,bsptrees,voxel2voxels,linked_voxels,minimums,spacing,domain_width):
    distances=np.empty(len(points))
    counts=np.empty(len(points),dtype=np.int64)
    for i,point in enumerate(points):
        distances[i],counts[i]=linked_voxels_point2mesh_distance_bsp(points[i],triangle_vertices,bsptrees,voxel2voxels,linked_voxels,minimums,spacing,domain_width)
    return distances,counts

@njit(parallel=True)
def verify_linked_voxel_distance(test_points,triangles,voxels2triangles,candidate_triangles,voxel2voxel,linked_voxels,minimums,spacing,domain_width,tol):
    ntri=int(len(triangles)//9)
    unflat_triangles=triangles.reshape((ntri,3,3))
    errors=np.empty(len(test_points))
    for i in prange(len(test_points)):
        linked_dist,_=linked_voxels_point2mesh_distance_nobsp(test_points[i],triangles,voxels2triangles,candidate_triangles,voxel2voxel,linked_voxels,minimums,spacing,domain_width,tol)
        exhaustive_distance=sqrt(min([point_to_triangle_squared_distance(t[0],t[1],t[2],test_points[i],1e-12) for t in unflat_triangles]))
        errors[i]=np.abs(linked_dist-exhaustive_distance)
    return errors

@njit(parallel=True)
def verify_linked_voxel_bsp_distance(test_points,triangle_vertices,bsptrees,voxel2voxel,linked_voxels,minimums,spacing,domain_width):
    errors=np.empty(len(test_points))
    for i in prange(len(test_points)):
        linked_dist,_=linked_voxels_point2mesh_distance_bsp(test_points[i],triangle_vertices,bsptrees,voxel2voxel,linked_voxels,minimums,spacing,domain_width)
        exhaustive_distance=sqrt(min([point_to_triangle_squared_distance(t[0],t[1],t[2],test_points[i],1e-12) for t in triangle_vertices]))
        errors[i]=np.abs(linked_dist-exhaustive_distance)
    return errors