'''
code to use voxelized GVDs to compute distance from points to meshes
'''
from math import fabs,sqrt

import numpy as np

from numba import cuda,njit,prange
from numba.types import float32 as nb_float32
from numba.types import int64 as nb_int64

from point2mesh.util.ArrayTricks import threeD_multi_to_1D

from point2mesh import triangle_mesh

def point2mesh_via_gpu(points,mesh,voxels2triangles,candidate_triangles,minimums,spacing,domain_width,threads_per_block=1024):
    '''
    use ragged array representation of GVD to compute point to mesh distance
    '''
    triangles=cuda.to_device(mesh.triangles.reshape((len(mesh.triangles)*9,)))
    queries=cuda.to_device(points)
    distance=np.empty(len(points))
    blocks_per_grid=(len(points)+(threads_per_block-1))//threads_per_block
    kernel_signature=(blocks_per_grid,threads_per_block)
    point2mesh_via_gpu_kernel[kernel_signature](queries,triangles,voxels2triangles,candidate_triangles,minimums,spacing,domain_width,distance)
    return distance

@cuda.jit()
def point2mesh_via_gpu_kernel(points,triangles,voxels2triangles,candidate_triangles,minimums,spacing,domain_width,output):
    tol=nb_float32(1e-12)
    thread_index=cuda.grid(1)#this is the query number
    if thread_index>=len(points):
        return
    point=points[thread_index]
    xid=nb_int64((point[0]-minimums[0])//spacing)
    yid=nb_int64((point[1]-minimums[1])//spacing)
    zid=nb_int64((point[2]-minimums[2])//spacing)
    if xid>=domain_width[0] or xid<0 or yid>=domain_width[1] or yid<0 or zid>=domain_width[2] or zid<0:
        #point is outside the grid on which we have sets of triangles precomputed
        output[thread_index]=nb_float32(np.inf)
    else:
        array_pos=threeD_multi_to_1D(xid,yid,zid,domain_width)
        triangle_id_start=voxels2triangles[array_pos]
        triangle_id_end=voxels2triangles[array_pos+1]
        triangle_ids=candidate_triangles[triangle_id_start:triangle_id_end]
        mindistsquared=nb_float32(np.inf)
        for triangle_id in triangle_ids:
            idx=triangle_id*9
            candidate_dist_squared=fabs(triangle_mesh.cuda_point_to_triangle_squared_distance(triangles[idx:idx+9],point,tol))
            if candidate_dist_squared<mindistsquared:
                mindistsquared=candidate_dist_squared
        output[thread_index]=sqrt(mindistsquared)
    return

@njit
def point2mesh_distance_cpu(point,triangles,voxels2triangles,candidate_triangles,minimums,spacing,domain_width,tol):
    int_idx=((point-minimums)//spacing).astype(np.int64)
    if np.any(int_idx>=domain_width) or np.any(int_idx<0):
        return np.inf,0
    else:
        array_pos=threeD_multi_to_1D(int_idx[0],int_idx[1],int_idx[2],domain_width)
        triangle_id_start=voxels2triangles[array_pos]
        triangle_id_end=voxels2triangles[array_pos+1]
        triangle_ids=candidate_triangles[triangle_id_start:triangle_id_end]
        mindistsquared=np.inf
        for triangle_id in triangle_ids:
            idx=triangle_id*9
            candidate_dist_squared=fabs(triangle_mesh.cuda_point_to_triangle_squared_distance(triangles[idx:idx+9],point,tol))
            if candidate_dist_squared<mindistsquared:
                mindistsquared=candidate_dist_squared
        return sqrt(mindistsquared),len(triangle_ids)

@njit
def point2mesh_via_cpu_serial(points,triangles,voxels2triangles,candidate_triangles,minimums,spacing,domain_width):
    distances=np.empty(len(points))
    counts=np.empty(len(points),dtype=np.int64)
    tol=1e-12
    for i,point in enumerate(points):
        distances[i],counts[i]=point2mesh_distance_cpu(point,triangles,voxels2triangles,candidate_triangles,minimums,spacing,domain_width,tol)
    return distances,counts

@njit(parallel=True)
def point2mesh_via_cpu_parallel(points,triangles,voxels2triangles,candidate_triangles,minimums,spacing,domain_width):
    distances=np.empty(len(points))
    counts=np.empty(len(points),dtype=np.int64)
    tol=1e-12
    for i in prange(len(points)):
        distances[i],counts[i]=point2mesh_distance_cpu(points[i],triangles,voxels2triangles,candidate_triangles,minimums,spacing,domain_width,tol)
    return distances,counts

@njit(parallel=True)
def verify_ragged_array_distance(points,triangles,voxels2triangles,candidate_triangles,minimums,spacing,domain_width):
    errors=np.empty(len(points))
    tol=1e-12
    ntri=int(len(triangles)//9)
    unflat_triangles=triangles.reshape((ntri,3,3))
    for i in prange(len(points)):
        dist,_=point2mesh_distance_cpu(points[i],triangles,voxels2triangles,candidate_triangles,minimums,spacing,domain_width,tol)
        errors[i]=fabs(triangle_mesh.brute_force_point2mesh_cpu(points[i],unflat_triangles)-dist)
    return errors