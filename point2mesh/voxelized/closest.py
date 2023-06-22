'''
code to use voxelized GVDs to compute closest point on a mesh to query points
'''
import numpy as np

from numba import njit

from point2mesh.util.ArrayTricks import threeD_multi_to_1D

from point2mesh import triangle_mesh

@njit
def point2mesh_closest_cpu(query_point,triangles,voxels2triangles,candidate_triangles,minimums,spacing,domain_width,tol):
    '''
    compute the closest point on a set of triangles to a query point

    Parameters: query_point : (3,) float array
                    the query point, must be within the domain the voxelization was constructed for
                triangles : (ntri*9,) float array
                    the triangle array, flattened
                voxels2triangles : (n_voxel,) integer array
                    each entry stores the first index in candidate_triangles that corresponds to that voxel
                candidate_triangles : (max(voxels2triangles)+1,) integer array
                    array of triangle indices. This is the concatenation of the arrays of triangles that correspond to each voxel
                minimums : (3,) float array
                    the minimum x-y-z position inside the voxelization
                spacing : float
                    the edge length of the voxels
                domain_width : (3,) int array
                    the number of voxels in each dimension
                tol : float
                    the tolerance to use for point to triangle calculations
    Returns:    closest_point : (3,) float array
                    the coordinates of the closest point on a triangle to query_point. NaNs if query_point is outside the voxelization
                squared_distance : float
                    the squared distance from the closest_point to the query_point
                count : int
                    the number of triangles closest point was computed for
    '''
    int_idx=((query_point-minimums)//spacing).astype(np.int64)
    closest_point=np.array([np.nan,np.nan,np.nan])
    if np.any(int_idx>=domain_width) or np.any(int_idx<0):
        return closest_point,np.inf,0
    else:
        array_pos=threeD_multi_to_1D(int_idx[0],int_idx[1],int_idx[2],domain_width)
        triangle_id_start=voxels2triangles[array_pos]
        triangle_id_end=voxels2triangles[array_pos+1]
        triangle_ids=candidate_triangles[triangle_id_start:triangle_id_end]
        mindistsquared=np.inf
        for triangle_id in triangle_ids:
            idx=triangle_id*9
            cx,cy,cz=triangle_mesh.cuda_closest_point_on_triangle(triangles[idx:idx+9],query_point,tol)
            ex=query_point[0]-cx
            ey=query_point[1]-cy
            ez=query_point[2]-cz
            candidate_dist_squared=ex*ex+ey*ey+ez*ez
            if candidate_dist_squared<mindistsquared:
                mindistsquared=candidate_dist_squared
                closest_point[0]=cx
                closest_point[1]=cy
                closest_point[2]=cz
        return closest_point,mindistsquared,len(triangle_ids)

@njit
def closest_point_on_mesh_via_cpu_serial(points,triangles,voxels2triangles,candidate_triangles,minimums,spacing,domain_width):
    '''
    compute the closest point on a set of triangles to query points

    Parameters: points : (npts,3) float array
                    the query points, must be within the domain the voxelization was constructed for (returns nans for closest point if this fails)
                triangles : (ntri*9,) float array
                    the triangle array, flattened
                voxels2triangles : (n_voxel,) integer array
                    each entry stores the first index in candidate_triangles that corresponds to that voxel
                candidate_triangles : (max(voxels2triangles)+1,) integer array
                    array of triangle indices. This is the concatenation of the arrays of triangles that correspond to each voxel
                minimums : (3,) float array
                    the minimum x-y-z position inside the voxelization
                spacing : float
                    the edge length of the voxels
                domain_width : (3,) int array
                    the number of voxels in each dimension
                tol : float
                    the tolerance to use for point to triangle calculations
    Returns:    closest_point : (npts,3) float array
                    the coordinates of the closest point on a triangle to query_point. row is NaNs if query_point is outside the voxelization
                squared_distances : (npts,) float array
                    the squared distance from the closest_point to the query_point
                count : (npts,) int array
                    the number of triangles closest point was computed for
    '''
    closest_points=np.empty_like(points)
    squared_distances=np.empty(len(points))
    counts=np.empty(len(points),dtype=np.int64)
    tol=1e-12
    for i,point in enumerate(points):
        closest_points[i],squared_distances[i],counts[i]=point2mesh_closest_cpu(point,triangles,voxels2triangles,candidate_triangles,minimums,spacing,domain_width,tol)
    return closest_points,squared_distances,counts