'''
Boost Software License - Version 1.0 - August 17th, 2003

Permission is hereby granted, free of charge, to any person or organization
obtaining a copy of the software and accompanying documentation covered by
this license (the "Software") to use, reproduce, display, distribute,
execute, and transmit the Software, and to prepare derivative works of the
Software, and to permit third-parties to whom the Software is furnished to
do so, all subject to the following:

The copyright notices in the Software and this entire statement, including
the above license grant, this restriction and the following disclaimer,
must be included in all copies of the Software, in whole or in part, and
all derivative works of the Software, unless such copies or derivative
works are solely in the form of machine-executable object code generated by
a source language processor.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

Based on:
David Eberly, Geometric Tools, Redmond WA 98052
Copyright (c) 1998-2023
Distributed under the Boost Software License, Version 1.0.
https://www.boost.org/LICENSE_1_0.txt
https://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
Version: 6.0.2022.01.06
'''
import numpy as np

from numba import njit,prange,cuda

from point2mesh.util.Math import normsquared,singleton_clip

def gpu_squared_distance_triangles_to_aligned_box(triangle_ids,triangle_vertices_on_gpu,side_lengths,center,threads_per_block=1024):
    n=len(triangle_ids)
    distances=np.empty(n,dtype=np.float32)
    blocks_per_grid=(n+(threads_per_block-1))//threads_per_block
    kernel_signature=(blocks_per_grid,threads_per_block)
    ids_on_gpu=cuda.to_device(triangle_ids)
    side_lengths_on_gpu=cuda.to_device(side_lengths.astype(np.float32))
    center_on_gpu=cuda.to_device(center.astype(np.float32))
    kernel_squared_distance_triangles_to_aligned_box[kernel_signature](ids_on_gpu,triangle_vertices_on_gpu,side_lengths_on_gpu,center_on_gpu,distances)
    return distances

@cuda.jit()
def kernel_squared_distance_triangles_to_aligned_box(triangle_ids,triangle_vertices,side_lengths,center,distances):
    tidx=cuda.grid(1)
    if tidx>=len(triangle_ids) or triangle_ids[tidx]>=len(triangle_vertices):
        return
    distances[tidx]=squared_distance_triangle_to_aligned_box(triangle_vertices[triangle_ids[tidx]],side_lengths,center)
@njit(parallel=False)
def parallel_squared_distance_triangles_to_aligned_box(triangles,side_lengths,center):
    n=len(triangles)
    dsquared=np.empty(n)
    for i in prange(n):
        dsquared[i]=squared_distance_triangle_to_aligned_box(triangles[i],side_lengths,center)
    return dsquared

@njit
def squared_distance_triangle_to_aligned_box(triangle,side_lengths,center):
    pt_on_tri,pt_on_box=closest_point_on_triangle_to_aligned_box(triangle,side_lengths,center)
    return normsquared(pt_on_tri-pt_on_box)
@njit
def closest_point_on_triangle_to_aligned_box(triangle,side_lengths,center):
    shifted_triangle=triangle-center
    closest_on_tri,closest_on_box=closest_point_on_triangle_to_canonical_box(shifted_triangle,side_lengths)
    return closest_on_tri+center,closest_on_box+center
@njit
def closest_point_on_triangle_to_canonical_box(triangle,side_lengths):
    '''
    find closest point on a triangle to an axis aligned box centered at the origin

    Parameters: triangle : (3,3) float array
                    each entry is a vertex
                side_lengths : (3,) float array
                    x,y,z side lengths of the axis aligned box
    Returns: (3,) float closest point on triangle to the box, (3,) float closest point on the box to the triangle
    '''
    E10=triangle[1]-triangle[0]
    E20=triangle[2]-triangle[0]
    K=np.cross(E10,E20)
    sqrLength=np.dot(K,K)
    N=K/np.sqrt(sqrLength)
    closest_on_tri,closest_on_box=closest_point_on_plane_to_canonical_box(N,triangle[0],side_lengths)
    delta=closest_on_tri-triangle[0]
    KxDelta=np.cross(K,delta)
    barycentric1=np.dot(E20,KxDelta)/sqrLength
    barycentric2=-np.dot(E10,KxDelta)/sqrLength
    barycentric0=1-barycentric1-barycentric2
    barycentric=np.array([barycentric0,barycentric1,barycentric2])
    if np.any(barycentric<0) or np.any(barycentric>1):
        #closest plane point not in triangle, so check triangle edges against the box
        i0=2
        sqrDistance=np.inf
        for i1 in range(3):
            candidate_closest_on_segment,candidate_closest_on_box=closest_point_segment_canonical_box(triangle[i0],triangle[i1],side_lengths)
            newDistSquared=normsquared(candidate_closest_on_segment-candidate_closest_on_box)
            if newDistSquared<sqrDistance:
                sqrDistance=newDistSquared
                closest_on_tri=candidate_closest_on_segment
                closest_on_box=candidate_closest_on_box
            i0=i1
    return closest_on_tri,closest_on_box
@njit
def closest_point_on_plane_to_canonical_box(normal,point_on_plane,side_lengths):
    '''
    finds closest point on a plane to an axis aligned box centered at the origin

    Parameters: normal : (3,) float array
                    unit normal of the plane
                point_on_plane : (3,) float array
                    point on the plane
                side_lengths : (3,) float array
                    x,y,z side lengths of the axis aligned box
    Returns: (3,) float closest point on plane to the box, (3,) float closest point on the box to the plane

    based on Geometric Tools DistPlane3CananicalBox3.h:
    // David Eberly, Geometric Tools, Redmond WA 98052
    // Copyright (c) 1998-2023
    // Distributed under the Boost Software License, Version 1.0.
    // https://www.boost.org/LICENSE_1_0.txt
    // https://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
    // Version: 6.0.2022.01.06
    '''
    constant=np.dot(normal,point_on_plane)
    extents=side_lengths/2
    #first reflect the plane so its in the first octant
    origin=constant*normal
    normal=normal.copy()
    reflect=np.zeros(3,dtype=np.bool_)
    for i in range(3):
        if normal[i]<0:
            origin*=-1
            normal[i]*=-1
            reflect[i]=True
    #compute plane-box closest points
    if normal[0]>0:
        if normal[1]>0:
            if normal[2]>0:
                closest_on_plane,closest_on_box=plane_canonicalbox_3d_query(origin,normal,extents)
            else:
                closest_on_plane,closest_on_box=plane_canonicalbox_2d_query(0,1,2,origin,normal,extents)
        else:
            if normal[2]>0:
                closest_on_plane,closest_on_box=plane_canonicalbox_2d_query(0,2,1,origin,normal,extents)
            else:
                closest_on_plane,closest_on_box=plane_canonicalbox_1d_query(0,1,2,origin,extents)
    else:
        if normal[1]>0:
            if normal[2]>0:
                closest_on_plane,closest_on_box=plane_canonicalbox_2d_query(1,2,0,origin,normal,extents)
            else:
                closest_on_plane,closest_on_box=plane_canonicalbox_1d_query(1,2,0,origin,extents)
        else:
            if normal[2]>0:
                closest_on_plane,closest_on_box=plane_canonicalbox_1d_query(2,0,1,origin,extents)
            else:
                closest_on_plane,closest_on_box=plane_canonicalbox_0d_query(origin,extents)
    #undo reflections
    for i in range(3):
        if reflect[i]:
            closest_on_plane[i]*=-1
            closest_on_box[i]*=-1
    return closest_on_plane,closest_on_box

@njit
def plane_canonicalbox_3d_query(origin,normal,extents):
    '''
    computes paired closest points for plane to canonical box for generic case. Internal use only.

    Parameters: origin : (3,) float array
                    origin of the plane
                normal : (3,) float array
                    unit normal for the plane
                extents : (3,) float array
                    half the side lengths of the box
    Returns:    closest_on_plane : (3,) float array
                    the closest point to the box on the plane
                closest_on_box : (3,) float array
                    the closest point on the box to the plane
    '''
    dmin=-np.dot(normal,extents+origin)
    if dmin>=0:
        closest_on_plane=-extents-dmin*normal
        closest_on_box=-extents
    else:
        dmax=np.dot(normal,extents-origin)
        if dmax<=0:
            closest_on_plane=extents-dmax*normal
            closest_on_box=extents
        else:
            s=2*dmin/(dmin-dmax)-1
            closest_on_plane=s*extents
            closest_on_box=closest_on_plane.copy()
    return closest_on_plane,closest_on_box
@njit
def plane_canonicalbox_2d_query(i0,i1,i2,origin,normal,extents):
    '''
    computes paired closest points for plane to canonical box when one normal component is 0. Internal use only.

    Parameters: i0 : integer
                    value indicates which normal components are non-0
                i1 : integer
                    value indicates which normal components are non-0
                i2 : integer
                    value indicates which nornal components are non-0
                origin : (3,) float array
                    origin of the plane
                normal : (3,) float array
                    unit normal for the plane
                extents : (3,) float array
                    half the side lengths of the box
    Returns:    closest_on_plane : (3,) float array
                    the closest point to the box on the plane
                closest_on_box : (3,) float array
                    the closest point on the box to the plane
    '''
    dmin=-(normal[i0]*(extents[i0]+origin[i0])+normal[i1]*(extents[i1]+origin[i1]))
    if dmin>=0:
        closest_on_plane=np.array([-extents[i0]-dmin*normal[i0],-extents[i1]-dmin*normal[i1],extents[i2]])
        closest_on_box=np.array([-extents[i0],-extents[i1],extents[i2]])
    else:
        dmax=normal[i0]*(extents[i0]-origin[i0])+normal[i1]*(extents[i1]-origin[i1])
        if dmax<=0:
            closest_on_plane=np.array([extents[i0]-dmax*normal[i0],extents[i1]-dmax*normal[i1],extents[i2]])
            closest_on_box=extents.copy()
        else:
            s=2*dmin/(dmin-dmax)-1
            closest_on_plane=np.array([s*extents[i0],s*extents[i1],extents[i2]])
            closest_on_box=closest_on_plane.copy()
    return closest_on_plane,closest_on_box
@njit
def plane_canonicalbox_1d_query(i0,i1,i2,origin,extents):
    '''
    computes paired closest points for plane to canonical box when two normal components are 0. Internal use only.

    Parameters: i0 : integer
                    value indicates which normal components are non-0
                i1 : integer
                    value indicates which normal components are non-0
                i2 : integer
                    value indicates which nornal components are non-0
                origin : (3,) float array
                    origin of the plane
                extents : (3,) float array
                    half the side lengths of the box
    Returns:    closest_on_plane : (3,) float array
                    the closest point to the box on the plane
                closest_on_box : (3,) float array
                    the closest point on the box to the plane
    '''
    closest_on_plane=np.array([origin[i0],extents[i1],extents[i2]])
    closest_on_box=np.array([singleton_clip(origin[i0],-extents[i0],extents[i0]),extents[i1],extents[i2]])
    return closest_on_plane,closest_on_box
@njit
def plane_canonicalbox_0d_query(origin,extents):
    '''
    computes paired closest points for plane to canonical box when all normal components are 0. Internal use only.

    Parameters: origin : (3,) float array
                    origin of the plane
                extents : (3,) float array
                    half the side lengths of the box
    Returns:    closest_on_plane : (3,) float array
                    the closest point to the box on the plane
                closest_on_box : (3,) float array
                    the closest point on the box to the plane
    '''
    closest_on_plane=origin.copy()
    closest_on_box=np.clip(origin,-extents,extents)
    return closest_on_plane,closest_on_box
@njit
def closest_point_segment_canonical_box(end0,end1,side_lengths):
    segDirection=end1-end0
    closest_on_line,closest_on_box,parameter=closest_point_line_canonical_box(end0,segDirection,side_lengths)
    if parameter>=0:
        if parameter<=1:
            return closest_on_line,closest_on_box
        else:
            closest=closest_point_point_canonical_box(end1,side_lengths)
            return end1,closest
    else:
        closest=closest_point_point_canonical_box(end0,side_lengths)
        return end0,closest
@njit
def closest_point_line_canonical_box(end,direction_vector,side_lengths):
    extents=side_lengths/2
    origin=end.copy()
    direction=direction_vector.copy()
    reflect=np.zeros(3,np.bool_)
    for i in range(3):
        if direction[i]<0:
            origin[i]*=-1
            direction[i]*=-1
            reflect[i]=True
    #compute line box closest point
    if direction[0]>0:
        if direction[1]>0:
            if direction[2]>0:
                parameter=line_canonicalbox_3d_query(origin,direction,extents)
            else:
                parameter=line_canonical_box_2d_query(0,1,2,origin,direction,extents)
        else:
            if direction[2]>0:
                parameter=line_canonical_box_2d_query(0,2,1,origin,direction,extents)
            else:
                parameter=line_canonicalbox_1d_query(0,1,2,origin,direction,extents)
    else:
        if direction[1]>0:
            if direction[2]>0:
                parameter=line_canonical_box_2d_query(1,2,0,origin,direction,extents)
            else:
                parameter=line_canonicalbox_1d_query(1,0,2,origin,direction,extents)
        else:
            if direction[2]>0:
                parameter=line_canonicalbox_1d_query(2,0,1,origin,direction,extents)
            else:
                parameter=line_canonicalbox_0d_query(origin,extents)
    for i in range(3):
        if reflect[i]:
            origin[i]*=-1
    closest_on_line=end+parameter*direction_vector
    return closest_on_line,origin,parameter

@njit
def line_canonicalbox_3d_query(origin,direction,extents):
    PmE=origin-extents
    prodDxPy=direction[0]*PmE[1]
    prodDyPx=direction[1]*PmE[0]
    if prodDyPx>=prodDxPy:
        prodDzPx=direction[2]*PmE[0]
        prodDxPz=direction[0]*PmE[2]
        if prodDzPx>=prodDxPz:
            return line_face_query(0,1,2,origin,direction,PmE,extents)
        else:
            return line_face_query(2,0,1,origin,direction,PmE,extents)
    else:
        prodDzPy=direction[2]*PmE[1]
        prodDyPz=direction[1]*PmE[2]
        if prodDzPy>=prodDyPz:
            return line_face_query(1,2,0,origin,direction,PmE,extents)
        else:
            return line_face_query(2,0,1,origin,direction,PmE,extents)
@njit
def line_canonical_box_2d_query(i0,i1,i2,origin,direction,extents):
    sqrDistance=0.0
    PmE0=origin[i0]-extents[i1]
    PmE1=origin[i1]-extents[i1]
    prod0=direction[i1]*PmE0
    prod1=direction[i0]*PmE1
    if prod0>=prod1:
        origin[i0] = extents[i0]
        PpE1 = origin[i1] + extents[i1]
        delta = prod0 - direction[i0] * PpE1
        if (delta >= 0):
            lenSqr = direction[i0] * direction[i0] +direction[i1] * direction[i1]
            sqrDistance += delta * delta / lenSqr
            origin[i1] = -extents[i1]
            parameter = -(direction[i0] * PmE0 +direction[i1] * PpE1) / lenSqr
        
        else:
            origin[i1] -= prod0 / direction[i0]
            parameter = -PmE0 / direction[i0]
        
    
    else:
        # line intersects P[i1] = e[i1]
        origin[i1] = extents[i1]

        PpE0 = origin[i0] + extents[i0]
        delta = prod1 - direction[i1] * PpE0
        if (delta >= 0):
        
            lenSqr = direction[i0] * direction[i0] +direction[i1] * direction[i1]
            sqrDistance += delta * delta / lenSqr
            origin[i0] = -extents[i0]
            parameter = -(direction[i0] * PpE0 +direction[i1] * PmE1) / lenSqr
        
        else:
        
            origin[i0] -= prod1 / direction[i1]
            parameter = -PmE1 / direction[i1]

    if (origin[i2] < -extents[i2]):
    
        delta = origin[i2] + extents[i2]
        sqrDistance += delta * delta
        origin[i2] = -extents[i2]
    
    elif (origin[i2] > extents[i2]):
    
        delta = origin[i2] - extents[i2]
        sqrDistance += delta * delta
        origin[i2] = extents[i2]
    return parameter
@njit
def line_canonicalbox_1d_query(i0,i1,i2,origin,direction,extents):
    parameter=(extents[i0]-origin[i0])/direction[i0]
    origin[i0]=extents[i0]
    for i in (i1,i2):
        if origin[i]<-extents[i]:
            delta=origin[i]+extents[i]
            origin[i]=-extents[i]
        elif origin[i]>extents[i]:
            delta=origin[i]-extents[i]
            origin[i]=extents[i]
    return parameter
@njit
def line_canonicalbox_0d_query(origin,extents):
    for i in range(3):
        if origin[i]<-extents[i]:
            origin[i]=-extents[i]
        elif origin[i]>extents[i]:
            origin[i]=extents[i]
    return 0.0
@njit
def line_face_query(i0,i1,i2,origin,direction,PmE,extents):
    '''
    internal code for line to face query. Mutates origin to be the closest point on the box. Returns distance along direction from origin to the closest point on the line
    '''
    parameter=0
    sqrDistance=0.0
    PpE=origin+extents
    if direction[i0]*PpE[i1]>=direction[i1]*PmE[i0]:
        if direction[i0]*PpE[i2]>=direction[i2]*PmE[i0]:
            origin[i0] = extents[i0]
            origin[i1] -= direction[i1] * PmE[i0] / direction[i0]
            origin[i2] -= direction[i2] * PmE[i0] / direction[i0]
            parameter = -PmE[i0] / direction[i0]  
        else:   
            # v[i1] >= -e[i1], v[i2] < -e[i2]
            lenSqr = direction[i0] * direction[i0] +direction[i2] * direction[i2]
            tmp = lenSqr * PpE[i1] - direction[i1] * (direction[i0] * PmE[i0] +direction[i2] * PpE[i2])
            if (tmp <= 2 * lenSqr * extents[i1]):     
                t = tmp / lenSqr
                lenSqr += direction[i1] * direction[i1]
                tmp = PpE[i1] - t
                delta = direction[i0] * PmE[i0] + direction[i1] * tmp +direction[i2] * PpE[i2]
                parameter = -delta / lenSqr
                sqrDistance += PmE[i0] * PmE[i0] + tmp * tmp +PpE[i2] * PpE[i2] + delta * parameter
                origin[i0] = extents[i0]
                origin[i1] = t - extents[i1]
                origin[i2] = -extents[i2]        
            else:        
                lenSqr += direction[i1] * direction[i1]
                delta = direction[i0] * PmE[i0] + direction[i1] * PmE[i1] +direction[i2] * PpE[i2]
                parameter = -delta / lenSqr
                sqrDistance += PmE[i0] * PmE[i0] + PmE[i1] * PmE[i1]+ PpE[i2] * PpE[i2] + delta * parameter

                origin[i0] = extents[i0]
                origin[i1] = extents[i1]
                origin[i2] = -extents[i2] 
    else:  
        if (direction[i0] * PpE[i2] >= direction[i2] * PmE[i0]):      
            # v[i1] < -e[i1], v[i2] >= -e[i2]
            lenSqr = direction[i0] * direction[i0] +direction[i1] * direction[i1]
            tmp = lenSqr * PpE[i2] - direction[i2] * (direction[i0] * PmE[i0] +direction[i1] * PpE[i1])
            if (tmp <= 2 * lenSqr * extents[i2]):          
                t = tmp / lenSqr
                lenSqr += direction[i2] * direction[i2]
                tmp = PpE[i2] - t
                delta = direction[i0] * PmE[i0] + direction[i1] * PpE[i1] +direction[i2] * tmp
                parameter = -delta / lenSqr
                sqrDistance += PmE[i0] * PmE[i0] + PpE[i1] * PpE[i1] +tmp * tmp + delta * parameter

                origin[i0] = extents[i0]
                origin[i1] = -extents[i1]
                origin[i2] = t - extents[i2]         
            else:     
                lenSqr += direction[i2] * direction[i2]
                delta = direction[i0] * PmE[i0] + direction[i1] * PpE[i1] +direction[i2] * PmE[i2]
                parameter = -delta / lenSqr
                sqrDistance += PmE[i0] * PmE[i0] + PpE[i1] * PpE[i1] +PmE[i2] * PmE[i2] + delta * parameter

                origin[i0] = extents[i0]
                origin[i1] = -extents[i1]
                origin[i2] = extents[i2]     
        else:
            # v[i1] < -e[i1], v[i2] < -e[i2]
            lenSqr = direction[i0] * direction[i0] +direction[i2] * direction[i2]
            tmp = lenSqr * PpE[i1] - direction[i1] * (direction[i0] * PmE[i0] +direction[i2] * PpE[i2])
            if (tmp >= 0): 
                # v[i1]-edge is closest
                if (tmp <= 2 * lenSqr * extents[i1]):    
                    t = tmp / lenSqr
                    lenSqr += direction[i1] * direction[i1]
                    tmp = PpE[i1] - t
                    delta = direction[i0] * PmE[i0] + direction[i1] * tmp +direction[i2] * PpE[i2]
                    parameter = -delta / lenSqr
                    sqrDistance += PmE[i0] * PmE[i0] + tmp * tmp +PpE[i2] * PpE[i2] + delta * parameter
                    origin[i0] = extents[i0]
                    origin[i1] = t - extents[i1]
                    origin[i2] = -extents[i2]     
                else:
                    lenSqr += direction[i1] * direction[i1]
                    delta = direction[i0] * PmE[i0] + direction[i1] * PmE[i1]+ direction[i2] * PpE[i2]
                    parameter = -delta / lenSqr
                    sqrDistance += PmE[i0] * PmE[i0] + PmE[i1] * PmE[i1] + PpE[i2] * PpE[i2] + delta * parameter

                    origin[i0] = extents[i0]
                    origin[i1] = extents[i1]
                    origin[i2] = -extents[i2]          
                return parameter
            lenSqr = direction[i0] * direction[i0] +direction[i1] * direction[i1]
            tmp = lenSqr * PpE[i2] - direction[i2] * (direction[i0] * PmE[i0] +direction[i1] * PpE[i1])
            if (tmp >= 0):
                # v[i2]-edge is closest
                if (tmp <= 2 * lenSqr * extents[i2]):
                    t = tmp / lenSqr
                    lenSqr += direction[i2] * direction[i2]
                    tmp = PpE[i2] - t
                    delta = direction[i0] * PmE[i0] + direction[i1] * PpE[i1] +direction[i2] * tmp
                    parameter = -delta / lenSqr
                    sqrDistance += PmE[i0] * PmE[i0] + PpE[i1] * PpE[i1] +tmp * tmp + delta * parameter

                    origin[i0] = extents[i0]
                    origin[i1] = -extents[i1]
                    origin[i2] = t - extents[i2] 
                else: 
                    lenSqr += direction[i2] * direction[i2]
                    delta = direction[i0] * PmE[i0] + direction[i1] * PpE[i1] + direction[i2] * PmE[i2]
                    parameter = -delta / lenSqr
                    sqrDistance += PmE[i0] * PmE[i0] + PpE[i1] * PpE[i1]+ PmE[i2] * PmE[i2] + delta * parameter

                    origin[i0] = extents[i0]
                    origin[i1] = -extents[i1]
                    origin[i2] = extents[i2]
                return parameter
            # (v[i1],v[i2])-corner is closest
            lenSqr += direction[i2] * direction[i2]
            delta = direction[i0] * PmE[i0] + direction[i1] * PpE[i1] +direction[i2] * PpE[i2]
            parameter = -delta / lenSqr
            sqrDistance += PmE[i0] * PmE[i0] + PpE[i1] * PpE[i1]+ PpE[i2] * PpE[i2] + delta * parameter

            origin[i0] = extents[i0]
            origin[i1] = -extents[i1]
            origin[i2] = -extents[i2]
    return parameter
@njit
def closest_point_point_canonical_box(point,side_lengths):
    closest=point.copy()
    extents=side_lengths/2
    N=len(point)
    for i in range(N):
        if point[i]<-extents[i]:
            closest[i]=-extents[i]
        elif point[i]>extents[i]:
            closest[i]=extents[i]
    return closest