'''
generic distance and closest point functions for points to triangular meshes
'''
from math import ceil,sqrt
import numpy as np
from numba import njit,cuda,prange
import fcl
import trimesh

@njit
def closest_point_on_triangle(triangle,query,tol=None):
    '''
    compute the coordinates of the point in a triangle closest to a 3D point
    
    @param triangle: (3,3) array of the triangle vertices; ie np.array([A,B,C])
    @param query: (3,) array of the query point
    @param tol: None (default) or positive float; treat any number smaller as 0. If None, defaults to smallest normal # of triangle data type
    @return (3,) array the nearest point to query inside or on edges of triangle
    
    Direct transcription of ClosestPtPointTriangle from Real-Time Collision Detection by Christer Ericson (2004)
    '''
    if tol is None:
        tol=np.finfo(triangle.dtype).tiny
    a=triangle[0]
    b=triangle[1]
    c=triangle[2]
    
    #First thing is to check if the query point is closest to vertices A or B
    ab=b-a
    ac=c-a
    ap=query-a
    
    #check vertex a
    d1=np.dot(ab,ap)
    d2=np.dot(ac,ap)
    if d1<=tol and d2<=tol:
        return a
    
    #check vertex b
    bp=query-b
    d3=np.dot(ab,bp)
    d4=np.dot(ac,bp)
    if d3>=-tol and d4<=d3:
        return b
    
    #values computed for checking A and B can be used to check if closest to edge AB
    vc=d1*d4-d3*d2
    if vc<=tol and d1>=-tol and d3<=tol:
        v=d1/(d1-d3)
        return a+v*ab
    
    #now check if closest to vertex C
    cp=query-c
    d5=np.dot(ab,cp)
    d6=np.dot(ac,cp)
    if d6>=-tol and d5<=d6:
        return c
    
    #now check if closest to edge AC
    vb=d5*d2-d1*d6
    if vb<=tol and d2>=-tol and d6<=tol:
        w=d2/(d2-d6)
        return a+w*ac
    
    #check if closest to edge BC
    va=d3*d6-d5*d4
    if va<=tol and (d4-d3)>=-tol and (d5-d6)>=-tol:
        w=(d4-d3)/((d4-d3)+(d5-d6))
        return b+w*(c-b)
    
    #must be inside the face
    denom=1/(va+vb+vc)
    v=vb*denom
    w=vc*denom
    return a+ab*v+ac*w

@njit
def cuda_3d_error(array1,id1,array2,id2):
    e1=array1[id1]-array2[id2]
    e2=array1[id1+1]-array2[id2+1]
    e3=array1[id1+2]-array2[id2+2]
    return e1,e2,e3

@njit
def cuda_dot(x1,y1,z1,x2,y2,z2):
    return x1*x2+y1*y2+z1*z2

@cuda.jit
def closest_point_on_triangles_cuda(triangles,queries,output):
    '''
    compute the coordinates of the point in a triangle closest to a 3D point
    
    @param triangles: (n,3,3) array of the triangle vertices; ie np.array([[A1,B1,C1]])
    @param queries: (n,3) array of the query points
    @param output: (n,3) array of the nearest point to query inside or on edges of triangle
    
    Direct transcription of ClosestPtPointTriangle from Real-Time Collision Detection by Christer Ericson (2004)
    then converted for CUDA using numba.cuda
    '''
    tol=1e-12
    
    #find our place in the arrays
    thread_index=cuda.grid(1)
    point_index=3*thread_index
    if point_index+2>=output.size:
        return
    triangle_index=9*thread_index
    aidx=triangle_index
    bidx=triangle_index+3
    cidx=triangle_index+6
    
    #First thing is to check if the query point is closest to vertices A or B
    #check vertex a
    abx,aby,abz=cuda_3d_error(triangles,bidx,triangles,aidx)
    acx,acy,acz=cuda_3d_error(triangles,cidx,triangles,aidx)
    apx,apy,apz=cuda_3d_error(queries,point_index,triangles,aidx)
    d1=cuda_dot(abx,aby,abz,apx,apy,apz)
    d2=cuda_dot(acx,acy,acz,apx,apy,apz)
    if d1<=tol and d2<=tol:
        output[point_index]=triangles[aidx]
        output[point_index+1]=triangles[aidx+1]
        output[point_index+2]=triangles[aidx+2]
        return
    
    #check vertex b
    bpx,bpy,bpz=cuda_3d_error(queries,point_index,triangles,bidx)
    d3=cuda_dot(abx,aby,abz,bpx,bpy,bpz)
    d4=cuda_dot(acx,acy,acz,bpx,bpy,bpz)
    if d3>=-tol and d4<=d3:
        output[point_index]=triangles[bidx]
        output[point_index+1]=triangles[bidx+1]
        output[point_index+2]=triangles[bidx+2]
        return
    
    #values computed for checking A and B can be used to check if closest to edge AB
    vc=d1*d4-d3*d2
    if vc<=tol and d1>=-tol and d3<=tol:
        v=d1/(d1-d3)
        output[point_index]=triangles[aidx]+v*abx
        output[point_index+1]=triangles[aidx+1]+v*aby
        output[point_index+2]=triangles[aidx+2]+v*abz
        return
    
    #now check if closest to vertex C
    cpx,cpy,cpz=cuda_3d_error(queries,point_index,triangles,cidx)
    d5=cuda_dot(abx,aby,abz,cpx,cpy,cpz)
    d6=cuda_dot(acx,acy,acz,cpx,cpy,cpz)
    if d6>=-tol and d5<=d6:
        output[point_index]=triangles[cidx]
        output[point_index+1]=triangles[cidx+1]
        output[point_index+2]=triangles[cidx+2]
        return
    
    #now check if closest to edge AC
    vb=d5*d2-d1*d6
    if vb<=tol and d2>=-tol and d6<=tol:
        w=d2/(d2-d6)
        output[point_index]=triangles[aidx]+w*acx
        output[point_index+1]=triangles[aidx+1]+w*acy
        output[point_index+2]=triangles[aidx+2]+w*acz
        return
    
    #check if closest to edge BC
    va=d3*d6-d5*d4
    if va<=tol and (d4-d3)>=-tol and (d5-d6)>=-tol:
        w=(d4-d3)/((d4-d3)+(d5-d6))
        bcx,bcy,bcz=cuda_3d_error(triangles, cidx, triangles, bidx)
        output[point_index]=triangles[bidx]+w*bcx
        output[point_index+1]=triangles[bidx+1]+w*bcy
        output[point_index+2]=triangles[bidx+2]+w*bcz
        return
    
    #must be inside the face
    denom=1/(va+vb+vc)
    v=vb*denom
    w=vc*denom
    output[point_index]=triangles[aidx]+abx*v+acx*w
    output[point_index+1]=triangles[aidx+1]+aby*v+acy*w
    output[point_index+2]=triangles[aidx+2]+abz*v+acz*w
    return

@njit
def cuda_closest_point_on_triangle(triangle,query,tol):
    '''
    compute closest point on a flattened triangle to a query point

    Parameters: triangle : (9,) float array
                    flattened triangle as if from np.concatenate([vertex0,vertex1,vertex2])
                query : (3,) float array
                    query point
                tol : float
                    values smaller than this in absolute value are considered 0
    Returns:    x : float
                    first coordinate of closest point
                y : float
                    second coordinate of closest point
                z : float
                    third coordinate of closest point
    '''
    A=triangle[0:3]
    B=triangle[3:6]
    C=triangle[6:9]
    return closest_point_on_triangle(A,B,C,query,tol)

@njit
def closest_point_on_triangle(A,B,C,query,tol):
    '''
    compute the coordinates of the point in a triangle closest to a single 3D point
    
    Parameters: A : (3,) float array
                    first vertex
                B : (3,) float array
                    second vertex
                C : (3,) float array
                    third vertex
                query : (3,) float array
                    the query point
                tol : float
                    values smaller than this in absolute value are considered 0
    Returns:    x : float
                    first coordinate of closest point
                y : float
                    second coordinate of closest point
                z : float
                    third coordinate of closest point
    
    transcription of ClosestPtPointTriangle from Real-Time Collision Detection by Christer Ericson (2004)
    '''    
    #First thing is to check if the query point is closest to vertices A or B
    #check vertex a
    abx,aby,abz=cuda_vec3_minus(B,A)
    acx,acy,acz=cuda_vec3_minus(C,A)
    apx,apy,apz=cuda_vec3_minus(query,A)
    d1=cuda_dot(abx,aby,abz,apx,apy,apz)
    d2=cuda_dot(acx,acy,acz,apx,apy,apz)
    if d1<=tol and d2<=tol:
        return A[0],A[1],A[2]
    
    #check vertex b
    bpx,bpy,bpz=cuda_vec3_minus(query,B)
    d3=cuda_dot(abx,aby,abz,bpx,bpy,bpz)
    d4=cuda_dot(acx,acy,acz,bpx,bpy,bpz)
    if d3>=-tol and d4<=d3:
        return B[0],B[1],B[2]
    
    #values computed for checking A and B can be used to check if closest to edge AB
    vc=d1*d4-d3*d2
    if vc<=tol and d1>=-tol and d3<=tol:
        v=d1/(d1-d3)
        return A[0]+v*abx,A[1]+v*aby,A[2]+v*abz
    
    #now check if closest to vertex C
    cpx,cpy,cpz=cuda_vec3_minus(query,C)
    d5=cuda_dot(abx,aby,abz,cpx,cpy,cpz)
    d6=cuda_dot(acx,acy,acz,cpx,cpy,cpz)
    if d6>=-tol and d5<=d6:
        return C[0],C[1],C[2]
    
    #now check if closest to edge AC
    vb=d5*d2-d1*d6
    if vb<=tol and d2>=-tol and d6<=tol:
        w=d2/(d2-d6)
        return A[0]+w*acx,A[1]+w*acy,A[2]+w*acz

    #check if closest to edge BC
    va=d3*d6-d5*d4
    if va<=tol and (d4-d3)>=-tol and (d5-d6)>=-tol:
        w=(d4-d3)/((d4-d3)+(d5-d6))
        bcx,bcy,bcz=cuda_vec3_minus(C,B)
        return B[0]+w*bcx,B[1]+w*bcy,B[2]+w*bcz
    
    #must be inside the face
    denom=va+vb+vc
    v=vb/denom
    w=vc/denom
    return A[0]+v*abx+w*acx,A[1]+v*aby+w*acy,A[2]+v*abz+w*acz

@cuda.jit
def non_unique_closest_point_on_triangles_cuda(triangles,queries,query_pairs,output):
    '''
    compute the coordinates of the point in a triangle closest to a 3D point, when neither triangles nor queries are unique
    
    @param triangles: (n*3*3,) array of the triangle vertices; ie np.array([[A1,B1,C1]])
    @param queries: (m*3,) array of the query points
    @param query_pairs: (mk2) integer array; each entry is a point to triangle dist query where first is id of triangle, second is id of queriy point
    @param output: (k*3) array of the nearest point to query inside or on edges of triangle
    
    note that triangle and query point ids are internally converted into offsets into 1D array. 
    But they can be passed as if triangles and queries were handled as multi-D arrays
    Generalization of a direct transcription of ClosestPtPointTriangle from Real-Time Collision Detection by Christer Ericson (2004)
    then converted for CUDA using numba.cuda
    '''
    tol=1e-12
    
    #find our place in the arrays
    thread_index=cuda.grid(1)#this is the query number
    output_id=thread_index*3
    if output_id+2>=output.size:
        return
    
    triangle_id=query_pairs[thread_index*2]
    point_id=query_pairs[thread_index*2+1]
    point_index=3*point_id
    triangle_index=9*triangle_id
    aidx=triangle_index
    bidx=triangle_index+3
    cidx=triangle_index+6
    
    #First thing is to check if the query point is closest to vertices A or B
    #check vertex a
    abx,aby,abz=cuda_3d_error(triangles,bidx,triangles,aidx)
    acx,acy,acz=cuda_3d_error(triangles,cidx,triangles,aidx)
    apx,apy,apz=cuda_3d_error(queries,point_index,triangles,aidx)
    d1=cuda_dot(abx,aby,abz,apx,apy,apz)
    d2=cuda_dot(acx,acy,acz,apx,apy,apz)
    if d1<=tol and d2<=tol:
        output[output_id]=triangles[aidx]
        output[output_id+1]=triangles[aidx+1]
        output[output_id+2]=triangles[aidx+2]
        return
    
    #check vertex b
    bpx,bpy,bpz=cuda_3d_error(queries,point_index,triangles,bidx)
    d3=cuda_dot(abx,aby,abz,bpx,bpy,bpz)
    d4=cuda_dot(acx,acy,acz,bpx,bpy,bpz)
    if d3>=-tol and d4<=d3:
        output[output_id]=triangles[bidx]
        output[output_id+1]=triangles[bidx+1]
        output[output_id+2]=triangles[bidx+2]
        return
    
    #values computed for checking A and B can be used to check if closest to edge AB
    vc=d1*d4-d3*d2
    if vc<=tol and d1>=-tol and d3<=tol:
        v=d1/(d1-d3)
        output[output_id]=triangles[aidx]+v*abx
        output[output_id+1]=triangles[aidx+1]+v*aby
        output[output_id+2]=triangles[aidx+2]+v*abz
        return
    
    #now check if closest to vertex C
    cpx,cpy,cpz=cuda_3d_error(queries,point_index,triangles,cidx)
    d5=cuda_dot(abx,aby,abz,cpx,cpy,cpz)
    d6=cuda_dot(acx,acy,acz,cpx,cpy,cpz)
    if d6>=-tol and d5<=d6:
        output[output_id]=triangles[cidx]
        output[output_id+1]=triangles[cidx+1]
        output[output_id+2]=triangles[cidx+2]
        return
    
    #now check if closest to edge AC
    vb=d5*d2-d1*d6
    if vb<=tol and d2>=-tol and d6<=tol:
        w=d2/(d2-d6)
        output[output_id]=triangles[aidx]+w*acx
        output[output_id+1]=triangles[aidx+1]+w*acy
        output[output_id+2]=triangles[aidx+2]+w*acz
        return
    
    #check if closest to edge BC
    va=d3*d6-d5*d4
    if va<=tol and (d4-d3)>=-tol and (d5-d6)>=-tol:
        w=(d4-d3)/((d4-d3)+(d5-d6))
        bcx,bcy,bcz=cuda_3d_error(triangles, cidx, triangles, bidx)
        output[output_id]=triangles[bidx]+w*bcx
        output[output_id+1]=triangles[bidx+1]+w*bcy
        output[output_id+2]=triangles[bidx+2]+w*bcz
        return
    
    #must be inside the face
    denom=1/(va+vb+vc)
    v=vb*denom
    w=vc*denom
    output[output_id]=triangles[aidx]+abx*v+acx*w
    output[output_id+1]=triangles[aidx+1]+aby*v+acy*w
    output[output_id+2]=triangles[aidx+2]+abz*v+acz*w
    return

@cuda.jit
def non_unique_point_to_triangle_cuda(triangles,queries,query_pairs,output):
    '''
    compute minimum squared distance from point to a triangle when neither triangles nor queries are unique
    
    @param triangles: (n*3*3,) array of the triangle vertices; ie np.array([[A1,B1,C1]])
    @param queries: (m*3,) array of the query points
    @param query_pairs: (k*2) integer array; each entry is a point to triangle dist query where first is id of triangle, second is id of queriy point
    @param output: (k,) array of the squared distance from query to the nearest point inside or on edges of triangle
    
    note that triangle and query point ids are internally converted into offsets into 1D array. 
    But they can be passed as if triangles and queries were handled as multi-D arrays
    Generalization of a direct transcription of ClosestPtPointTriangle from Real-Time Collision Detection by Christer Ericson (2004)
    then converted for CUDA using numba.cuda
    '''
    tol=1e-12
    
    #find our place in the arrays
    thread_index=cuda.grid(1)#this is the query number
    if thread_index>=output.size:
        return
    triangle_id=query_pairs[thread_index*2]
    point_id=query_pairs[thread_index*2+1]
    point_index=3*point_id
    triangle_index=9*triangle_id
    output[thread_index]=cuda_point_to_triangle_squared_distance(triangles[triangle_index:triangle_index+9],queries[point_index:point_index+3],tol)

@njit
def cuda_vec3_minus(left,right):
    return left[0]-right[0],left[1]-right[1],left[2]-right[2]

@njit
def cuda_point_to_triangle_squared_distance(triangle,query,tol):
    '''
    compute distance from a flattened triangle to a query point

    triangle: (9,) array
    query: (3,) array
    tol: float
    '''
    A=triangle[0:3]
    B=triangle[3:6]
    C=triangle[6:9]
    return point_to_triangle_squared_distance(A,B,C,query,tol)
@njit
def point_to_triangle_squared_distance(A,B,C,query,tol):
    '''
    given vertices of a triangle and a query point compute the shortest distance
    '''
    #First thing is to check if the query point is closest to vertices A or B
    #check vertex a
    abx,aby,abz=cuda_vec3_minus(B,A)
    acx,acy,acz=cuda_vec3_minus(C,A)
    apx,apy,apz=cuda_vec3_minus(query,A)
    d1=cuda_dot(abx,aby,abz,apx,apy,apz)
    d2=cuda_dot(acx,acy,acz,apx,apy,apz)
    if d1<=tol and d2<=tol:
        #||e||^2=||ap||^2
        return apx*apx+apy*apy+apz*apz
    
    #check vertex b
    bpx,bpy,bpz=cuda_vec3_minus(query,B)
    d3=cuda_dot(abx,aby,abz,bpx,bpy,bpz)
    d4=cuda_dot(acx,acy,acz,bpx,bpy,bpz)
    if d3>=-tol and d4<=d3:
        #||e||^2=||bp||^2
        return bpx*bpx+bpy*bpy+bpz*bpz
    
    #values computed for checking A and B can be used to check if closest to edge AB
    vc=d1*d4-d3*d2
    if vc<=tol and d1>=-tol and d3<=tol:
        #||e||^2=||ap||^2-v^2||ab||^2 for v s.t. A+vAB is the nearest point
        v=d1/(d1-d3)
        return apx*apx+apy*apy+apz*apz-v*v*(abx*abx+aby*aby+abz*abz)
    
    #now check if closest to vertex C
    cpx,cpy,cpz=cuda_vec3_minus(query,C)
    d5=cuda_dot(abx,aby,abz,cpx,cpy,cpz)
    d6=cuda_dot(acx,acy,acz,cpx,cpy,cpz)
    if d6>=-tol and d5<=d6:
        #||e||^2=||cp||^2
        return cpx*cpx+cpy*cpy+cpz*cpz
    
    #now check if closest to edge AC
    vb=d5*d2-d1*d6
    if vb<=tol and d2>=-tol and d6<=tol:
        #||e||^2=||ap||^2-w^2||ac||^2 for w s.t. A+wAC is the nearest point
        w=d2/(d2-d6)
        return apx*apx+apy*apy+apz*apz-w*w*(acx*acx+acy*acy+acz*acz)
    
    #check if closest to edge BC
    va=d3*d6-d5*d4
    if va<=tol and (d4-d3)>=-tol and (d5-d6)>=-tol:
        #||e||^2=||bp||^2-w^2||bc||^2 for w s.t. B+wBC is the nearest point
        w=(d4-d3)/((d4-d3)+(d5-d6))
        bcx,bcy,bcz=cuda_vec3_minus(C,B)
        return bpx*bpx+bpy*bpy+bpz*bpz-w*w*(bcx*bcx+bcy*bcy+bcz*bcz)
    
    #must be inside the face
    denom=va+vb+vc
    v=vb/denom
    w=vc/denom
    ex=apx-v*abx-w*acx
    ey=apy-v*aby-w*acy
    ez=apz-v*abz-w*acz
    return ex*ex+ey*ey+ez*ez

@njit(parallel=False)
def multiple_points_to_triangle_squared_distance(A,B,C,queries,tol):
    n=len(queries)
    distances=np.empty(n)
    for i in prange(n):
        distances[i]=point_to_triangle_squared_distance(A,B,C,queries[i],tol)
    return distances

@njit
def points_to_triangle_max_squared_distance(A,B,C,queries,tol):
    dist=-1.0
    for query in queries:
        candidate=point_to_triangle_squared_distance(A,B,C,query,tol)
        if candidate>dist:
            dist=candidate
    return dist

def closest_point_on_triangles_trimesh(triangles, queries, tol=None):
    '''
    compute the coordinates of the point in triangle closest to 3D point for n triangle, point pairs
    
    @param triangles: (n,3,3) array of the triangle vertices; ie np.array([[A1,B1,C1]])
    @param queries: (n,3) array of the query points
    @param tol: None (default) or positive float; treat any number smaller as 0. If None, defaults to smallest normal # of triangles data type
    @return (n,3) array of the nearest point to query inside or on edges of triangle
    
    Copied from trimesh.triangles.closest_point for comparison and testing
    https://github.com/mikedh/trimesh/blob/main/trimesh/triangles.py
    Implements the method from "Real Time Collision Detection" and
    use the same variable names as "ClosestPtPointTriangle" to avoid
    being any more confusing.
    '''
    if tol is None:
        tol=np.finfo(triangles.dtype).tiny
    # store the location of the closest point
    result = np.zeros_like(queries)
    # which points still need to be handled
    remain = np.ones(len(queries), dtype=bool)

    # if we dot product this against a (n, 3)
    # it is equivalent but faster than array.sum(axis=1)
    ones = [1.0, 1.0, 1.0]

    # get the three points of each triangle
    # use the same notation as RTCD to avoid confusion
    a = triangles[:, 0, :]
    b = triangles[:, 1, :]
    c = triangles[:, 2, :]

    # check if P is in vertex region outside A
    ab = b - a
    ac = c - a
    ap = queries - a
    # this is a faster equivalent of:
    # diagonal_dot(ab, ap)
    d1 = np.dot(ab * ap, ones)
    d2 = np.dot(ac * ap, ones)

    # is the point at A
    is_a = np.logical_and(d1 < tol, d2 < tol)
    if any(is_a):
        result[is_a] = a[is_a]
        remain[is_a] = False

    # check if P in vertex region outside B
    bp = queries - b
    d3 = np.dot(ab * bp, ones)
    d4 = np.dot(ac * bp, ones)

    # do the logic check
    is_b = (d3 > -tol) & (d4 <= d3) & remain
    if any(is_b):
        result[is_b] = b[is_b]
        remain[is_b] = False

    # check if P in edge region of AB, if so return projection of P onto A
    vc = (d1 * d4) - (d3 * d2)
    is_ab = ((vc < tol) &
             (d1 > -tol) &
             (d3 < tol) & remain)
    if any(is_ab):
        v = (d1[is_ab] / (d1[is_ab] - d3[is_ab])).reshape((-1, 1))
        result[is_ab] = a[is_ab] + (v * ab[is_ab])
        remain[is_ab] = False

    # check if P in vertex region outside C
    cp = queries - c
    d5 = np.dot(ab * cp, ones)
    d6 = np.dot(ac * cp, ones)
    is_c = (d6 > -tol) & (d5 <= d6) & remain
    if any(is_c):
        result[is_c] = c[is_c]
        remain[is_c] = False

    # check if P in edge region of AC, if so return projection of P onto AC
    vb = (d5 * d2) - (d1 * d6)
    is_ac = (vb < tol) & (d2 > -tol) & (d6 < tol) & remain
    if any(is_ac):
        w = (d2[is_ac] / (d2[is_ac] - d6[is_ac])).reshape((-1, 1))
        result[is_ac] = a[is_ac] + w * ac[is_ac]
        remain[is_ac] = False

    # check if P in edge region of BC, if so return projection of P onto BC
    va = (d3 * d6) - (d5 * d4)
    is_bc = ((va < tol) &
             ((d4 - d3) > - tol) &
             ((d5 - d6) > -tol) & remain)
    if any(is_bc):
        d43 = d4[is_bc] - d3[is_bc]
        w = (d43 / (d43 + (d5[is_bc] - d6[is_bc]))).reshape((-1, 1))
        result[is_bc] = b[is_bc] + w * (c[is_bc] - b[is_bc])
        remain[is_bc] = False

    # any remaining points must be inside face region
    if any(remain):
        # point is inside face region
        denom = 1.0 / (va[remain] + vb[remain] + vc[remain])
        v = (vb[remain] * denom).reshape((-1, 1))
        w = (vc[remain] * denom).reshape((-1, 1))
        # compute Q through its barycentric coordinates
        result[remain] = a[remain] + (ab[remain] * v) + (ac[remain] * w)

    return result

@njit
def closest_point_on_triangles_numba(triangles, queries, tol=None):
    '''
    compute the coordinates of the point in triangle closest to 3D point for n triangle, point pairs
    
    @param triangles: (n,3,3) array of the triangle vertices; ie np.array([[A1,B1,C1]])
    @param queries: (n,3) array of the query points
    @param tol: None (default) or positive float; treat any number smaller as 0. If None, defaults to smallest normal # of triangles data type
    @return (n,3) array of the nearest point to query inside or on edges of triangle
    
    Implements the method from "Real Time Collision Detection" and
    use the same variable names as "ClosestPtPointTriangle" to avoid
    being any more confusing.
    
    For unclear reasons the automatic parallelization doesn't work here
    '''
    if tol is None:
        tol=np.finfo(triangles.dtype).tiny
    n=len(triangles)
    result=np.empty((n,3))
    for i in prange(n):
        result[i]=closest_point_on_triangle(triangles[i], queries[i], tol)
    return result

@njit
def closest_point_on_triangles_trimesh_numba(triangles, queries, tol=None):
    '''
    compute the coordinates of the point in triangle closest to 3D point for n triangle, point pairs
    
    @param triangles: (n,3,3) array of the triangle vertices; ie np.array([[A1,B1,C1]])
    @param queries: (n,3) array of the query points
    @param tol: None (default) or positive float; treat any number smaller as 0. If None, defaults to smallest normal # of triangles data type
    @return (n,3) array of the nearest point to query inside or on edges of triangle
    
    modified from trimesh.triangles.closest_point for numba jitting
    https://github.com/mikedh/trimesh/blob/main/trimesh/triangles.py
    Implements the method from "Real Time Collision Detection" and
    use the same variable names as "ClosestPtPointTriangle" to avoid
    being any more confusing.
    '''

    if tol is None:
        tol=np.finfo(triangles.dtype).tiny
    # store the location of the closest point
    result = np.zeros_like(queries)
    # which points still need to be handled
    remain = np.ones(len(queries), dtype=np.bool_)

    # if we dot product this against a (n, 3)
    # it is equivalent but faster than array.sum(axis=1)
    ones = np.ones((3,))

    # get the three points of each triangle
    # use the same notation as RTCD to avoid confusion
    a = triangles[:, 0, :]
    b = triangles[:, 1, :]
    c = triangles[:, 2, :]

    # check if P is in vertex region outside A
    ab = b - a
    ac = c - a
    ap = queries - a
    # this is a faster equivalent of:
    # diagonal_dot(ab, ap)
    d1 = np.dot(ab * ap, ones)
    d2 = np.dot(ac * ap, ones)

    # is the point at A
    is_a = np.logical_and(d1 < tol, d2 < tol)
    if np.any(is_a):
        result[is_a] = a[is_a]
        remain[is_a] = False

    # check if P in vertex region outside B
    bp = queries - b
    d3 = np.dot(ab * bp, ones)
    d4 = np.dot(ac * bp, ones)

    # do the logic check
    is_b = (d3 > -tol) & (d4 <= d3) & remain
    if np.any(is_b):
        result[is_b] = b[is_b]
        remain[is_b] = False

    # check if P in edge region of AB, if so return projection of P onto A
    vc = (d1 * d4) - (d3 * d2)
    is_ab = ((vc < tol) &
             (d1 > -tol) &
             (d3 < tol) & remain)
    if np.any(is_ab):
        v = (d1[is_ab] / (d1[is_ab] - d3[is_ab])).reshape((-1, 1))
        result[is_ab] = a[is_ab] + (v * ab[is_ab])
        remain[is_ab] = False

    # check if P in vertex region outside C
    cp = queries - c
    d5 = np.dot(ab * cp, ones)
    d6 = np.dot(ac * cp, ones)
    is_c = (d6 > -tol) & (d5 <= d6) & remain
    if np.any(is_c):
        result[is_c] = c[is_c]
        remain[is_c] = False

    # check if P in edge region of AC, if so return projection of P onto AC
    vb = (d5 * d2) - (d1 * d6)
    is_ac = (vb < tol) & (d2 > -tol) & (d6 < tol) & remain
    if np.any(is_ac):
        w = (d2[is_ac] / (d2[is_ac] - d6[is_ac])).reshape((-1, 1))
        result[is_ac] = a[is_ac] + w * ac[is_ac]
        remain[is_ac] = False

    # check if P in edge region of BC, if so return projection of P onto BC
    va = (d3 * d6) - (d5 * d4)
    is_bc = ((va < tol) &
             ((d4 - d3) > - tol) &
             ((d5 - d6) > -tol) & remain)
    if np.any(is_bc):
        d43 = d4[is_bc] - d3[is_bc]
        w = (d43 / (d43 + (d5[is_bc] - d6[is_bc]))).reshape((-1, 1))
        result[is_bc] = b[is_bc] + w * (c[is_bc] - b[is_bc])
        remain[is_bc] = False

    # any remaining points must be inside face region
    if np.any(remain):
        # point is inside face region
        denom = 1.0 / (va[remain] + vb[remain] + vc[remain])
        v = (vb[remain] * denom).reshape((-1, 1))
        w = (vc[remain] * denom).reshape((-1, 1))
        # compute Q through its barycentric coordinates
        result[remain] = a[remain] + (ab[remain] * v) + (ac[remain] * w)

    return result
    
def brute_force_points2mesh(points,triangles_on_device,threads_per_block=1024):
    '''
    compute distance from each of a set of points to a collection of triangles

    Parameters: points : (n,3) float array
                    the points to compute distances for
                triangles_on_device : (m*9,) float device array
                    the vertices of the triangles, as a flattened C-order device array
                threads_per_block : int or empty (default 1024)
                    # of threads to use per block (only obeyed approximately)
    Return:     distances : (n,) float array
                    the actual, guaranteed non-negative, distance from each point to the nearest triangle
    '''
    n=len(points)
    m=int(len(triangles_on_device)//9)

    #use a 2D grid, with first dim used for stepping along points and the second stepping along triangles
    tpb=int(sqrt(threads_per_block))
    blocks_per_grid_x=ceil(n/tpb)
    blocks_per_grid_y=ceil(m/tpb)
    blocks_per_grid=(blocks_per_grid_x,blocks_per_grid_y)
    kernel_signature=(blocks_per_grid,(tpb,tpb))

    distances=cuda.device_array((n,m))
    queries=cuda.to_device(points)
    brute_force_points2mesh_kernel[kernel_signature](queries,triangles_on_device,distances)
    return np.min(distances.copy_to_host(),1)

@cuda.jit
def brute_force_points2mesh_kernel(queries,triangles,distances):
    tol=1e-12
    qid,tid=cuda.grid(2)
    if qid>=len(distances) or tid*9>=len(triangles):
        return
    else:
        distances[qid,tid]=sqrt(abs(cuda_point_to_triangle_squared_distance(triangles[tid*9:tid*9+9],queries[qid],tol)))

def point2mesh_via_fcl(points,mesh,fcl_collision_mesh=None):
    if fcl_collision_mesh is None:
        bvh=trimesh.collision.mesh_to_BVH(mesh)
        fcl_collision_mesh=fcl.CollisionObject(bvh,fcl.Transform())
    sphere=fcl.Sphere(.0001)
    return np.array([fcl.distance(fcl_collision_mesh,fcl.CollisionObject(sphere,fcl.Transform(np.eye(3),pt))) for pt in points])

@njit
def brute_force_point2mesh_cpu(point,triangles):
    tol=1e-12
    return np.sqrt(min([point_to_triangle_squared_distance(t[0],t[1],t[2],point,tol) for t in triangles]))

@njit(parallel=False)
def tri_to_points_squared_hausdorff(triangles,points):
    '''
    one sided hausdorff (squared) distance from triangles to points; i.e. maximize over points and minimze over triangles
    '''
    tol=1e-12
    ntri=len(triangles)
    npts=len(points)
    dist=np.empty(ntri)
    for i in prange(ntri):
        tri=triangles[i]
        dist[i]=np.max(multiple_points_to_triangle_squared_distance(tri[0],tri[1],tri[2],points,tol))
    best_tri=np.argmin(dist)
    return dist[best_tri],best_tri

def gpu_tris_to_points_squared_hausdorff(triangle_ids,triangle_vertices_on_device,points,threads_per_block=1024):
    '''
    compute squared distance from some triangles to some points, then maximize over points and minimize over triangles using GPU

    Parameters: triangle_ids : m entry list or array of integers
                    the indices of the triangles to test
                triangle_vertices_on_device : (n,3,3) float device array (32bit for best performance)
                    the vertices of all triangles, to be indexed by entries of triangle_ids
                points : (k,3) float array (cast to 32 bit for GPU)
                    the query points to test
                threads_per_block : int, default 1024
                    # of threads to launch per block
    Returns: 32 bit distance minimized over triangles and maximized over points
    '''
    m=len(triangle_vertices_on_device)

    blocks_per_grid=(m+(threads_per_block-1))//threads_per_block
    kernel_signature=(blocks_per_grid,threads_per_block)

    distances=cuda.device_array(m,np.float32)
    queries=cuda.to_device(points.astype(np.float32))
    kernel_tris_to_points_max_squared_distance[kernel_signature](cuda.to_device(triangle_ids),triangle_vertices_on_device,queries,distances)
    return min_reduce(distances)

@cuda.jit
def kernel_tris_to_points_max_squared_distance(triangle_ids,triangle_vertices_on_device,points,distances):
    tidx=cuda.grid(1)
    if tidx>=len(triangle_ids) or triangle_ids[tidx]>=len(triangle_vertices_on_device):
        return
    triangle_id=triangle_ids[tidx]
    distances[tidx]=points_to_triangle_max_squared_distance(triangle_vertices_on_device[triangle_id,0],triangle_vertices_on_device[triangle_id,1],triangle_vertices_on_device[triangle_id,2],points,1e-12)
    
@cuda.reduce
def min_reduce(a,b):
    return min(a,b)
