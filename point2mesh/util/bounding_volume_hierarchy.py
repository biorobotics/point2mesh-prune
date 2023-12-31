from collections import namedtuple
from math import sqrt,fabs
import numpy as np
from numba import njit,prange
from numba.typed import List

from point2mesh.util.Math import stable_triangle_area,normsquared
from point2mesh.util import priority_queue,triangle_geometry

from point2mesh.triangle_mesh import point_to_triangle_squared_distance,brute_force_point2mesh_cpu

from typing import Iterable
from typing import List as AnnotationList
from numpy.typing import ArrayLike

rss=namedtuple("RSS",["rotation_matrix","center","half_lengths","radius"])
rssnode=namedtuple("rssnode",["triangles","parent","child_plus","child_minus","centroid","axes","RSS","depth","is_leaf"])

@njit
def set_children(node:rssnode,child_plus:int,child_minus:int):
    is_leaf=child_plus<0 and child_minus<0
    return rssnode(node.triangles,node.parent,child_plus,child_minus,node.centroid,node.axes,node.RSS,node.depth,is_leaf)
@njit
def set_leaf(node:rssnode,is_leaf:bool):
    return rssnode(node.triangles,node.parent,node.child_plus,node.child_minus,node.centroid,node.axes,node.RSS,node.depth,is_leaf)

@njit
def construct_rss_tree(triangle_ids:Iterable[int],triangle_vertices:ArrayLike,max_depth=np.inf,leaf_size=1)->AnnotationList[rssnode]:
    '''
    build a binary tree of Rectangular Swept Spheres top down following the approach of Li, Shellshear, Bohlin, and Carlson 2020

    Parameters: triangle_ids : iterable of int
                    the indices of the triangles to include in the hierarchy, length<=n
                triangle_vertices : (n,3,3) float array
                    the vertices of all n triangles
                max_depth : int or inf (default)
                    don't split nodes with depth>=max_depth (root has depth 1!)
                leaf_size : int or 1 (default)
                    stop splitting when the set of triangles assigned is =<leaf_size
    Return:     rsstree : numba List of rssnode
                    the nodes of the rss tree
    '''
    local_triangles=triangle_vertices[triangle_ids]
    center,split_axes=get_center_and_split_axes(local_triangles)
    root_rss=fit_RSS(local_triangles,split_axes)
    node=rssnode(triangle_ids,-1,-1,-1,center,split_axes,root_rss,1,1>=max_depth or len(triangle_ids)<=leaf_size)
    split_stack=List([(node,True)])#list of tuples: rssnode, True if is child_plus (False if is child_minus, None if no parent)
    nodes=List()
    while len(split_stack)>0:
        node,is_child_plus=split_stack.pop()
        next_node_index=len(nodes)
        
        if not node.is_leaf and node.depth<max_depth and len(node.triangles)>leaf_size:
            child_plus,child_minus,success=IPS_split_rule(next_node_index,node,triangle_vertices,leaf_size)
            if success:
                split_stack.append((child_minus,False))
                split_stack.append((child_plus,True))
            else:
                #no split found
                node=set_leaf(node,True)
        else:
            node=set_leaf(node,True)
        nodes.append(node)
        if node.parent>=0:
            parent=nodes[node.parent]
            if is_child_plus:
                nodes[node.parent]=set_children(parent,next_node_index,parent.child_minus)
            else:
                nodes[node.parent]=set_children(parent,parent.child_plus,next_node_index)
    return List(nodes)

@njit
def get_center_and_split_axes(triangles:ArrayLike):
    '''
    pick a point and axes to split a group of triangles along using the inertia properties following Li 2020 Sec IV

    Parameters: triangles : (n,3,3) float
                    array of triangle vertices
    Returns:    centroid : (3,) float
                    coordinates of the centroid of the triangles
                eigvecs : (3,3) float
                    three unit eigenvectors of the scaled inertia tensor, in decreasing order of eigenvalue
    '''
    n=len(triangles)
    areas=np.array([stable_triangle_area(tri) for tri in triangles])
    Amin=np.min(areas)
    centroid=np.sum(np.sum(triangles,1)*np.expand_dims(areas,1),0)/np.sum(areas)/3
    #scaled inertia tensor; scaling doesn't affect eigendirections
    I=triangle_inertia_tensor_contrib(triangles[0],areas[0]/Amin,centroid)
    for i in range(1,n):
        I+=triangle_inertia_tensor_contrib(triangles[i],areas[i]/Amin,centroid)
    w,v=np.linalg.eig(I)
    order=np.argsort(w)[::-1]
    eigvecs=v[order]
    return centroid,eigvecs

@njit
def triangle_inertia_tensor_contrib(triangle:ArrayLike,area:float,centroid:ArrayLike):
    vertex_moment_arms=triangle-centroid
    I=np.empty((3,3),dtype=triangle.dtype)
    for j in range(3):
        for k in range(3):
            I[j,k]=np.sum(vertex_moment_arms[:,j]*vertex_moment_arms[:,k])
    return area*I

@njit
def fit_RSS(triangles:ArrayLike,axes:ArrayLike):
    '''
    fits a rectangular swept sphere volume to triangles using given axes, following the approach in PQP (see PQP/src/BV.cpp)

    Parameters: triangles : (n,3,3) float
                    array of triangle vertices to fit an RSS to
                axes : (3,3) float
                    three unit eigenvectors in decreasing order of eigenvalue
    '''
    n=len(triangles)
    #create rotation matrix following PQP/src/Build.cpp/build_recurse
    R=np.empty((3,3))
    R[:,0]=axes[0]
    R[:,1]=axes[1]
    R[0,2]=axes[0,1]*axes[1,2]-axes[1,1]*axes[0,2]
    R[1,2]=axes[1][0]*axes[0,2]-axes[0,0]*axes[1,2]
    R[2,2]=axes[0,0]*axes[1,1]-axes[1,0]*axes[0,1]
    #rotate vertices to orientation of the RSS
    points=triangles.reshape((n*3,3))@R#this takes the matrix-vector product of R.T and each entry of triangles to produce an n*3,3 array 

    #compute rectangle thickness in local z
    minz=np.min(points[:,2])
    maxz=np.max(points[:,2])
    r=(maxz-minz)/2
    radsqr=r*r
    cz=(minz+maxz)/2

    #get x,y bounds
    minx,maxx=get_range(points,cz,radsqr,0)
    miny,maxy=get_range(points,cz,radsqr,1)

    #grow lengths if there are uncovered points near the corners
    for point in points:
        if point[0]>maxx:
            if point[1]>maxy:
                maxx,maxy=extend_corners(point,maxx,maxy,cz,radsqr,True,True)
            elif point[1]<miny:
                maxx,miny=extend_corners(point,maxx,miny,cz,radsqr,True,False)
        elif point[0]<minx:
            if point[1]>maxy:
                minx,maxy=extend_corners(point,minx,maxy,cz,radsqr,False,True)
            elif point[1]<miny:
                minx,miny=extend_corners(point,minx,miny,cz,radsqr,False,False)
    c=np.array([(maxx+minx)/2,(maxy+miny)/2,cz])
    rectangle_center=R@c
    rectangle_sides=np.array([maxx-minx,maxy-miny])
    rectangle_sides[rectangle_sides<0]=0

    return rss(R.T.copy(),rectangle_center,rectangle_sides/2,r)


@njit
def get_range(points:ArrayLike,cz:float,radsqr:float,axis_id:int):
    #compute an initial length of rectangle along local direction
    minidx=np.argmin(points[:,axis_id])
    maxidx=np.argmax(points[:,axis_id])
    dz=points[minidx,2]-cz
    minx=points[minidx,axis_id]+sqrt(max(radsqr-dz*dz,0))
    dz=points[maxidx,2]-cz
    maxx=points[maxidx,axis_id]-sqrt(max(radsqr-dz*dz,0))

    #expand along local x until all points contained
    for pt in points:
        if pt[axis_id]<minx:
            dz=pt[2]-cz
            x=pt[axis_id]+sqrt(max(radsqr-dz*dz,0))
            if x<minx:
                minx=x
        elif pt[axis_id]>maxx:
            dz=pt[2]-cz
            x=pt[axis_id]-sqrt(max(radsqr-dz*dz,0))
            if x>maxx:
                maxx=x
    return minx,maxx

@njit       
def extend_corners(point:ArrayLike,xbound:float,ybound:float,cz:float,radsqr:float,xup:bool,yup:bool):
    a=sqrt(0.5)
    if xup:
        xdir=1
    else:
        xdir=-1
    if yup:
        ydir=1
    else:
        ydir=-1
    dx=point[0]-xbound
    dy=point[1]-ybound
    u=xdir*dx*a+ydir*dy*a
    t=(xdir*a*u-dx)*(xdir*a*u-dx)+(ydir*a*u-dy)*(ydir*a*u-dy)+(cz-point[2])*(cz-point[2])
    u-=sqrt(max(radsqr-t,0))
    if u>0:
        xbound+=xdir*u*a
        ybound+=ydir*u*a
    return xbound,ybound

@njit
def IPS_split_rule(parent_idx:int,parent_node:rssnode,triangle_vertices:ArrayLike,leaf_size:int):
    triangle_indices=parent_node.triangles
    n=len(triangle_indices)
    triangle_centroid_diff=np.sum(triangle_vertices[triangle_indices],1)/3-parent_node.centroid#n,3
    best_cost=np.inf
    for ax in parent_node.axes:
        not_split=False
        plus=[]
        minus=[]
        projections=triangle_centroid_diff@ax
        for j in range(n):
            if projections[j]>=0:
                plus.append(triangle_indices[j])
            else:
                minus.append(triangle_indices[j])
        #fit RSS to the candidate split
        plus=np.array(plus,dtype=np.int64)
        minus=np.array(minus,dtype=np.int64)
        plus_tri=triangle_vertices[plus]
        if len(plus_tri)>0:
            plus_centroid,plus_axes=get_center_and_split_axes(plus_tri)
            plus_rss=fit_RSS(plus_tri,plus_axes)
        else:
            plus_centroid=np.full(3,np.nan)
            plus_axes=np.full((3,3),np.nan)
            plus_rss=rss(np.eye(3),np.full(3,np.nan),np.zeros(3),0.0)
            not_split=True

        minus_tri=triangle_vertices[minus]
        if len(minus_tri)>0:
            minus_centroid,minus_axes=get_center_and_split_axes(minus_tri)
            minus_rss=fit_RSS(minus_tri,minus_axes)
        else:
            minus_centroid=np.full(3,np.nan)
            minus_axes=np.full((3,3),np.nan)
            minus_rss=rss(np.eye(3),np.full(3,np.nan),np.zeros(3),0.0)
            not_split=True
        if not_split:
            cost=np.inf
        else:
            cost=rss_surface_area(plus_rss)+rss_surface_area(minus_rss)
        if cost<=best_cost:#<= ensures that if all axes put all tris in one or the other, things still get populated
            best_cost=cost
            plus_idx=plus
            minus_idx=minus

            plus_centroid_star=plus_centroid
            plus_axes_star=plus_axes
            plus_rss_star=plus_rss

            minus_centroid_star=minus_centroid
            minus_axes_star=minus_axes
            minus_rss_star=minus_rss
    plus_leaf=len(plus_idx)<=leaf_size or len(minus_idx)==0
    minus_leaf=len(minus_idx)<=leaf_size or len(plus_idx)==0
    plus_node=rssnode(plus_idx,parent_idx,-1,-1,plus_centroid_star,plus_axes_star,plus_rss_star,parent_node.depth+1,plus_leaf)
    minus_node=rssnode(minus_idx,parent_idx,-1,-1,minus_centroid_star,minus_axes_star,minus_rss_star,parent_node.depth+1,minus_leaf)
    return plus_node,minus_node,best_cost<np.inf

@njit
def rss_surface_area(rss_query:rss)->float:
    '''
    compute the surface area of a specified RSS
    '''
    #faces contribute twice the surface area of underlying rectangle
    x,y=rss_query.half_lengths*2
    faces_sa=2*x*y
    #corners add the surface area of a sphere of appropriate radius
    sphere_sa=4*np.pi*rss_query.radius*rss_query.radius
    #edges add the surface area of 2 cylinders of appropriate radius and height matching the side lengths
    cylinder_sa=np.pi*rss_query.radius*rss_query.radius*(x+y)
    return faces_sa+sphere_sa+cylinder_sa

@njit
def point_rss_squared_distance(point:ArrayLike,rss_query:rss)->float:
    '''
    compute the minimum squared distance from a point to a specified RSS (0 if inside)

    finds distance to the 3D rectangle and subtracts the radius, returning 0 if negative
    based on SqDistPointOBB in sec 5.1.5.1 of Ericson, Christer. "Real-Time Collision Detection," CRC Press 2004.
    "rotation_matrix","center","side_lengths","radius"
    '''
    #find rectangle distance
    v=point-rss_query.center
    local_v=rss_query.rotation_matrix@v
    #z part
    sqDist=local_v[2]*local_v[2]
    #clamp x and y parts
    for i in range(2):
        project_on_local=local_v[i]
        e=rss_query.half_lengths[i]
        if project_on_local<-e:
            excess=project_on_local+e
        elif project_on_local>e:
            excess=project_on_local-e
        else:
            excess=0.0
        sqDist+=excess*excess
    #sqrt and remove radius
    actual_dist=sqrt(sqDist)-rss_query.radius
    if actual_dist<0:
        return 0.0
    else:
        return actual_dist*actual_dist
    
@njit
def aligned_box_rss_distance(side_lengths:ArrayLike,center:ArrayLike,rss_query:rss):
    '''
    finds distance from an axis-aligned box to a specified RSS (negative if intersecting)

    Parameters: side_lengths : (3,) float array
                    length of box in each axis
                center : (3,) float array
                    center of box
                rss_query : rss
                    the RSS to test against
    Return:     distance:
                    the distance between the two (negative if intersecting)
    '''
    #first, find the point on the plane of the RSS's rectangle that is closest to the box
    normal=rss_query.rotation_matrix[2]#this vector is perpendicular to the rectangle of the RSS
    rss_center_in_canonical_frame=rss_query.center-center#a point in the rectangle, now in a frame that is // to world frame and centered on the center of the box

    pt_on_plane_in_canonical,pt_on_box_in_canonical=triangle_geometry.closest_point_on_plane_to_canonical_box(normal,rss_center_in_canonical_frame,side_lengths)

    #determine if the point on the rectangle plane is in the rectangle or not
    v=pt_on_plane_in_canonical-rss_center_in_canonical_frame
    local_v=rss_query.rotation_matrix@v
    in_rectangle=True
    for i in range(2):
        project_on_local=local_v[i]
        e=rss_query.half_lengths[i]
        if project_on_local<-e or project_on_local>e:
            in_rectangle=False
    if in_rectangle:
        #closest point on RSS to box is at rss_query.radius along the vector from the point on the plane to the point on the box
        actual_dist=np.linalg.norm(pt_on_box_in_canonical-pt_on_plane_in_canonical)-rss_query.radius
    else:
        #closest point on RSS to box is at rss_query.radius along the vector from a point on one of the edges of the rectangle to a point on the box
        e0_in_world=rss_query.half_lengths[0]*rss_query.rotation_matrix[0]
        e1_in_world=rss_query.half_lengths[1]*rss_query.rotation_matrix[1]
        corners_in_canonical=[rss_center_in_canonical_frame+e0_in_world+e1_in_world,
                              rss_center_in_canonical_frame+e0_in_world-e1_in_world,
                              rss_center_in_canonical_frame-e0_in_world-e1_in_world,
                              rss_center_in_canonical_frame-e0_in_world+e1_in_world]
        sqrDistance=np.inf
        i0=3
        for i1 in range(4):
            candidate_closest_on_segment,candidate_closest_on_box=triangle_geometry.closest_point_segment_canonical_box(corners_in_canonical[i0],corners_in_canonical[i1],side_lengths)
            newDistSquared=normsquared(candidate_closest_on_segment-candidate_closest_on_box)
            if newDistSquared<sqrDistance:
                sqrDistance=newDistSquared
            i0=i1
        actual_dist=sqrt(sqrDistance)-rss_query.radius
    return actual_dist

@njit
def get_aligned_box_distance_queue(side_lengths:ArrayLike,center:ArrayLike,rsstree:AnnotationList[rss],triangle_vertices:ArrayLike):
    '''
    use RSStree and priority queue to find distance from an axis-aligned box to a triangle mesh

    Parameters: side_lengths : (3,) float array
                    length of box in each axis
                center : (3,) float array
                    center of box
                rsstree : numba.typed.List of rssnodes
                    the RSStree data structure
                triangle_vertices : (n,3,3) float array
                    the vertices of each triangle in the mesh
    Returns:    distance : float
                    the distance from query to mesh
                count : int
                    number of triangles tested
                best_triangle : int
                    index of triangle that was closest
    '''
    return get_aligned_box_distance_with_bound(side_lengths,center,rsstree,triangle_vertices,np.inf)

@njit
def get_aligned_box_distance_with_bound(side_lengths:ArrayLike,center:ArrayLike,rsstree:AnnotationList[rss],triangle_vertices:ArrayLike,upper_bound_squared:float):
    '''
    use RSStree and priority queue to find distance from an axis-aligned box to a triangle mesh

    Parameters: side_lengths : (3,) float array
                    length of box in each axis
                center : (3,) float array
                    center of box
                rsstree : numba.typed.List of rssnodes
                    the RSStree data structure
                triangle_vertices : (n,3,3) float array
                    the vertices of each triangle in the mesh
                upper_bound_squared : float
                    a value known to be no smaller than the squared distance from the box
    Returns:    distance : float
                    the distance from query to mesh
                count : int
                    number of triangles tested
                best_triangle : int
                    index of triangle that was closest. Negative if no triangle within upper_bound_squared.
    '''
    best_triangle=-1
    best_distance=sqrt(upper_bound_squared)
    count=0
    queue=priority_queue.make_queue(0,-np.inf)
    while not priority_queue.is_empty(queue):
        queue,lower_bound,nodeid=priority_queue.get_item(queue)
        node=rsstree[nodeid]
        if best_distance<lower_bound:
            #nothing left in the queue that can do better
            break
        if not node.is_leaf:
            #we need to process child_plus if it exists and the bounding volume is closer to query than our best distance yet found
            if node.child_plus>=0:
                distance_to_plus_rss=aligned_box_rss_distance(side_lengths,center,rsstree[node.child_plus].RSS)
                if distance_to_plus_rss<best_distance:
                    queue=priority_queue.add_item(queue,node.child_plus,distance_to_plus_rss)
            if node.child_minus>=0:
                distance_to_minus_rss=aligned_box_rss_distance(side_lengths,center,rsstree[node.child_minus].RSS)
                if distance_to_minus_rss<best_distance:
                    queue=priority_queue.add_item(queue,node.child_minus,distance_to_minus_rss)
        else:
            #leaf node, which may be empty
            #get a candidate distance as the nearest triangle in the leaf node
            for triangle_id in node.triangles:
                triangle=triangle_vertices[triangle_id]
                tri_dist=sqrt(triangle_geometry.squared_distance_triangle_to_aligned_box(triangle,side_lengths,center))
                count+=1
                if tri_dist<best_distance:
                    best_distance=tri_dist
                    best_triangle=triangle_id
    return best_distance,count,best_triangle

@njit
def is_aligned_box_distance_lte(side_lengths,center,rsstree,triangle_vertices,upper_bound):
    '''
    returns if the distance from an axis aligned box to the mesh is <= a provided upper bound

    Parameters: side_lengths : (3,) float array
                    length of box in each axis
                center : (3,) float array
                    center of box
                rsstree : numba.typed.List of rssnodes
                    the RSStree data structure
                triangle_vertices : (n,3,3) float array
                    the vertices of each triangle in the mesh
                upper_bound : float
                    the distance to see if the true distance is <=
    Returns:    answer : bool
                    True if squared distance is <= upper_bound_squared
                tighter_upper_bound : float
                    an upper bound for the distance. If answer is False, this equals upper_bound and should not be relied upon
                count : int
                    the number of point to triangle distance tests conducted
    '''
    best_distance=upper_bound
    count=0
    queue=priority_queue.make_queue(0,0.0)
    while not priority_queue.is_empty(queue):
        queue,lower_bound,nodeid=priority_queue.get_item(queue)
        node=rsstree[nodeid]
        if best_distance<lower_bound:
            #nothing left in the queue that can do better
            break
        if not node.is_leaf:
            #we need to process child_plus if it exists and the bounding volume is closer to query than our best distance yet found
            if node.child_plus>=0:
                distance_to_plus_rss=aligned_box_rss_distance(side_lengths,center,rsstree[node.child_plus].RSS)
                if distance_to_plus_rss<best_distance:
                    queue=priority_queue.add_item(queue,node.child_plus,distance_to_plus_rss)
            if node.child_minus>=0:
                distance_to_minus_rss=aligned_box_rss_distance(side_lengths,center,rsstree[node.child_minus].RSS)
                if distance_to_minus_rss<best_distance:
                    queue=priority_queue.add_item(queue,node.child_minus,distance_to_minus_rss)
        else:
            #leaf node, which may be empty
            #get a candidate distance as the nearest triangle in the leaf node
            for triangle_id in node.triangles:
                triangle=triangle_vertices[triangle_id]
                tri_dist=sqrt(triangle_geometry.squared_distance_triangle_to_aligned_box(triangle,side_lengths,center))
                count+=1
                if tri_dist<=best_distance:
                    return True,tri_dist,count
    return False,best_distance,count

@njit
def aligned_box_get_nearby_triangles(side_lengths,center,rsstree,triangle_vertices,upper_bound):
    '''
    returns all the triangles within some distance of an axis-aligned box

    Parameters: side_lengths : (3,) float array
                    length of box in each axis
                center : (3,) float array
                    center of box
                rsstree : numba.typed.List of rssnodes
                    the RSStree data structure
                triangle_vertices : (n,3,3) float array
                    the vertices of each triangle in the mesh
                upper_bound : float
                    the distance to collect triangles within
    Returns:    triangle_ids : numba.typed.List of ints
                    indices of the triangles within upper_bound of the axis_aligned box
                distances : numba.typed.List of float
                    distances to the triangles in triangle_ids
                count : int
                    the number of point to triangle distance tests conducted
    '''
    triangle_ids=list()
    distances=list()
    queue=List([(-np.inf,0)])
    count=0
    while len(queue)>0:
        lower_bound,nodeid=queue.pop()
        node=rsstree[nodeid]
        if lower_bound<upper_bound:
            if not node.is_leaf:
                #we need to process child_plus if it exists and the bounding volume is closer to query than our upper_bound
                if node.child_plus>=0:
                    distance_to_plus_rss=aligned_box_rss_distance(side_lengths,center,rsstree[node.child_plus].RSS)
                    if distance_to_plus_rss<upper_bound:
                        queue.append((distance_to_plus_rss,node.child_plus))
                if node.child_minus>=0:
                    distance_to_minus_rss=aligned_box_rss_distance(side_lengths,center,rsstree[node.child_minus].RSS)
                    if distance_to_minus_rss<upper_bound:
                        queue.append((distance_to_minus_rss,node.child_minus))
            else:
                #leaf node, which may be empty
                #get a candidate distance as the nearest triangle in the leaf node
                for triangle_id in node.triangles:
                    triangle=triangle_vertices[triangle_id]
                    tri_dist=sqrt(triangle_geometry.squared_distance_triangle_to_aligned_box(triangle,side_lengths,center))
                    count+=1
                    if tri_dist<=upper_bound:
                        triangle_ids.append(triangle_id)
                        distances.append(tri_dist)
    return triangle_ids,distances,count

@njit
def get_distance_queue(query:ArrayLike,rsstree:AnnotationList[rssnode],triangle_vertices:ArrayLike):
    '''
    compute distance query to the mesh

    Parameters: query : (3,) float array
                    the point to test
                rsstree : numba.typed.List of rssnodes
                    the RSStree data structure
                triangle_vertices : (n,3,3) float array
                    the vertices of each triangle in the mesh
    Returns:    distance : float
                    the distance from query to mesh
                count : int
                    number of triangles tested
    '''
    return get_distance_with_bound(query,rsstree,triangle_vertices,np.inf)

@njit
def get_distance_with_bound(query:ArrayLike,rsstree:AnnotationList[rssnode],triangle_vertices:ArrayLike,upper_bound_squared:float):
    '''
    compute distance query to the mesh, assumes distance is below some passed in number

    Parameters: query : (3,) float array
                    the point to test
                rsstree : numba.typed.List of rssnodes
                    the RSStree data structure
                triangle_vertices : (n,3,3) float array
                    the vertices of each triangle in the mesh
                upper_bound_squared : float
                    the squared distance the true distance is <=
    Returns:    distance : float
                    the distance from query to mesh, unless upper_bound_squared was wrong in which case this is sqrt(upper_bound_squared)
                count : int
                    number of triangles tested
    '''
    best_distance_squared=upper_bound_squared
    count=0
    queue=priority_queue.make_queue(0,0.0)
    while not priority_queue.is_empty(queue):
        queue,lower_bound,nodeid=priority_queue.get_item(queue)
        node=rsstree[nodeid]
        if best_distance_squared<lower_bound:
            #nothing left in the queue that can do better
            break
        if not node.is_leaf:
            #we need to process child_plus if it exists and the bounding volume is closer to query than our best distance yet found
            if node.child_plus>=0:
                distance_to_plus_rss=point_rss_squared_distance(query,rsstree[node.child_plus].RSS)
                if distance_to_plus_rss<best_distance_squared:
                    queue=priority_queue.add_item(queue,node.child_plus,distance_to_plus_rss)
            if node.child_minus>=0:
                distance_to_minus_rss=point_rss_squared_distance(query,rsstree[node.child_minus].RSS)
                if distance_to_minus_rss<best_distance_squared:
                    queue=priority_queue.add_item(queue,node.child_minus,distance_to_minus_rss)
        else:
            #leaf node, which may be empty
            #get a candidate distance as the nearest triangle in the leaf node
            tol=1e-12
            for triangle_id in node.triangles:
                triangle=triangle_vertices[triangle_id]
                tri_dist_squared=fabs(point_to_triangle_squared_distance(triangle[0],triangle[1],triangle[2],query,tol))
                count+=1
                if tri_dist_squared<best_distance_squared:
                    best_distance_squared=tri_dist_squared
    return sqrt(best_distance_squared),count

@njit
def is_distance_lte(query,rsstree,triangle_vertices,upper_bound_squared):
    '''
    returns if the distance from a query to the mesh is <= a provided upper bound

    Parameters: query : (3,) float array
                    the point to test
                rsstree : numba.typed.List of rssnodes
                    the RSStree data structure
                triangle_vertices : (n,3,3) float array
                    the vertices of each triangle in the mesh
                upper_bound_squared : float
                    the squared distance to see if the true distance is <=
    Returns:    answer : bool
                    True if squared distance is <= upper_bound_squared
                tighter_upper_bound : float
                    an upper bound for the squared distance. If answer is False, this equals upper_bound_squared and should not be relied upon
                count : int
                    the number of point to triangle distance tests conducted
    '''
    best_distance_squared=upper_bound_squared
    count=0
    queue=priority_queue.make_queue(0,0.0)
    while not priority_queue.is_empty(queue):
        queue,lower_bound,nodeid=priority_queue.get_item(queue)
        node=rsstree[nodeid]
        if best_distance_squared<lower_bound:
            #nothing left in the queue that can do better
            break
        if not node.is_leaf:
            #we need to process child_plus if it exists and the bounding volume is closer to query than our best distance yet found
            if node.child_plus>=0:
                distance_to_plus_rss=point_rss_squared_distance(query,rsstree[node.child_plus].RSS)
                if distance_to_plus_rss<best_distance_squared:
                    queue=priority_queue.add_item(queue,node.child_plus,distance_to_plus_rss)
            if node.child_minus>=0:
                distance_to_minus_rss=point_rss_squared_distance(query,rsstree[node.child_minus].RSS)
                if distance_to_minus_rss<best_distance_squared:
                    queue=priority_queue.add_item(queue,node.child_minus,distance_to_minus_rss)
        else:
            #leaf node, which may be empty
            #get a candidate distance as the nearest triangle in the leaf node
            tol=1e-12
            for triangle_id in node.triangles:
                triangle=triangle_vertices[triangle_id]
                tri_dist_squared=fabs(point_to_triangle_squared_distance(triangle[0],triangle[1],triangle[2],query,tol))
                count+=1
                if tri_dist_squared<=best_distance_squared:
                    return True,best_distance_squared,count
    return False,best_distance_squared,count

@njit
def point2mesh_via_rss_serial(points,triangle_vertices,rsstree):
    '''
    compute distance query to the mesh

    Parameters: points : (n,3) float array
                    the points to test
                triangle_vertices : (n,3,3) float array
                    the vertices of each triangle in the mesh
                rsstree : numba.typed.List of rssnodes
                    the RSStree data structure
    Returns:    distances : (n,) float array
                    the distance from query to mesh
                counts : (n,) int array
                    number of triangles tested
    '''
    distances=np.empty(len(points))
    counts=np.empty(len(points),dtype=np.int64)
    for i,point in enumerate(points):
        distances[i],counts[i]=get_distance_queue(point,rsstree,triangle_vertices)
    return distances,counts

def point2mesh_via_rss_and_kdtree(points,triangle_vertices,rsstree,kdtree):
    '''
    compute distance query to the mesh, assumes distance is below some passed in number. Tries to use a kdtree to get a bound on each distance before using rsstree.

    Parameters: points : (n,3) float array
                    the points to test
                triangle_vertices : (n,3,3) float array
                    the vertices of each triangle in the mesh
                rsstree : numba.typed.List of rssnodes
                    the RSStree data structure
                kdtree : scipy.spatial.KDTree
                    KDTree of the mesh vertices
    Returns:    distances : (n,) float array
                    the distance from query to mesh
                counts : (n,) int array
                    number of triangles tested
    '''
    distances=np.empty(len(points))
    counts=np.empty(len(points),dtype=np.int64)
    distances=kdtree.query(points)[0]
    distances*=distances#squared distance from each query point to a point on the mesh
    for i,point in enumerate(points):
        distances[i],counts[i]=get_distance_with_bound(point,rsstree,triangle_vertices,distances[i])
    return distances,counts

@njit(parallel=True)
def verify_rsstree_distance(test_points,triangle_vertices,rsstree):
    '''
    compute distance query to the mesh using brute force and RSStree, then compare

    Parameters: test_points : (n,3) float array
                    the points to test
                triangle_vertices : (n,3,3) float array
                    the vertices of each triangle in the mesh
                rsstree : numba.typed.List of rssnodes
                    the RSStree data structure
    Returns:    errors : (n,) float array
                    the absolute difference in distance from RSStree and brute force
    '''
    errors=np.empty(len(test_points))
    for i in prange(len(test_points)):
        linked_dist,_=get_distance_queue(test_points[i],rsstree,triangle_vertices)
        exhaustive_distance=brute_force_point2mesh_cpu(test_points[i],triangle_vertices)
        errors[i]=np.abs(linked_dist-exhaustive_distance)
    return errors