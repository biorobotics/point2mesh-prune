'''
implementation of a Binary Space Partition Tree (BSP-tree) following Ch 12 of
Mark de Berg, Marc van Kreveld, Mark Overmars, Otfried Schwarzkopf. "Computational geometry: algorithms and applications" 2nd-revised edition (2000)
'''
from point2mesh.util import priority_queue
from collections import namedtuple
from math import sqrt,fabs,floor

import numpy as np
from numba import njit,prange
from numba.typed import List,Dict

from point2mesh.triangle_mesh import point_to_triangle_squared_distance


bspnode=namedtuple("bspnode",["triangles","parent","child_plus","child_minus","point","normal","depth","is_leaf"])

@njit
def is_split_useful(triangles_above,triangles_below,total_triangles):
    labove=len(triangles_above)
    lbelow=len(triangles_below)
    #just peeling off the split triangle is not useful, so above must include at least 2 and not every
    #must be at least one triangle sent below
    #and below should not include every triangle except the split
    return 1<labove and labove<total_triangles and 0<lbelow and lbelow<total_triangles-1

@njit
def split_on_first(triangle_ids,triangle_vertex_array,arg,free_splits):
    t1=triangle_vertex_array[triangle_ids[0]]
    AB=t1[1]-t1[0]
    AC=t1[2]-t1[0]
    normal=np.cross(AB,AC)
    normal/=np.linalg.norm(normal)
    pt=t1[0]
    signs=np.array([np.dot(triangle_vertex_array[tid]-pt,normal) for tid in triangle_ids])
    triangles_above,triangles_below=get_split(0,triangle_ids,triangle_vertex_array,signs)
    success=is_split_useful(triangles_above,triangles_below,len(triangle_ids))
    return triangles_above,triangles_below,pt,normal,success,free_splits

@njit
def split_on_first_helpful(triangle_ids,triangle_vertex_array,arg,free_splits):
    total=len(triangle_ids)
    signs=np.empty((total,3),triangle_vertex_array.dtype)#will be completely populated if the current triangle is a free split
    for idx in range(total):
        t1=triangle_vertex_array[triangle_ids[idx]]
        AB=t1[1]-t1[0]
        AC=t1[2]-t1[0]
        normal=np.cross(AB,AC)
        normal/=np.linalg.norm(normal)
        pt=t1[0]
        for i,tid in enumerate(triangle_ids):
            signs[i]=np.dot(triangle_vertex_array[tid]-pt,normal)
        triangles_above,triangles_below=get_split(idx,triangle_ids,triangle_vertex_array,signs)
        if is_split_useful(triangles_above,triangles_below,total):
            return triangles_above,triangles_below,pt,normal,True,free_splits
    return split_on_first(triangle_ids,triangle_vertex_array,arg,free_splits)#return a split on the first triangle if none of the triangles actually divide things

@njit
def split_random(triangle_ids,triangle_vertex_array,rng,free_splits):
    '''
    split on a random triangle
    '''
    total=len(triangle_ids)
    left_to_try=total
    choices=[i for i in range(total)]
    signs=np.empty((total,3),triangle_vertex_array.dtype)#will be completely populated if the current triangle is a free split
    while left_to_try>0:
        randf=rng.random()
        random_choice=floor(abs(randf)*left_to_try)
        if random_choice<0:
            random_choice=0
        elif random_choice>=left_to_try:
            random_choice=left_to_try-1
        idx=choices[random_choice]
        t1=triangle_vertex_array[triangle_ids[idx]]
        AB=t1[1]-t1[0]
        AC=t1[2]-t1[0]
        normal=np.cross(AB,AC)
        pt=t1[0].copy()
        for i,tid in enumerate(triangle_ids):
            signs[i]=np.dot(triangle_vertex_array[tid]-pt,normal)

        triangles_above,triangles_below=get_split(idx,triangle_ids,triangle_vertex_array,signs)
        if is_split_useful(triangles_above,triangles_below,total):
            normal/=np.linalg.norm(normal)
            return triangles_above,triangles_below,pt,normal,True,free_splits
        else:
            left_to_try-=1
            choices=choices[:random_choice]+choices[random_choice+1:]
    normal/=np.linalg.norm(normal)
    return triangles_above,triangles_below,pt,normal,False,free_splits

@njit
def split_free(triangle_ids,triangle_vertex_array,rng,free_splits):
    '''
    check every triangle to see if it is a free split, returning the first such found. Give a random split if no free splits found
    '''
    total=len(triangle_ids)
    signs=np.empty((total,3),triangle_vertex_array.dtype)#will be completely populated if the current triangle is a free split
    for idx in range(total):
        t1=triangle_vertex_array[triangle_ids[idx]]
        AB=t1[1]-t1[0]
        AC=t1[2]-t1[0]
        normal=np.cross(AB,AC)
        pt=t1[0].copy()
        for i in range(total):
            free=False
            signs[i]=np.dot(triangle_vertex_array[triangle_ids[i]]-pt,normal)
            if i!=idx:
                if np.all(signs[i]>=0) or np.all(signs[i]<=0):
                    #all vertices of this triangle are to one side
                    free=True
            else:
                free=True
            if not free:
                #found a triangle that we would have to put in both partitions
                break
        if free:
            triangles_above,triangles_below=get_split(idx,triangle_ids,triangle_vertex_array,signs)
            if is_split_useful(triangles_above,triangles_below,total):
                normal/=np.linalg.norm(normal)
                return triangles_above,triangles_below,pt,normal,True,free_splits
    return split_random(triangle_ids,triangle_vertex_array,rng,free_splits)

@njit
def split_free_cache(triangle_ids,triangle_vertex_array,rng,useless_splits):
    '''
    check every triangle to see if it is a free split, returning the first such found. Give a random split if no free splits found
    '''
    child_useless_splits=useless_splits#a copy will be made before useless_splits is mutated
    total=len(triangle_ids)
    signs=np.empty((total,3),triangle_vertex_array.dtype)#will be completely populated if the current triangle is a free split
    for idx in range(total):
        free=False
        splitting_triangle_index=triangle_ids[idx]
        # if useless_splits[splitting_triangle_index]:
        #     continue
        t1=triangle_vertex_array[splitting_triangle_index]
        AB=t1[1]-t1[0]
        AC=t1[2]-t1[0]
        normal=np.cross(AB,AC)
        pt=t1[0].copy()
        for i in range(total):
            free=False
            signs[i]=np.dot(triangle_vertex_array[splitting_triangle_index]-pt,normal)
            if i!=idx:
                if np.all(signs[i]>=0) or np.all(signs[i]<=0):
                    #all vertices of this triangle are to one side
                    free=True
            else:
                free=True
            if not free:
                #found a triangle that we would have to put in both partitions
                break
        if free:
            triangles_above,triangles_below=get_split(idx,triangle_ids,triangle_vertex_array,signs)
            if is_split_useful(triangles_above,triangles_below,total):
                normal/=np.linalg.norm(normal)
                # if child_useless_splits is None:
                #     #no changes were made so no point in copying
                #     child_useless_splits=useless_splits
                return triangles_above,triangles_below,pt,normal,True,child_useless_splits
            else:
                pass
                # if child_useless_splits is None:
                #     #a new useless split exists at this node and below so make a copy and indicate the useless split
                #     child_useless_splits=useless_splits.copy()
                # child_useless_splits[splitting_triangle_index]=True
    # if child_useless_splits is None:
    #     #no changes were made so no point in copying
    #     child_useless_splits=useless_splits
    ta,tb,pt,normal,success,_=split_random(triangle_ids,triangle_vertex_array,rng,child_useless_splits)
    return ta,tb,pt,normal,success,child_useless_splits

@njit
def split_free_track_intersections(triangle_ids,triangle_vertex_array,rng,all_intersections):
    '''
    check every triangle to see if it is a free split, returning the first such found. Give a random split if no free splits found
    Does not work because numba does not currently support sets in dictionaries
    '''
    total=len(triangle_ids)
    signs=np.empty((total,3),triangle_vertex_array.dtype)#will be completely populated if the current triangle is a free split
    for idx in range(total):
        free=True
        for tidx in triangle_vertex_array:
            if tidx in all_intersections[triangle_ids[idx]]:
                #still an intersecting triangle
                free=False
                break
        if free:
            #free split!
            t1=triangle_vertex_array[triangle_ids[idx]]
            AB=t1[1]-t1[0]
            AC=t1[2]-t1[0]
            normal=np.cross(AB,AC)
            pt=t1[0].copy()
            for i,tid in enumerate(triangle_ids):
                signs[i]=np.dot(triangle_vertex_array[tid]-pt,normal)
            triangles_above,triangles_below=get_split(idx,triangle_ids,triangle_vertex_array,signs)
            if is_split_useful(triangles_above,triangles_below,total):
                normal/=np.linalg.norm(normal)
                return triangles_above,triangles_below,pt,normal,True
    return split_random(triangle_ids,triangle_vertex_array,rng)

@njit
def get_intersecting_triangles(idx,triangle_ids,triangle_vertex_array):
    t1=triangle_vertex_array[triangle_ids[idx]]
    AB=t1[1]-t1[0]
    AC=t1[2]-t1[0]
    normal=np.cross(AB,AC)
    pt=t1[0].copy()
    total=len(triangle_ids)
    intersections=set()
    for i in range(total):
        signs=np.dot(triangle_vertex_array[triangle_ids[i]]-pt,normal)
        if i!=idx:
            if np.any(signs<0) and np.any(signs>0):
                #some vertices of this triangle are to each side
                intersections.add(triangle_ids[i])
    return intersections

@njit
def get_all_intersections(triangle_ids,triangle_vertex_array):
    all_intersections=Dict()
    all_intersections[-1]=set((1,))#for typing purposes
    for i,idx in enumerate(triangle_ids):
        all_intersections[idx]=get_intersecting_triangles(i,triangle_ids,triangle_vertex_array)
    return all_intersections

@njit
def is_split_free(idx,triangle_ids,triangle_vertex_array):
    t1=triangle_vertex_array[triangle_ids[idx]]
    AB=t1[1]-t1[0]
    AC=t1[2]-t1[0]
    normal=np.cross(AB,AC)
    pt=t1[0].copy()
    total=len(triangle_ids)
    for i in range(total):
        free=False
        signs=np.dot(triangle_vertex_array[triangle_ids[i]]-pt,normal)
        if i!=idx:
            if np.all(signs>=0) or np.all(signs<=0):
                #all vertices of this triangle are to one side
                free=True
        else:
            free=True
        if not free:
            #found a triangle that we would have to put in both partitions
            return False
    return True

@njit
def get_signs(idx,triangle_ids,triangle_vertex_array):
    total=len(triangle_ids)
    signs=np.empty((total,3),triangle_vertex_array.dtype)
    t1=triangle_vertex_array[triangle_ids[idx]]
    AB=t1[1]-t1[0]
    AC=t1[2]-t1[0]
    normal=np.cross(AB,AC)
    pt=t1[0].copy()
    for i in range(total):
        signs[i]=np.dot(triangle_vertex_array[triangle_ids[i]]-pt,normal)
    return signs

@njit
def get_split(idx,triangle_ids,triangle_vertex_array,signs=None):
    total=len(triangle_ids)
    if signs is None:
        signs=get_signs(idx,triangle_ids,triangle_vertex_array)
    #for our application we don't want fragments, so any intersected triangles will get assigned to both partitions
    triangles_above=List([triangle_ids[i] for i in range(total) if i!=idx and np.any(signs[i]>0)])#skip the splitting triangle and anything coplanar
    triangles_below=List([triangle_ids[i] for i in range(total) if i!=idx and np.any(signs[i]<0)])#skip the splitting triangle

    #assign coplanar tris to the above partition
    for i in range(total):
        if i!=idx:
            if np.all(signs[i]==0):
                triangles_above.append(i)

    #we can assign the splitting triangle to just one partition. Put it at the back so we don't try to split with it again if we call split_on_first
    triangles_above.append(triangle_ids[idx])
    return triangles_above,triangles_below

@njit
def construct(split_function,split_arg,triangles,max_depth,leaf_size):
    '''
    construct binary splits of passed triangles according to specified splitting rule, up to a maximum depth and/or leaf size

    Parameters: split_function : function handle: list of triangles indices,array of all triangle vertices, split_arg->first child tris, second child tris, plane pt, plane normal
                    function tests if should be split and returns the split, or all in first child tris w/ None for plane pt and normal
                split_arg : anything
                     passed to split_function as third argument
                triangles : n,3,3 np array of triangles corners
                    the set of triangles to build a tree for
                max_depth : int or None
                    if not None, don't split nodes with depth>=max_depth (root has depth 1!)
                leaf_size : int or None
                    if not None, stop splitting when the set of triangles assigned is =<leaf_size
    '''
    n=len(triangles)
    return construct_on_subset(split_function,split_arg,triangles,range(n),max_depth,leaf_size)

@njit
def construct_on_subset(split_function,split_arg,triangles,indices_to_use,max_depth,leaf_size):
    if max_depth is None:
        max_depth=np.inf
    if leaf_size is None:
        leaf_size=1
    useless_splits=np.zeros(len(triangles),dtype=np.bool_)
    split_stack=List([(List(indices_to_use),-1,True,1,useless_splits)])#list of tuples: list of triangle idxs, index of parent node in nodes, True if is child_plus (False if is child_minus, None if no parent), depth of this node,set of guaranteed free splits
    nodes=[]
    while len(split_stack)>0:
        triangle_collection,parent_index,is_child_plus,depth,free_splits=split_stack.pop()
        next_node_index=len(nodes)
        if depth<max_depth and len(triangle_collection)>leaf_size:
            triangles_above,triangles_below,pt,normal,success,child_free_splits=split_function(triangle_collection,triangles,split_arg,free_splits)
            is_leaf=True
            if success and len(triangles_below)>0:
                split_stack.append((triangles_below,next_node_index,False,depth+1,child_free_splits))
                is_leaf=False
            if success and len(triangles_above)>0:
                split_stack.append((triangles_above,next_node_index,True,depth+1,child_free_splits))
                is_leaf=False
        else:
            #leaf node has undefined splitting plane
            pt=np.full(3,np.nan)
            normal=np.full(3,np.nan)
            is_leaf=True
            
        nodes.append(bspnode(triangle_collection,parent_index,-1,-1,pt,normal,depth,is_leaf))
        if parent_index>=0:
            parent=nodes[parent_index]
            if is_child_plus:
                nodes[parent_index]=bspnode(parent.triangles,parent.parent,next_node_index,parent.child_minus,parent.point,parent.normal,parent.depth,False)
            else:
                nodes[parent_index]=bspnode(parent.triangles,parent.parent,parent.child_plus,next_node_index,parent.point,parent.normal,parent.depth,False)
    return List(nodes)

@njit
def get_distance_recursive(query,node,bspnodes,triangle_vertices,best_distance=np.inf,distance_cache=None,count=0):
    if distance_cache is None:
        distance_cache=np.full(len(triangle_vertices),-1.0)
    #descend the tree
    if not node.is_leaf:
        #default to the non-empty child if only one exists
        if node.child_plus<0:
            if node.child_minus>=0:
                next_node=bspnodes[node.child_minus]
                other_child=None
            else:
                #both children are empty, but node.point is defined. This should never happen
                raise ValueError("A node has two empty children but is not marked leaf")
        elif node.child_minus<0:
            next_node=bspnodes[node.child_plus]
            other_child=None
        else:
            #both children are nonempty, so step into the one whose side of the plane we are on
            normal_distance=np.dot(query-node.point,node.normal)
            if normal_distance>=0:
                next_node=bspnodes[node.child_plus]
                other_child=bspnodes[node.child_minus]
            else:
                next_node=bspnodes[node.child_minus]
                other_child=bspnodes[node.child_plus]
        candidate,count=get_distance_recursive(query,next_node,bspnodes,triangle_vertices,best_distance,distance_cache,count)
        best_distance=min(best_distance,candidate)
        #check if other child node has a smaller distance
        if other_child is not None:
            if best_distance>normal_distance:
                candidate,count=get_distance_recursive(query,other_child,bspnodes,triangle_vertices,best_distance,distance_cache,count)
                best_distance=min(best_distance,candidate)
        return best_distance,count
    else:
        #get a candidate distance as the nearest triangle in the leaf node
        tol=1e-12
        mindistsquared=best_distance**2
        for triangle_id in node.triangles:
            if distance_cache[triangle_id]<0:
                triangle=triangle_vertices[triangle_id]
                distance_cache[triangle_id]=fabs(point_to_triangle_squared_distance(triangle[0],triangle[1],triangle[2],query,tol))
                count+=1
            if distance_cache[triangle_id]<mindistsquared:
                mindistsquared=distance_cache[triangle_id]
        return sqrt(mindistsquared),count

@njit
def get_distance_leaf(query,node,triangle_vertices):
    tol=1e-12
    mindistsquared=np.inf
    for triangle_id in node.triangles:
        triangle=triangle_vertices[triangle_id]
        candidate_dist_squared=fabs(point_to_triangle_squared_distance(triangle[0],triangle[1],triangle[2],query,tol))
        if candidate_dist_squared<mindistsquared:
            mindistsquared=candidate_dist_squared
    return sqrt(mindistsquared)

@njit
def get_distance_queue(query,bspnodes,triangle_vertices):
    best_distance_squared=np.inf
    distance_cache=np.full(len(triangle_vertices),-1.0)
    best_distance_squared,count=get_distance_squared(query,bspnodes,triangle_vertices,distance_cache,best_distance_squared)
    return sqrt(best_distance_squared),count

@njit
def get_distance_squared(query,bspnodes,triangle_vertices,distance_cache,best_distance_squared):
    queue=priority_queue.make_queue(0,0.0)
    count=0
    while not priority_queue.is_empty(queue):
        queue,lower_bound,nodeid=priority_queue.get_item(queue)
        node=bspnodes[nodeid]
        if best_distance_squared<lower_bound:
            #nothing left in the queue that can do better
            break
        if not node.is_leaf:
            normal_distance=np.dot(query-node.point,node.normal)
            if normal_distance>=0:
                next_node=node.child_plus
                other_child=node.child_minus
            else:
                next_node=node.child_minus
                other_child=node.child_plus
            #triangles in either side of the partition can't be closer than the parent allowed, so lower_bound is the smallest legal priority
            queue=priority_queue.add_item(queue,next_node,lower_bound)
            new_lower_bound=normal_distance**2
            if new_lower_bound<best_distance_squared:
                #we need to add both children
                queue=priority_queue.add_item(queue,other_child,max(lower_bound,new_lower_bound))
        else:
            #leaf node, which may be empty
            #get a candidate distance as the nearest triangle in the leaf node
            tol=1e-12
            for triangle_id in node.triangles:
                if distance_cache[triangle_id]<0:
                    triangle=triangle_vertices[triangle_id]
                    distance_cache[triangle_id]=fabs(point_to_triangle_squared_distance(triangle[0],triangle[1],triangle[2],query,tol))
                    count+=1
                if distance_cache[triangle_id]<best_distance_squared:
                    best_distance_squared=distance_cache[triangle_id]
    return best_distance_squared,count

@njit
def distance_record_traversal(query,bspnodes,triangle_vertices):
    node_ids=[]
    best_distance_squared=np.inf
    distance_cache=np.full(len(triangle_vertices),-1.0)
    count=0
    queue=priority_queue.make_queue(0,0.0)
    while not priority_queue.is_empty(queue):
        queue,lower_bound,nodeid=priority_queue.get_item(queue)
        node_ids.append(nodeid)
        node=bspnodes[nodeid]
        if best_distance_squared<lower_bound:
            #nothing left in the queue that can do better
            break
        if not node.is_leaf:
            normal_distance=np.dot(query-node.point,node.normal)
            if normal_distance>=0:
                next_node=node.child_plus
                other_child=node.child_minus
            else:
                next_node=node.child_minus
                other_child=node.child_plus
            #triangles in either side of the partition can't be closer than the parent allowed, so lower_bound is the smallest legal priority
            if next_node>=0:
                queue=priority_queue.add_item(queue,next_node,lower_bound)
            if other_child>=0:
                new_lower_bound=normal_distance**2
                if new_lower_bound<best_distance_squared:
                    #we need to add both children
                    queue=priority_queue.add_item(queue,other_child,max(lower_bound,new_lower_bound))
        else:
            #leaf node, which may be empty
            #get a candidate distance as the nearest triangle in the leaf node
            tol=1e-12
            for triangle_id in node.triangles:
                if distance_cache[triangle_id]<0:
                    triangle=triangle_vertices[triangle_id]
                    distance_cache[triangle_id]=fabs(point_to_triangle_squared_distance(triangle[0],triangle[1],triangle[2],query,tol))
                    count+=1
                if distance_cache[triangle_id]<best_distance_squared:
                    best_distance_squared=distance_cache[triangle_id]
    return sqrt(best_distance_squared),count,node_ids

@njit
def point2mesh_via_bsp_serial(points,triangle_vertices,bspnodes):
    distances=np.empty(len(points))
    counts=np.empty(len(points),dtype=np.int64)
    for i,point in enumerate(points):
        distances[i],counts[i]=get_distance_queue(point,bspnodes,triangle_vertices)
    return distances,counts

@njit(parallel=True)
def verify_bsp_distance(test_points,bspnodes,triangle_iterable):
    '''
    compare the result of the bsp distance calc and brute force on specified test points

    test_points: (n,3) float array of query points
    bspnodes: list of bspnode
    triangle_iterable: iterable of the triangles partitioned by the bsp tree
    '''
    errors=np.empty(len(test_points))
    for i in prange(len(test_points)):
        bspdist,count=get_distance_recursive(test_points[i],bspnodes[0],bspnodes,triangle_iterable)
        exhaustive_distance=sqrt(min([point_to_triangle_squared_distance(t[0],t[1],t[2],test_points[i],1e-12) for t in triangle_iterable]))
        errors[i]=np.abs(bspdist-exhaustive_distance)
    return errors

@njit(parallel=True)
def verify_queue_distance(test_points,bspnodes,triangle_iterable):
    '''
    compare the result of the queue-based bsp distance calc and brute force on specified test points

    test_points: (n,3) float array of query points
    bspnodes: list of bspnode
    triangle_iterable: iterable of the triangles partitioned by the bsp tree
    '''
    errors=np.empty(len(test_points))
    for i in prange(len(test_points)):
        bspdist,count=get_distance_queue(test_points[i],bspnodes,triangle_iterable)
        exhaustive_distance=sqrt(min([point_to_triangle_squared_distance(t[0],t[1],t[2],test_points[i],1e-12) for t in triangle_iterable]))
        errors[i]=np.abs(bspdist-exhaustive_distance)
    return errors