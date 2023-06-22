from math import fabs,sqrt

from multiprocessing import Pool

import timeit

import numpy as np

from numba import njit,cuda,set_parallel_chunksize
from numba.types import float64 as nb_float64
from numba.types import int64 as nb_int64

import trimesh

try:
    import pysdf
except ImportError:
    pass
try:
    from CGAL.CGAL_Kernel import Point_3,Triangle_3
    from CGAL.CGAL_AABB_tree import AABB_tree_Triangle_3_soup
except ImportError:
    pass
from point2mesh.voxelized.construction import make_pruned_point_to_mesh_ragged_array
from point2mesh.voxelized.extended_linked_voxel_structure import make_linked_voxels_ragged_array,point2mesh_via_linked_cpu_serial,make_voxel_bsptrees,point2mesh_via_linked_bsp_cpu_serial
from point2mesh.triangle_mesh import cuda_point_to_triangle_squared_distance
from point2mesh.util import binary_space_partition_tree,bounding_volume_hierarchy
from point2mesh.util.ArrayTricks import threeD_multi_to_1D
from point2mesh.voxelized.distance import point2mesh_distance_cpu,point2mesh_via_gpu_kernel,point2mesh_via_cpu_serial,point2mesh_via_cpu_parallel
from point2mesh.triangle_mesh import brute_force_points2mesh
load_mesh=trimesh.load_mesh

def draw_samples(mesh,expansion_factor,n_query):
    #scale bounding box
    box_transform=mesh.bounding_box.transform
    box_size=expansion_factor*mesh.bounding_box.extents
    big_box=trimesh.primitives.Box(extents=box_size,transform=box_transform)
    #sample points
    if isinstance(n_query,int):
        n_query=[n_query]
    return [big_box.sample_volume(nq) for nq in n_query]

def trial(n_query,n_runs,spacing,mesh_path,expansion_factor,to_do=None,chunk_size=1):
    if to_do is None:
        to_do=["CGAL","trimesh","pysdf","Serial CPU Linked No BSP","Serial CPU Linked With BSP","Serial CPU GVD","Parallel CPU GVD","More Parallel CPU GVD","32bit GPU GVD","64bit GPU GVD","32bit GPU brute force"]
    mesh=load_mesh(mesh_path)
    query_points=draw_samples(mesh,expansion_factor,n_query)
    timing=[dict() for _ in range(len(n_query))]
    counts=[dict() for _ in range(len(n_query))]
    #CGAL
    if "CGAL" in to_do:
        print("Setting up CGAL")
        tree=CGAL_initialize(mesh)
        run_multiple_sizes(query_points,CGAL_query_preprocess,CGAL_test,(tree,),n_runs,"CGAL",timing)

    #trimesh
    if "trimesh" in to_do:
        run_multiple_sizes(query_points,no_preprocess,trimesh_test,(mesh,),n_runs,"trimesh",timing)

    #pysdf
    if "pysdf" in to_do:
        print("Setting up pysdf")
        sdf=pysdf_initialize(mesh)
        run_multiple_sizes(query_points,no_preprocess,pysdf_test,(sdf,),n_runs,"pysdf",timing)

    for key in to_do:
        case="bsp d="
        if key[:len(case)]==case:
            depth=int(key[len(case):])
            CPU_bsp_depth_trials(query_points,mesh,depth,n_runs,timing,counts)

    for key in to_do:
        case="bsp l="
        if key[:len(case)]==case:
            leaf=int(key[len(case):])
            CPU_bsp_leaf_trials(query_points,mesh,leaf,n_runs,timing,counts)

    for key in to_do:
        case="rss d="
        if key[:len(case)]==case:
            depth=int(key[len(case):])
            CPU_rss_depth_trials(query_points,mesh,depth,n_runs,timing,counts)

    for key in to_do:
        case="rss l="
        if key[:len(case)]==case:
            leaf=int(key[len(case):])
            CPU_rss_leaf_trials(query_points,mesh,leaf,n_runs,timing,counts)

    #Linked Voxel structure from Hauth, Murtezaoglu, and Linsen 2009 https://doi.org/10.1016/j.cad.2009.06.007
    if "Serial CPU Linked No BSP" in to_do or "Serial CPU Linked With BSP" in to_do:
        print("Setting up CPU Linked Voxel")
        ct,vt,lv,vv,m,dw=CPU_linked_voxel_initialize(spacing,mesh,expansion_factor)
        if "Serial CPU Linked No BSP" in to_do:
            #compile
            CPU_serial_linked_voxel_test(query_points[0],mesh,ct,vt,lv,vv,m,spacing,dw)
            #loop
            run_multiple_sizes(query_points,no_preprocess,CPU_serial_linked_voxel_test,(mesh,ct,vt,lv,vv,m,spacing,dw),n_runs,"Serial CPU Linked No BSP",timing,counts)
        if "Serial CPU Linked With BSP" in to_do:
            print("Setting up BSP trees")
            bsptrees=make_voxel_bsptrees(10,np.random.default_rng(0),vt,ct,mesh.triangles)
            #compile
            CPU_serial_linked_voxel_bsp_test(query_points[0],mesh,bsptrees,lv,vv,m,spacing,dw)
            #loop
            run_multiple_sizes(query_points,no_preprocess,CPU_serial_linked_voxel_bsp_test,(mesh,bsptrees,lv,vv,m,spacing,dw),n_runs,"Serial CPU Linked With BSP",timing,counts)
            print("Cleaning up BSP trees")
            del bsptrees
        print("Cleaning up CPU Linked Voxel")
        del ct,vt,lv,vv,m,dw

    #serial CPU
    if "Serial CPU GVD" in to_do or "Parallel CPU GVD" in to_do or "More Parallel CPU GVD" in to_do:
        print("Setting up CPU GVD")
        ct,vt,m,dw=CPU_GVD_initialize(spacing,mesh,expansion_factor)
        if "Serial CPU GVD" in to_do:
            #compile
            CPU_serial_GVD_test(query_points[0],mesh,ct,vt,m,spacing,dw)
            #loop
            run_multiple_sizes(query_points,no_preprocess,CPU_serial_GVD_test,(mesh,ct,vt,m,spacing,dw),n_runs,"Serial CPU GVD",timing,counts)
        if "Parallel CPU GVD" in to_do:
            #compile
            CPU_parallel_GVD_test(query_points[0],mesh,ct,vt,m,spacing,dw)
            #loop
            run_multiple_sizes(query_points,no_preprocess,CPU_parallel_GVD_test,(mesh,ct,vt,m,spacing,dw),n_runs,"Parallel CPU GVD",timing,counts)
        if "More Parallel CPU GVD" in to_do:
            #compile
            CPU_more_parallel_GVD_test(query_points[0],mesh,ct,vt,m,spacing,dw,chunk_size)
            #loop
            run_multiple_sizes(query_points,no_preprocess,CPU_more_parallel_GVD_test,(mesh,ct,vt,m,spacing,dw),n_runs,"More Parallel CPU GVD",timing)
        print("Cleaning up CPU GVD")
        del ct,vt,m,dw

    #32bit GPU
    if "32bit GPU GVD" in to_do:
        print("Setting up 32bit GPU GVD")
        spacing32,ct32,vt32,m32,dw32,tri32=GPU_32bit_GVD_initialize(spacing,mesh,expansion_factor)
        #compile
        GPU_32bit_GVD_test(query_points[0],tri32,512,ct32,vt32,m32,spacing32,dw32)
        #loop
        run_multiple_sizes(query_points,no_preprocess,GPU_32bit_GVD_test(tri32,512,ct32,vt32,m32,spacing32,dw32),n_runs,"32bit GPU GVD",timing)
        print("Cleaning up 32bit GPU GVD")
        del spacing32,ct32,vt32,m32,dw32,tri32
    #64bit GPU
    if "64bit GPU GVD" in to_do:
        print("Setting up 64bit GPU GVD")
        spacing64,ct64,vt64,m64,dw64,tri64=GPU_64bit_GVD_initialize(spacing,mesh,expansion_factor)
        #compile
        GPU_64bit_GVD_test(query_points[0],tri64,512,ct64,vt64,m64,spacing64,dw64)
        #loop
        run_multiple_sizes(query_points,no_preprocess,GPU_64bit_GVD_test,(tri64,512,ct64,vt64,m64,spacing64,dw64),n_runs,"64bit GPU GVD",timing)
        print("Cleaning up 64bit GPU GVD")
        del spacing64,ct64,vt64,m64,dw64,tri64
    #32bit brute force GPU
    if "32bit GPU brute force" in to_do:
        print("Setting up 32 bit GPU brute force")
        triangles32=GPU_32bit_brute_force_initialize(mesh)
        #compile
        GPU_32bit_brute_force_test(query_points[0],triangles32,512)
        #loop
        run_multiple_sizes(query_points,no_preprocess,GPU_32bit_brute_force_test,(triangles32,512),n_runs,"32bit GPU brute force",timing)
    #averages
    print("Average durations")
    for i,query in enumerate(query_points):
        print(str(len(query))+" points")
        for name in timing[i]:
            print_avg(timing[i][name],name)
    print("Average triangles tested")
    for i,query in enumerate(query_points):
        print(str(len(query))+" points")
        for name in counts[i]:
            print_avg(counts[i][name],name)
    return timing,counts

def print_avg(times,name):
    print(name+": "+str(np.mean(times)))
def run_and_time(distance_func,points,args,n_runs,name,timing=None,counts=None):
    print("Looping "+name+" "+str(n_runs)+" times")
    times=[]
    count_list=[]
    for _ in range(n_runs):
        s=timeit.default_timer()
        if counts is not None:
            _,count_array=distance_func(points,*args)
        else:
            distance_func(points,*args)
            count_array=None
        e=timeit.default_timer()
        times.append(e-s)
        print(str(times[-1])+" s")
        if count_array is not None:
            count_list.append(np.mean(count_array))
    if isinstance(timing,dict):
        timing[name]=times
    if counts is not None:
        counts[name]=count_list
    return times,counts

def run_multiple_sizes(query_points,preprocessor,distance_func,args,n_runs,name,timing,counts=None):
    for i,query in enumerate(query_points):
        print(name+" on "+str(len(query))+" points")
        preprocessed=preprocessor(query)
        if counts is None:
            count=None
        else:
            count=counts[i]
        _=run_and_time(distance_func,preprocessed,args,n_runs,name,timing[i],count)

def no_preprocess(points):
    return points

def CGAL_initialize(mesh):
    triangles=[]
    v=mesh.vertices
    for face in mesh.faces:
        corners=tuple(Point_3(v[f][0],v[f][1],v[f][2]) for f in face)
        triangles.append(Triangle_3(*corners))
    tree=AABB_tree_Triangle_3_soup(triangles)
    tree.accelerate_distance_queries()
    return tree

def CGAL_query_preprocess(query_points):
    return [Point_3(*qp) for qp in query_points]

def CGAL_test(cgal_points,tree):
    return [sqrt(tree.squared_distance(pt)) for pt in cgal_points]

def trimesh_initialize():
    pass

def trimesh_test(points,mesh):
    return trimesh.proximity.closest_point(mesh,points)[1]

def pysdf_initialize(mesh):
    return pysdf.SDF(mesh.vertices,mesh.faces,False)

def pysdf_test(points,sdf):
    return sdf(points)

def CPU_linked_voxel_initialize(spacing,mesh,expansion_factor):
    candidate_triangles,voxel2triangles,linked_voxels,voxel2voxels,minimums,domain_widths=make_linked_voxels_ragged_array(spacing,[mesh],expansion_factor,True,False,True)
    return candidate_triangles[0],voxel2triangles[0],linked_voxels[0],voxel2voxels[0],minimums[0],domain_widths[0]

def CPU_serial_linked_voxel_test(points,mesh,candidate_triangles,voxel2triangles,linked_voxels,voxel2voxels,minimums,spacing,domain_widths):
    return point2mesh_via_linked_cpu_serial(points,mesh.triangles.reshape((len(mesh.triangles)*9,)),voxel2triangles,candidate_triangles,voxel2voxels,linked_voxels,minimums,spacing,domain_widths)

def CPU_serial_linked_voxel_bsp_test(points,mesh,bsptrees,linked_voxels,voxel2voxels,minimums,spacing,domain_widths):
    return point2mesh_via_linked_bsp_cpu_serial(points,mesh.triangles,bsptrees,voxel2voxels,linked_voxels,minimums,spacing,domain_widths)

def CPU_bsp_depth_trials(query_points,mesh,depth,n_runs,timing,counts):
    print("Setting up BSP tree d="+str(depth))
    bspnodes=binary_space_partition_tree.construct(binary_space_partition_tree.split_random,np.random.default_rng(0),mesh.triangles,depth,1)
    #compile
    CPU_serial_bsp_test(query_points[0],mesh,bspnodes)
    #loop
    run_multiple_sizes(query_points,no_preprocess,CPU_serial_bsp_test,(mesh,bspnodes),n_runs,"bsp d="+str(depth),timing,counts)
    print("Cleaning up BSP tree d="+str(depth))
    del bspnodes

def CPU_bsp_leaf_trials(query_points,mesh,leaf,n_runs,timing,counts):
    print("Setting up BSP tree l="+str(leaf))
    bspnodes=binary_space_partition_tree.construct(binary_space_partition_tree.split_random,np.random.default_rng(0),mesh.triangles,None,leaf)
    #compile
    CPU_serial_bsp_test(query_points[0],mesh,bspnodes)
    #loop
    run_multiple_sizes(query_points,no_preprocess,CPU_serial_bsp_test,(mesh,bspnodes),n_runs,"bsp l="+str(leaf),timing,counts)
    print("Cleaning up BSP tree l="+str(leaf))
    del bspnodes

def CPU_serial_bsp_test(points,mesh,bspnodes):
    return binary_space_partition_tree.point2mesh_via_bsp_serial(points,mesh.triangles,bspnodes)

def CPU_rss_depth_trials(query_points,mesh,depth,n_runs,timing,counts):
    print("Setting up RSS tree d="+str(depth))
    rsstree=bounding_volume_hierarchy.construct_rss_tree(np.arange(len(mesh.triangles)),mesh.triangles,max_depth=depth)
    #compile
    CPU_serial_rss_test(query_points[0],mesh,rsstree)
    #loop
    run_multiple_sizes(query_points,no_preprocess,CPU_serial_rss_test,(mesh,rsstree),n_runs,"rss d="+str(depth),timing,counts)
    print("Cleaning up RSS tree d="+str(depth))
    del rsstree

def CPU_rss_leaf_trials(query_points,mesh,leaf,n_runs,timing,counts):
    print("Setting up RSS tree l="+str(leaf))
    rsstree=bounding_volume_hierarchy.construct_rss_tree(np.arange(len(mesh.triangles)),mesh.triangles,leaf_size=leaf)
    #compile
    CPU_serial_rss_test(query_points[0],mesh,rsstree)
    #loop
    run_multiple_sizes(query_points,no_preprocess,CPU_serial_rss_test,(mesh,rsstree),n_runs,"rss l="+str(leaf),timing,counts)
    print("Cleaning up RSS tree l="+str(leaf))
    del rsstree

def CPU_serial_rss_test(points,mesh,rsstree):
    return bounding_volume_hierarchy.point2mesh_via_rss_serial(points,mesh.triangles,rsstree)

def CPU_GVD_initialize(spacing,mesh,expansion_factor):
    candidate_triangles,voxel2triangles,minimums,domain_widths=make_pruned_point_to_mesh_ragged_array(spacing,[mesh],expansion_factor)
    return candidate_triangles[0],voxel2triangles[0],minimums[0],domain_widths[0]

def CPU_serial_GVD_test(points,mesh,candidate_triangles,voxel2triangles,minimums,spacing,domain_widths):
    return point2mesh_via_cpu_serial(points,mesh.triangles.reshape((len(mesh.triangles)*9,)),voxel2triangles,candidate_triangles,minimums,spacing,domain_widths)

def CPU_parallel_GVD_test(points,mesh,candidate_triangles,voxel2triangles,minimums,spacing,domain_widths):
    return point2mesh_via_cpu_parallel(points,mesh.triangles.reshape((len(mesh.triangles)*9,)),voxel2triangles,candidate_triangles,minimums,spacing,domain_widths)

def CPU_more_parallel_GVD_test(points,mesh,candidate_triangles,voxel2triangles,minimums,spacing,domain_widths,chunk_size):
    return point2mesh_via_cpu_more_parallel(points,mesh.triangles.reshape((len(mesh.triangles)*9,)),voxel2triangles,candidate_triangles,minimums,spacing,domain_widths,chunk_size)

def CPU_pool_initialize(mesh, candidate_triangles,voxels2triangle,minimums,spacing,domain_width,nworkers):
    triangles=mesh.triangles.reshape((len(mesh.triangles)*9,))
    tol=1e-12
    return Pool(nworkers,initializer=point2mesh_cpu_pool_initializer,initargs=(triangles,voxels2triangle,candidate_triangles,minimums,spacing,domain_width,tol))

def CPU_pool_test(points,pool):
    return pool.map(point2mesh_cpu_pool_worker,points)

def GPU_32bit_GVD_initialize(spacing,mesh,expansion_factor):
    candidate_triangles,voxel2triangles,minimums,domain_widths=make_pruned_point_to_mesh_ragged_array(spacing,[mesh],expansion_factor)
    candidates_gpu=cuda.to_device(candidate_triangles[0].astype(np.int64))
    voxel2triangles_gpu=cuda.to_device(voxel2triangles[0].astype(np.int64))
    triangles=cuda.to_device(mesh.triangles.reshape((len(mesh.triangles)*9,)).astype(np.float32))
    return np.float32(spacing),candidates_gpu,voxel2triangles_gpu,cuda.to_device(minimums[0].astype(np.float32)),cuda.to_device(domain_widths[0].astype(np.int64)),triangles

def GPU_32bit_GVD_test(points,triangles,threads_per_block,candidate_triangles,voxel2triangles,minimums,spacing,domain_widths):
    return point2mesh_via_gpu32bit(points,triangles,voxel2triangles,candidate_triangles,minimums,spacing,domain_widths,threads_per_block)

def GPU_64bit_GVD_initialize(spacing,mesh,expansion_factor):
    candidate_triangles,voxel2triangles,minimums,domain_widths=make_pruned_point_to_mesh_ragged_array(spacing,[mesh],expansion_factor)
    candidates_gpu=cuda.to_device(candidate_triangles[0].astype(np.int64))
    voxel2triangles_gpu=cuda.to_device(voxel2triangles[0].astype(np.int64))
    triangles=cuda.to_device(mesh.triangles.reshape((len(mesh.triangles)*9,)).astype(np.float64))
    return np.float64(spacing),candidates_gpu,voxel2triangles_gpu,cuda.to_device(minimums[0].astype(np.float64)),cuda.to_device(domain_widths[0].astype(np.int64)),triangles

def GPU_64bit_GVD_test(points,triangles,threads_per_block,candidate_triangles,voxel2triangles,minimums,spacing,domain_widths):
    return point2mesh_via_gpu64bit(points,triangles,voxel2triangles,candidate_triangles,minimums,spacing,domain_widths,threads_per_block)

def GPU_32bit_brute_force_initialize(mesh):
    return cuda.to_device(mesh.triangles.reshape((len(mesh.triangles)*9,)).astype(np.float32))

def GPU_32bit_brute_force_test(points,triangles,threads_per_block):
    return brute_force_points2mesh(points,triangles,threads_per_block)

@njit
def point2mesh_via_cpu_more_parallel(points,triangles,voxels2triangles,candidate_triangles,minimums,spacing,domain_width,chunk_size):
    set_parallel_chunksize(chunk_size)
    return point2mesh_via_cpu_parallel(points,triangles,voxels2triangles,candidate_triangles,minimums,spacing,domain_width)

def point2mesh_cpu_pool_initializer(triangles,voxels,candidates,minimums,spacing,widths,tol):
    global g_triangles,g_voxels,g_candidates,g_minimums,g_spacing,g_widths,g_tol
    g_triangles=triangles
    g_voxels=voxels
    g_candidates=candidates
    g_minimums=minimums
    g_spacing=spacing
    g_widths=widths
    g_tol=tol

@njit
def point2mesh_cpu_pool_worker(point):
    return point2mesh_distance_cpu(point,g_triangles,g_voxels,g_candidates,g_minimums,g_spacing,g_widths,g_tol)

def point2mesh_via_gpu32bit(points,triangles,voxels2triangles,candidate_triangles,minimums,spacing,domain_width,threads_per_block=1024):
    queries=cuda.to_device(points.astype(np.float32))
    distance=np.empty(len(points),dtype=np.float32)
    blocks_per_grid=(len(points)+(threads_per_block-1))//threads_per_block
    kernel_signature=(blocks_per_grid,threads_per_block)
    point2mesh_via_gpu_kernel32bit[kernel_signature](queries,triangles,voxels2triangles,candidate_triangles,minimums,spacing,domain_width,distance)
    return distance

point2mesh_via_gpu_kernel32bit=point2mesh_via_gpu_kernel

def point2mesh_via_gpu64bit(points,triangles,voxels2triangles,candidate_triangles,minimums,spacing,domain_width,threads_per_block=1024):
    queries=cuda.to_device(points)
    distance=np.empty(len(points))
    blocks_per_grid=(len(points)+(threads_per_block-1))//threads_per_block
    kernel_signature=(blocks_per_grid,threads_per_block)
    point2mesh_via_gpu_kernel64bit[kernel_signature](queries,triangles,voxels2triangles,candidate_triangles,minimums,spacing,domain_width,distance)
    return distance

@cuda.jit()
def point2mesh_via_gpu_kernel64bit(points,triangles,voxels2triangles,candidate_triangles,minimums,spacing,domain_width,output):
    tol=nb_float64(1e-12)
    thread_index=cuda.grid(1)#this is the query number
    if thread_index>=len(points):
        return
    point=points[thread_index]
    xid=nb_int64((point[0]-minimums[0])//spacing)
    yid=nb_int64((point[1]-minimums[1])//spacing)
    zid=nb_int64((point[2]-minimums[2])//spacing)
    if xid>=domain_width[0] or xid<0 or yid>=domain_width[1] or yid<0 or zid>=domain_width[2] or zid<0:
        #point is outside the grid on which we have sets of triangles precomputed
        output[thread_index]=nb_float64(np.inf)
    else:
        array_pos=threeD_multi_to_1D(xid,yid,zid,domain_width)
        triangle_id_start=voxels2triangles[array_pos]
        triangle_id_end=voxels2triangles[array_pos+1]
        triangle_ids=candidate_triangles[triangle_id_start:triangle_id_end]
        mindistsquared=nb_float64(np.inf)
        for triangle_id in triangle_ids:
            idx=triangle_id*9
            candidate_dist_squared=fabs(cuda_point_to_triangle_squared_distance(triangles[idx:idx+9],point,tol))
            if candidate_dist_squared<mindistsquared:
                mindistsquared=candidate_dist_squared
        output[thread_index]=sqrt(mindistsquared)
    return