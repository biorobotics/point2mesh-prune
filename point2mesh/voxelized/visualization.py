'''
code to visualize things related to voxelized representations of GVD of triangle meshes
'''
import numpy as np
import trimesh

from point2mesh.util.ArrayTricks import threeD_multi_to_1D,index_to_point

def add_rss(center,rotation_matrix,half_lengths,radius,scene=None):
    '''
    Parameters: center : (3,) float array
                    position of center of RSS given in world coordinates
                rotation_matrix : (3,3) float array
                    rotation matrix mapping world frame vectors to body frame vectors
                half_lengths : (2,) float array
                    (x,y) rectangle half lengths (x and y are the BODY axes)
                radius : float
                    the radius of the RSS
    '''
    if scene is None:
        scene=trimesh.Scene()
    #central sphere
    sphere=trimesh.primitives.Sphere(radius=2*radius,center=center)

    #box
    body_to_world=np.eye(4)
    body_to_world[:3,3]=center
    body_to_world[:3,:3]=rotation_matrix.T#this is the transformation that takes body frame positions to world frame positions
    box=trimesh.primitives.Box(extents=2*np.concatenate([half_lengths,[radius]]),transform=body_to_world.copy())
    
    #cylinders
    tf=body_to_world.copy()
    tf[:3,3]+=rotation_matrix.T@np.array([0,half_lengths[1],0])
    tf[:3,:3]=rotation_matrix[[2,0,1]]
    xplus=trimesh.primitives.Capsule(radius=radius,height=2*half_lengths[0],transform=tf.copy())

    scene.add_geometry(sphere)
    scene.add_geometry(box)
    scene.add_geometry(xplus)
    scene.add_geometry(trimesh.creation.axis())
    return scene

def visualize_triangle_candidates(mesh,hashmap,minimums,spacing,int_type,point):
    edges=trimesh.load_path(mesh.vertices[mesh.edges_unique])

    integer_coordinates=np.floor((point-minimums)/spacing).astype(int_type)
    triangle_ids=hashmap[tuple(integer_coordinates)]

    color_backup=mesh.visual.face_colors.copy()
    mesh.visual.face_colors[:,-1]=100
    intersection=mesh.submesh([triangle_ids],append=True)
    intersection.visual.face_colors=np.array([255,0,0,255])
    mesh.visual.face_colors[triangle_ids]=np.array([255,0,0,255])
    tf_putting_corner_at_0=np.eye(4)
    tf_putting_corner_at_0[:3,3]=spacing/2*np.ones((3,))
    voxel=trimesh.primitives.Box(extents=spacing*np.ones((3,)),transform=tf_putting_corner_at_0)
    tf=np.eye(4)
    tf[:3,3]=integer_coordinates*spacing+minimums
    voxel.apply_transform(tf)
    voxel.visual.face_colors=np.array([0,0,255,50])

    sphere=trimesh.primitives.Sphere(radius=.001,center=point)
    sphere.visual.face_colors=np.array([0,255,0,255])

    scene=trimesh.Scene([mesh,edges,voxel,sphere,intersection])
    scene.show()
    mesh.visual.face_colors=color_backup

def visualize_triangle_candidates_ragged(mesh,candidate_triangles,voxel2triangle,minimums,domain_width,spacing,int_type,point):
    edges=trimesh.load_path(mesh.vertices[mesh.edges_unique])

    integer_coordinates=np.floor((point-minimums)/spacing).astype(int_type)
    array_pos=threeD_multi_to_1D(*integer_coordinates,domain_width)
    s=voxel2triangle[array_pos]
    e=voxel2triangle[array_pos+1]
    triangle_ids=candidate_triangles[s:e].copy_to_host()

    color_backup=mesh.visual.face_colors.copy()
    mesh.visual.face_colors[:,-1]=100
    intersection=mesh.submesh([triangle_ids],append=True)
    intersection.visual.face_colors=np.array([255,0,0,255])
    mesh.visual.face_colors[triangle_ids]=np.array([255,0,0,255])
    tf_putting_corner_at_0=np.eye(4)
    tf_putting_corner_at_0[:3,3]=spacing/2*np.ones((3,))
    voxel=trimesh.primitives.Box(extents=spacing*np.ones((3,)),transform=tf_putting_corner_at_0)
    tf=np.eye(4)
    tf[:3,3]=integer_coordinates*spacing+minimums
    voxel.apply_transform(tf)
    voxel.visual.face_colors=np.array([0,0,255,50])

    sphere=trimesh.primitives.Sphere(radius=.001,center=point)
    sphere.visual.face_colors=np.array([0,255,0,255])

    scene=trimesh.Scene([mesh,edges,voxel,sphere,intersection])
    scene.show()
    mesh.visual.face_colors=color_backup

def cpu_visualize_triangle_candidates_ragged(mesh,candidate_triangles,voxel2triangle,minimums,domain_width,spacing,int_type,point):
    edges=trimesh.load_path(mesh.vertices[mesh.edges_unique])

    integer_coordinates=np.floor((point-minimums)/spacing).astype(int_type)
    array_pos=threeD_multi_to_1D(*integer_coordinates,domain_width)
    s=voxel2triangle[array_pos]
    e=voxel2triangle[array_pos+1]
    triangle_ids=candidate_triangles[s:e]

    color_backup=mesh.visual.face_colors.copy()
    mesh.visual.face_colors[:,-1]=100
    intersection=mesh.submesh([triangle_ids],append=True)
    intersection.visual.face_colors=np.array([255,0,0,255])
    mesh.visual.face_colors[triangle_ids]=np.array([255,0,0,255])
    tf_putting_corner_at_0=np.eye(4)
    tf_putting_corner_at_0[:3,3]=spacing/2*np.ones((3,))
    voxel=trimesh.primitives.Box(extents=spacing*np.ones((3,)),transform=tf_putting_corner_at_0)
    tf=np.eye(4)
    tf[:3,3]=integer_coordinates*spacing+minimums
    voxel.apply_transform(tf)
    voxel.visual.face_colors=np.array([0,0,255,50])

    sphere=trimesh.primitives.Sphere(radius=.001,center=point)
    sphere.visual.face_colors=np.array([0,255,0,255])

    scene=trimesh.Scene([mesh,edges,voxel,sphere,intersection])
    scene.show()
    mesh.visual.face_colors=color_backup

def highlight_triangles(mesh,triangle_ids,scene=None):
    edges=trimesh.load_path(mesh.vertices[mesh.edges_unique])

    color_backup=mesh.visual.face_colors.copy()
    mesh.visual.face_colors[:,-1]=100
    intersection=mesh.submesh([triangle_ids],append=True)
    intersection.visual.face_colors=np.array([255,0,0,255])
    mesh.visual.face_colors[triangle_ids]=np.array([255,0,0,255])
    if scene is None:
        scene=trimesh.Scene([mesh,edges,intersection])
    else:
        scene.add_geometry(mesh)
        scene.add_geometry(edges)
        scene.add_geometry(intersection)
    scene.show()
    mesh.visual.face_colors=color_backup

def highlight_triangles_multiple_groups(mesh,triangle_id_groups,colors,scene=None):
    edges=trimesh.load_path(mesh.vertices[mesh.edges_unique])

    color_backup=mesh.visual.face_colors.copy()
    mesh.visual.face_colors[:,-1]=100
    if scene is None:
        scene=trimesh.Scene([mesh,edges])
    else:
        scene.add_geometry([mesh,edges])
    for i,triangle_ids in enumerate(triangle_id_groups):
        intersection=mesh.submesh([triangle_ids],append=True)
        intersection.visual.face_colors=colors[i]
        mesh.visual.face_colors[triangle_ids]=colors[i]
        scene.add_geometry(intersection)
    
    scene.show()
    mesh.visual.face_colors=color_backup


def compare_ragged_and_dictionary_at_point(mesh,point,minimums,spacing,domain_width,voxel2triangle,candidate_triangles,hashmap,int_type):
    integer_coordinates=np.floor((point-minimums)/spacing).astype(int_type)
    compare_ragged_and_dictionary(mesh,integer_coordinates,minimums,spacing,domain_width,voxel2triangle,candidate_triangles,hashmap)

def compare_ragged_and_dictionary(mesh,integer_index,minimums,spacing,domain_width,voxel2triangle,candidate_triangles,hashmap):
    dictionary=set(hashmap[tuple(integer_index)])
    array_pos=threeD_multi_to_1D(*integer_index,domain_width)
    s=voxel2triangle[array_pos]
    e=voxel2triangle[array_pos+1]
    ragged=set(candidate_triangles[s:e].copy_to_host())

    missing=list(dictionary-ragged)
    extra=list(ragged-dictionary)
    shared=list(dictionary.intersection(ragged))
    triangle_id_groups=[missing,shared,extra]

    red=np.array([255,0,0,255])
    green=np.array([0,255,0,255])
    blue=np.array([0,0,255,255])
    colors=[red,green,blue]

    edges=trimesh.load_path(mesh.vertices[mesh.edges_unique])

    color_backup=mesh.visual.face_colors.copy()
    mesh.visual.face_colors[:,-1]=100
    scene=trimesh.Scene([mesh,edges])
    for i,triangle_ids in enumerate(triangle_id_groups):
        if len(triangle_ids)>0:
            intersection=mesh.submesh([triangle_ids],append=True)
            intersection.visual.face_colors=colors[i]
            mesh.visual.face_colors[triangle_ids]=colors[i]
            scene.add_geometry(intersection)

    #box
    shift=integer_index*spacing+minimums
    tf=np.eye(4)
    tf[:3,3]=spacing/2+shift
    voxel=trimesh.primitives.Box(extents=spacing*np.ones((3,)),transform=tf)
    voxel.visual.face_colors=np.array([0,0,255,100])
    scene.add_geometry(voxel)

    #centered sphere
    sphere=trimesh.primitives.Sphere(radius=.001,center=np.zeros((3,)))
    sphere.visual.face_colors=np.array([0,255,0,255])
    scene.add_geometry(sphere,transform=tf)
    
    
    scene.show()
    mesh.visual.face_colors=color_backup

def visualize_voxels(mesh,hashmap,minimums,spacing,int_type,nsamples):
    samples=mesh.bounding_box_oriented.sample_volume(nsamples)
    integer_coordinates=np.floor((samples-minimums)/spacing).astype(int_type)

    maximums=mesh.bounding_box.extents/2+mesh.bounding_box.transform[:3,3]
    gridvectors=tuple(np.arange(minimums[i],maximums[i],spacing) for i in range(3))

    edges=trimesh.load_path(mesh.vertices[mesh.edges_unique])
    color_backup=mesh.visual.face_colors.copy()
    mesh.visual.face_colors[:,-1]=50

    tf_putting_corner_at_0=np.eye(4)
    tf_putting_corner_at_0[:3,3]=spacing/2*np.ones((3,))
    good_voxel=trimesh.primitives.Box(extents=spacing*np.ones((3,)),transform=tf_putting_corner_at_0)
    good_voxel.visual.face_colors=np.array([0,0,255,100])
    bad_voxel=trimesh.primitives.Box(extents=spacing*np.ones((3,)),transform=tf_putting_corner_at_0)
    bad_voxel.visual.face_colors=np.array([255,0,0,100])

    sphere=trimesh.primitives.Sphere(radius=.001,center=np.zeros((3,)))
    sphere.visual.face_colors=np.array([0,255,0,255])

    

    scene=trimesh.Scene([mesh,edges])

    for i,pt in enumerate(samples):
        tf=np.eye(4)
        ic=integer_coordinates[i]
        tf[:3,3]=np.array([gridvectors[j][ic[j]] for j in range(3)])
        if tuple(ic) in hashmap:
            scene.add_geometry(good_voxel,transform=tf)
        else:
            scene.add_geometry(bad_voxel,transform=tf)
        tf_query=np.eye(4)
        tf_query[:3,3]=pt
        scene.add_geometry(sphere,transform=tf_query)
    scene.show()
    mesh.visual.face_colors=color_backup

def visualize_voxel_list(mesh,expansion_factor,spacing,voxel_list):
    minimums=(-mesh.bounding_box.extents/2+mesh.bounding_box.transform[:3,3])*expansion_factor
    maximums=(mesh.bounding_box.extents/2+mesh.bounding_box.transform[:3,3])*expansion_factor
    gridvectors=tuple(np.arange(minimums[i],maximums[i],spacing) for i in range(3))
    domain_width=np.array([len(gv) for gv in gridvectors])      
    edges=trimesh.load_path(mesh.vertices[mesh.edges_unique])
    color_backup=mesh.visual.face_colors.copy()
    mesh.visual.face_colors[:,-1]=50

    tf_putting_corner_at_0=np.eye(4)
    tf_putting_corner_at_0[:3,3]=spacing/2*np.ones((3,))
    good_voxel=trimesh.primitives.Box(extents=spacing*np.ones((3,)),transform=tf_putting_corner_at_0)
    good_voxel.visual.face_colors=np.array([0,0,255,100])

    sphere=trimesh.primitives.Sphere(radius=.001,center=np.zeros((3,)))
    sphere.visual.face_colors=np.array([0,255,0,255])

    

    scene=trimesh.Scene([mesh,edges])

    for idx in voxel_list:
        tf=np.eye(4)
        ic=index_to_point(idx,domain_width)
        tf[:3,3]=np.array([gridvectors[j][ic[j]] for j in range(3)])
        scene.add_geometry(good_voxel,transform=tf)
    scene.show()
    mesh.visual.face_colors=color_backup