# -*- coding: utf-8 -*-
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def mesh2pcd(triangle_mesh):
    pcd = o3d.geometry.PointCloud()
    pcd.points = triangle_mesh.vertices
    pcd.colors = triangle_mesh.vertex_colors
    pcd.normals = triangle_mesh.vertex_normals
    return pcd

def mesh2pcd_test(triangle_mesh):
    pcd = triangle_mesh.sample_points_uniformly(number_of_points = 5000)
    return pcd

def custom_draw_geometry(pcd, edit=False, save_name = None):
    # The following code achieves the same effect as:
    # o3d.visualization.draw_geometries([pcd])
    if edit:
        vis = o3d.visualization.VisualizerWithEditing()
    else:
        vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720)
    vis.add_geometry(pcd)
    vis.run()
    if save_name is not None:
        vis.capture_screen_image(save_name, True)
    vis.destroy_window()
    if edit:
        return vis.get_picked_points()
    else:
        return None

def custom_draw_geometry_with_key_callback(pcd, save_name=None):


    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        if save_name is None:
            return False
        return vis


    def load_render_option(vis):
        vis.get_render_option().load_from_json(
            "../../TestData/renderoption.json")
        return False


    def capture_depth(vis):
        depth = vis.capture_depth_float_buffer()
        plt.imshow(np.asarray(depth), cmap='gray')
        plt.show()
        return False


    def capture_image(vis):
        vis = change_background_to_black(vis)
        vis.capture_screen_image(save_name, True)
        return False


    key_to_callback = {}
    key_to_callback[ord("K")] = change_background_to_black
    key_to_callback[ord("R")] = load_render_option
    key_to_callback[ord(",")] = capture_depth
    key_to_callback[ord(".")] = capture_image
    o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)

def numpy2open3d_pcl(pcl,flag = False):
    pcl_new = o3d.geometry.PointCloud()
    if(flag):
        pcl = pcl[:,[2,1,0]]
    pcl_new.points = o3d.utility.Vector3dVector(pcl)
    return pcl_new

def invTrans(mat):
    P = mat[:3,:3]
    v = mat[:3,3]
    P_inv = np.linalg.inv(P)
    inv_mat = np.hstack((P_inv,-P_inv.dot(v)[:,None]))
    inv_mat = np.vstack((inv_mat,np.array([0,0,0,1])))
    return inv_mat

def angles2rotmat(x,y,z):
    
    rot_x = np.array([[ 1,         0,          0],
                      [ 0, np.cos(x), -np.sin(x)],
                      [ 0, np.sin(x),  np.cos(x)]])
    
    rot_y = np.array([[  np.cos(y), 0, np.sin(y)],
                      [          0, 1,         0],
                      [ -np.sin(y), 0, np.cos(y)]])
    
    rot_z = np.array([[ np.cos(z), -np.sin(z), 0],
                      [ np.sin(z),  np.cos(z), 0],
                      [         0,          0, 1]])
    
    rot = rot_x @ rot_y @ rot_z
    
    return rot

def rotate(pcl,R):
    points = np.array(pcl.points)
    points = (R@(points.T)).T
    pcl.points = o3d.utility.Vector3dVector(points)
    return pcl

def translate(pcl,transVec):
    points = np.array(pcl.points)
    points += np.array(transVec)
    pcl.points = o3d.utility.Vector3dVector(points)
    return pcl

def get_Rt_from_3_4_projection_matrix(P):
    R = P[:3,:3]
    t = P[:,3]
    return R,t

def transform_pcl_to_new_frame(pcl,R):
    # R to go from present frame to new frame
    points = np.array(pcl.points)
    points = points@R
    pcl.points = o3d.utility.Vector3dVector(points)
    return pcl



