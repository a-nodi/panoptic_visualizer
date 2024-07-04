import cv2
from tqdm import tqdm
import open3d as o3d
import numpy as np


class Visualizer:
    def __init__(self, configs):
        self.configs = configs
        self.colors = np.array([
            [0  , 0  , 0, 255],
            [100, 150, 245, 255],
            [100, 230, 245, 255],
            [30, 60, 150, 255],
            [80, 30, 180, 255],
            [100, 80, 250, 255],
            [255, 30, 30, 255],
            [255, 40, 200, 255],
            [150, 30, 90, 255],
            [255, 0, 255, 255],
            [255, 150, 255, 255],
            [75, 0, 75, 255],
            [175, 0, 75, 255],
            [255, 200, 0, 255],
            [255, 120, 50, 255],
            [0, 175, 0, 255],
            [135, 60, 0, 255],
            [150, 240, 80, 255],
            [255, 240, 150, 255],
            [255, 0, 0, 255]]
        ) / 255.0  # TODO: put out to color map 

    def wrapup_scenes(self, sequence_of_scenes, voxel_size):
        wraped_trajectory = []
        
        for trajectory_element in sequence_of_scenes:
            # Wrap up to Open3D camera parameters
            parameter = o3d.camera.PinholeCameraParameters()
            parameter.intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=self.configs["resolution"][0],
                height=self.configs["resolution"][1],
                intrinsic_matrix=trajectory_element[0]["intrinsic"]
            )
            
            # Wrap up to Open3D geometry        
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(trajectory_element["coords"])
            pcd.colors = o3d.utility.Vector3dVector(self.colors[trajectory_element["labels"]][:, :3])
            
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, 0.95 * voxel_size)
            
            wraped_trajectory.append(
                {
                    "voxel_grid": voxel_grid,
                    "parameter": parameter    
                }
            )
            
        return wraped_trajectory
    
    def visualize(self, wraped_up_scenes, voxel_size):
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=self.configs["resolution"][0], height=self.configs["resolution"][1])
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        ctr = vis.get_view_control()
        
        with tqdm(range(0, len(wraped_up_scenes)), disable=False) as pbar:
            pbar.set_description("Rendering image")
            for scene in wraped_up_scenes:
                
                vis.clear_geometries()
                vis.add_geometry(scene["voxel_grid"])

                ctr.convert_from_pinhole_camera_parameters(
                        scene["parameter"],
                        # allow_arbitrary=True
                )
                
                vis.poll_events()
                vis.update_renderer()
                
                image = vis.capture_screen_float_buffer(do_render=True)
                pbar.update(1)
                
        vis.destroy_window()
        return
        