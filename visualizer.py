import cv2
import os.path as osp
from tqdm import tqdm
import open3d as o3d
import numpy as np
from nuscenes.utils.color_map import get_colormap

class Visualizer:
    def __init__(self, configs, verbose=False):
        self.configs = configs
        
        self.color_map = self.load_color_map()
        self.verbose = verbose
    @staticmethod
    def load_color_map():
        _color_map = get_colormap()
        colors = [color for color in _color_map.values()]
        
        color_map = np.zeros((len(colors), 3))
        for i, color in enumerate(colors):
            color_map[i] = np.array(color) / 255.0
        
        return color_map
    
    @staticmethod
    def get_custom_instrinsic(resolution):
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            width=resolution[0], 
            height=resolution[1]
        )
        
        ctr = vis.get_view_control()
        intrinsic = np.copy(ctr.convert_to_pinhole_camera_parameters().intrinsic.intrinsic_matrix)
        vis.destroy_window()
        
        return intrinsic

    def wrapup_scenes(self, sequence_of_scenes, voxel_size):
        wraped_trajectory = []
        
        for trajectory_element in sequence_of_scenes:
            # Wrap up to Open3D camera parameters
            parameter = o3d.camera.PinholeCameraParameters()
            parameter.intrinsic.set_intrinsics(
                width=self.configs["resolution"][0],
                height=self.configs["resolution"][1],
                fx=trajectory_element["intrinsic"][0, 0],
                fy=trajectory_element["intrinsic"][1, 1],
                cx=trajectory_element["intrinsic"][0, 2],
                cy=trajectory_element["intrinsic"][1, 2]
            )
            parameter.extrinsic = trajectory_element["extrinsic"]
            
            # Wrap up to Open3D geometry        
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(trajectory_element["coord"])
            pcd.colors = o3d.utility.Vector3dVector(self.color_map[trajectory_element["label"]])
            
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
        # opt = vis.get_render_option()
        # opt.background_color = np.asarray([0, 0, 0])
        ctr = vis.get_view_control()
        sequence_of_image = []
        
        # Video fream size
        out = cv2.VideoWriter(
            filename=osp.join(self.configs["output_path"], self.configs["output_name"]),
            fourcc=cv2.VideoWriter_fourcc(*self.configs["output_codec"]), 
            fps=self.configs["fps"],
            frameSize=self.configs["resolution"]
        )   
        
        # Render images from the scenes
        with tqdm(range(0, len(wraped_up_scenes)), disable=not self.verbose) as pbar:
            pbar.set_description("Rendering scenes")
            for scene in wraped_up_scenes:
                
                vis.clear_geometries()
                vis.add_geometry(scene["voxel_grid"])

                """
                ctr.convert_from_pinhole_camera_parameters(
                        scene["parameter"],
                        allow_arbitrary=True
                )
                """
                
                _extrinsic = ctr.convert_to_pinhole_camera_parameters().extrinsic
                extrinsic = scene["parameter"].extrinsic
                
                vis.poll_events()
                vis.update_renderer()
                
                image = vis.capture_screen_float_buffer(do_render=True)
                image = (255 * np.asarray(image)).astype(np.uint8)
                sequence_of_image.append(image)
                out.write(image)
                pbar.update(1)

        vis.destroy_window()
        out.release()
        print(f"Video saved at {osp.join(self.configs['output_path'], self.configs['output_name'])}")
        
        return
        