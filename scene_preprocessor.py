import numpy as np
import torch
import MinkowskiEngine as ME

NUMBER_OF_NUSCENES_LABEL_TYPES = 32

class ScenePreprocessor:    

    @staticmethod
    def filter_out_scene(coords, labels):
        is_valid = (labels != 0)  # Filter out invalid labels
        valid_coords, valid_labels = coords[is_valid], labels[is_valid]  # Filter out invalid coords
    
        return valid_coords, valid_labels
    
    @staticmethod
    def one_hot_encode(label):
        return torch.nn.functional.one_hot(label.long(), NUMBER_OF_NUSCENES_LABEL_TYPES).float()
    
    @staticmethod
    def accumulate_voxel(coords, labels, voxel_size):  # sequence_of_labels
        concatenated_coords = torch.cat(coords, dim=0)
        concatenated_labels = torch.cat(labels, dim=0)

        quantized_scene = ME.SparseTensor(
            features=concatenated_labels,
            coordinates=ME.utils.batched_coordinates([concatenated_coords / voxel_size]),
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_SUM
        )
        
        accumulated_coord, accumulated_label = quantized_scene.decomposed_coordinates_and_features
        accumulated_coord = accumulated_coord[0] * voxel_size
        accumulated_label = accumulated_label[0].argmax(dim=-1)
        
        accumulated_coord, accumulated_label = accumulated_coord.numpy(force=True), accumulated_label.numpy(force=True)  # Convert torch tensor to numpy array
        
        return accumulated_coord, accumulated_label
    
    @staticmethod
    def transform_to_fpv_pose(extrinsic, height):
        # extrinsic = np.linalg.inv(extrinsic)
                
        """
        extrinsic[:3, :3] = np.array(  # OpenGL convention: flip y and z axis
            [[1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]]
        ) @ np.array(   # 90 degree rotation around z axis -> BEV to camera view
            [[0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]]
        ) @ np.array(  # 90 degree rotation around y axis
            [[1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]]
        ) @ extrinsic[:3, :3]
        """
        
        extrinsic[:3, :3] = np.array(
            [[0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]]
        ) @ extrinsic[:3, :3]
        
        extrinsic[0, 3], extrinsic[1, 3], extrinsic[2, 3] = -extrinsic[1, 3], -extrinsic[2, 3], extrinsic[0, 3]  # Reset axis            
        
        return extrinsic
    
    @staticmethod
    def transform_to_bev_pose(extrinsic, height):
        # extrinsic = np.linalg.inv(extrinsic)
        
        # Set view to bird eye view
        extrinsic[:3, :3] = np.array(
            [[1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]]
        ) @ extrinsic[:3, :3]
        
        extrinsic[1, 3] = -extrinsic[1, 3]
        extrinsic[2, 3] = height  # Set height
    
        return extrinsic
    
    @staticmethod
    def create_trajectory(coord, label, extrinsic, intrinsic, height, visualization_type):
        
        
        if visualization_type == "FPV":
            transform_pose = ScenePreprocessor.transform_to_fpv_pose
        elif visualization_type == "BEV":
            transform_pose = ScenePreprocessor.transform_to_bev_pose
        else:
            raise ValueError(f"Invalid visualization type: {visualization_type}")
        
        extrinsic = transform_pose(extrinsic, height)
        
        trajectory = {
            "coord": coord,
            "label": label,
            "extrinsic": extrinsic,
            "intrinsic": intrinsic
        }
        
        
        return trajectory
    